#include <iostream>
#include <fstream>
#include <queue>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <tuple>
#include <unordered_set>

#include "env.h"
#include "perf.h"

// #define NASSERT  // Uncomment to disable asserts

using namespace std;

// The largest acceptable map
#define MAX_X   12000
#define MAX_Y   12000

// For avoiding livelock when looking for nodes for speculative parallelism
#define LIVELOCK_COUNTER    4

int MAX_THREADS;
int EPSILON;    // Weighted A*
bool DO_SPECULATION;

typedef std::pair<int /*X*/, int /*Y*/> Pair;
typedef std::tuple<double /*Time*/, int /*X*/, int /*Y*/> Tuple;

std::vector<Pair> path; // Final path
std::vector<Tuple> expansions;   // The expanded nodes + time
std::vector<Tuple> speculativeCollisionCheckings;    // The explored node + time
std::vector<Tuple> usages;

Environment *env;
const int NUM_2D_DIRS = 8;
double gVals[MAX_X][MAX_Y];
bool closedList[MAX_X][MAX_Y] = {0};
long double baseTime;

enum CELL_STATE {UNKNOWN, FREE_SPEC, FREE_NONSPEC, COLLISION_SPEC, COLLISION_NONSPEC};
CELL_STATE cellsState[MAX_X][MAX_Y];

typedef unsigned long long StatCounter;
StatCounter __TotalExpansions = 0, __StraightExpansions = 0;
StatCounter __SpeculativeUsefull = 0, __SpeculativeThreads = 0;
StatCounter __TotalSpecAttempts = 0, __TotalSpecLoopIterations = 0;
StatCounter __TotalActiveThreads = 0, __TotalActiveNonSpecThreads = 0, __TotalActiveSpecThreads = 0;

void *checkCollisionNonSpec(void *args) {
    std::vector<Pair>* v = static_cast<std::vector<Pair> *>(args);

    for (auto p : *v) {
        int x = p.first, y = p.second;
        if (env->isFree(x, y))  cellsState[x][y] = FREE_NONSPEC;
        else cellsState[x][y] = COLLISION_NONSPEC;
    }

    free(args);
    return NULL;
}

void *checkCollisionSpec(void *args) {
    std::vector<Pair>* v = static_cast<std::vector<Pair> *>(args);
    assert(v->size() == 1);

    for (auto p : *v) {
        int x = p.first, y = p.second;
        if (env->isFree(x, y))  cellsState[x][y] = FREE_SPEC;
        else cellsState[x][y] = COLLISION_SPEC;
    }

    free(args);
    return NULL;
}

void astar() {

    // 8-connected graph
    int dX[NUM_2D_DIRS] = {-1, -1, -1,  0,  0,  1, 1, 1};
    int dY[NUM_2D_DIRS] = {-1,  0,  1, -1,  1, -1, 0, 1};

    pthread_t *threadId = new pthread_t[MAX_THREADS];

    struct Node {
        int x;
        int y;
        double g;
        double h;
        double f;
        Node *parent;

        Node(int _x, int _y, double _g, double _h, Node *_p) : x(_x), y(_y),
        g(_g), h(_h), parent(_p) {
            f = g + EPSILON * h;
        }
    };

    struct NodeComparator {
        bool operator()(const Node* left, const Node* right) {
            return left->f > right->f;
        }
    };

    std::priority_queue<Node*, std::vector<Node*>, NodeComparator> heap; // OPEN list

    int startX = env->getRobotX(), startY = env->getRobotY();
    int targetX = env->getTargetX(), targetY = env->getTargetY();

    auto getHeuristic = [targetX, targetY](int posX, int posY) -> double {
        double dist = 0;
        dist += (posX - targetX) * (posX - targetX);
        dist += (posY - targetY) * (posY - targetY);
        return sqrt(dist);
    };

    for (int i = 0; i < env->getMapX(); i++) {
        for (int j = 0; j < env->getMapY(); j++) {
            gVals[i][j] = INT_MAX;
            cellsState[i][j] = UNKNOWN;
        }
    }

    Node *start = new Node(startX, startY, 0 /*g*/, getHeuristic(startX, startY), NULL);
    heap.push(start);
    gVals[startX][startY] = 0;

    // Speculative parallelism metadata
    Node *lastNode = NULL;

    struct PairHash {
        std::size_t operator () (std::pair<int, int> const &pair) const {
            std::size_t h1 = std::hash<int>()(pair.first);
            std::size_t h2 = std::hash<int>()(pair.second);
            return h1 ^ h2;
        }
    };

    baseTime = get_etime_hw();
    int totalExpandedNodes = 0;
    int lastDirX = 0, lastDirY = 0; // Invalid motion

    while (!heap.empty()) {
        totalExpandedNodes++;
        if (totalExpandedNodes % 10000 == 0) printf("Expansion per second = %Lf, totalExpandedNodes=%d\n",
                1.0*totalExpandedNodes/(get_etime_hw() - baseTime), totalExpandedNodes);

        Node *expNode = heap.top();
        heap.pop();
        __TotalExpansions++;

        if (lastNode) {
            int currDirX = expNode->x - lastNode->x;
            int currDirY = expNode->y - lastNode->y;

            if (currDirX == lastDirX && currDirY == lastDirY) __StraightExpansions++;

            lastDirX = currDirX;
            lastDirY = currDirY;
        }

        expansions.push_back(std::make_tuple(get_etime_hw() - baseTime, expNode->x, expNode->y));

        // Re-expanding a node
        if (unlikely(closedList[expNode->x][expNode->y] == true && \
                    expNode->g >= gVals[expNode->x][expNode->y])) {
            continue;
        }

        closedList[expNode->x][expNode->y] = true;

        if (expNode->x == targetX && expNode->y == targetY) {
            Node *n = expNode;
            while (n->parent != NULL) {
                path.push_back(std::make_pair(n->x, n->y));
                n = n->parent;
            }
            path.push_back(std::make_pair(n->x, n->y));
            break;
        }

        // Neighbours for which we perform the costly collision checking.
        std::vector<int> activeNeighbours;

        // The same as activeNeighbours; just for faster search
        std::unordered_set<Pair, PairHash> pendingNodes;

        for (int i = 0; i < NUM_2D_DIRS; i++) {
            int newX = expNode->x + dX[i];
            int newY = expNode->y + dY[i];

            if (!env->isValid(newX, newY) || closedList[newX][newY] || cellsState[newX][newY] != UNKNOWN) continue;
            activeNeighbours.push_back(i);
            pendingNodes.insert(std::make_pair(newX, newY));
        }

        int threadWorkload = ceil((double)activeNeighbours.size() / MAX_THREADS);    // The number of cells every neighbour evaluates

        int activeNonSpecThreads = 0;
        int allActiveThreads = 0;
        bool visitedIndices[NUM_2D_DIRS] = {0};
        for (int i = 0; i < MAX_THREADS; i++) {
            std::vector<Pair>* args = new std::vector<Pair>();
            int currentWorkload = std::min(threadWorkload, (int)activeNeighbours.size() - i * threadWorkload); // The last thread may have smaller workload
            if (currentWorkload <= 0) break;

            for (int j = 0; j < currentWorkload; j++) {
                int index = i * threadWorkload + j;
                assert(!visitedIndices[index]);
                assert(index < activeNeighbours.size());
                assert(activeNeighbours[index] >= 0 && activeNeighbours[index] < NUM_2D_DIRS);
                visitedIndices[index] = true;

                int newX = expNode->x + dX[activeNeighbours[index]];
                int newY = expNode->y + dY[activeNeighbours[index]];
                args->push_back(std::make_pair(newX, newY));
            }

            activeNonSpecThreads++;
            allActiveThreads++;
            if (pthread_create(&threadId[i], NULL, checkCollisionNonSpec, (void*)args)) assert(false);
        }
        __TotalActiveNonSpecThreads += activeNonSpecThreads;
        __TotalActiveThreads += activeNonSpecThreads;

        // Speculatively do the computations of the next node(s)
        if (lastNode && DO_SPECULATION && activeNonSpecThreads < MAX_THREADS && !activeNeighbours.empty()) {

            int dirX = expNode->x - lastNode->x;
            int dirY = expNode->y - lastNode->y;

            dirX = expNode->x - expNode->parent->x;
            dirY = expNode->y - expNode->parent->y;

            // TODO: Consider using expNode->parent too.
            if (std::abs(dirX) <= 1 && std::abs(dirY) <= 1) {

                int specX = expNode->x, specY = expNode->y;

                int threadIndex = activeNonSpecThreads;
                int availableThreads = MAX_THREADS - activeNonSpecThreads;
                int livelockCounter = LIVELOCK_COUNTER;
                __TotalSpecAttempts++;

                while (availableThreads) {
                    __TotalSpecLoopIterations++;
                    // <specX, specY> is the expected-to-be-expanded node. We
                    // should evaluate its neighbours (and not itself).
                    specX += dirX;
                    specY += dirY;

                    // If the node itself is not valid, evaluating neighbours is
                    // meaningless. Also, continuing in this direction is not
                    // effective (further invalid nodes).
                    if (!env->isValid(specX, specY)) break;

                    for (int i = 0; i < NUM_2D_DIRS; i++) {
                        int newX = specX + dX[i];
                        int newY = specY + dY[i];

                        if (!env->isValid(newX, newY) || closedList[newX][newY] || cellsState[newX][newY] != UNKNOWN) continue;

                        // If the node is in the pendingNodes, it means that a
                        // non-speculative thread is already computing the
                        // computations.
                        if (pendingNodes.find(std::make_pair(newX, newY)) != pendingNodes.end()) continue;

                        std::vector<Pair>* args = new std::vector<Pair>();
                        args->push_back(std::make_pair(newX, newY));
                        pendingNodes.insert(std::make_pair(newX, newY));
                        __SpeculativeThreads++;
                        speculativeCollisionCheckings.push_back(std::make_tuple(get_etime_hw() - baseTime, newX, newY));
                        allActiveThreads++;
                        if (pthread_create(&threadId[threadIndex++], NULL, checkCollisionSpec, (void*)args)) assert(false);

                        availableThreads--;
                        if (availableThreads == 0) break;
                    }

                    livelockCounter--;
                    if (livelockCounter == 0) break;
                }

                __TotalActiveSpecThreads += threadIndex - activeNonSpecThreads;
                __TotalActiveThreads += threadIndex - activeNonSpecThreads;
            }

        }

        for (int i = 0; i < allActiveThreads; i++) {
            if (pthread_join(threadId[i], NULL)) assert(false);
        }

        for (int i = 0; i < NUM_2D_DIRS; i++) {
            int newX = expNode->x + dX[i];
            int newY = expNode->y + dY[i];

            if (!env->isValid(newX, newY) || closedList[newX][newY]) continue;
            assert(cellsState[newX][newY] != UNKNOWN);

            usages.push_back(std::make_tuple(get_etime_hw() - baseTime, newX, newY));

            if (cellsState[newX][newY] == COLLISION_NONSPEC) {
                continue;
            } else if (cellsState[newX][newY] == COLLISION_SPEC) {
                __SpeculativeUsefull++;
                cellsState[newX][newY] = COLLISION_NONSPEC; // Do not count multiple times
                continue;
            } else if (cellsState[newX][newY] == FREE_NONSPEC) {
            
            } else if (cellsState[newX][newY] == FREE_SPEC) {
                __SpeculativeUsefull++;
                cellsState[newX][newY] = FREE_NONSPEC;  // Do not count multiple times
            } else {
                assert(false);
            }

            double newCost = gVals[expNode->x][expNode->y] + 1;
            if (newCost < gVals[newX][newY]) {
                gVals[newX][newY] = newCost;
                heap.push(new Node(newX, newY, newCost, getHeuristic(newX, newY), expNode));
            }
        }

        lastNode = expNode;
    }
}

int main(int argc, const char **argv) {

    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << "  astar_epsilon  max_threads  do_speculation  map_file  path_file  csv_file" << std::endl;
        exit(EXIT_FAILURE);
    }

    EPSILON = atoi(argv[1]);
    MAX_THREADS = atoi(argv[2]);
    DO_SPECULATION = atoi(argv[3]);

    env = new Environment(argv[4]);
    assert(env->getMapX() <= MAX_X && env->getMapY() <= MAX_Y);

    astar();
    long double execTime = get_etime_hw() - baseTime;

    std::cout <<
        "Results" <<
        ": map_file=" << argv[4] << 
        ", EPSILON=" << EPSILON << 
        ", MAX_THREADS=" << MAX_THREADS <<
        ", DO_SPECULATION=" << DO_SPECULATION <<
        ", time=" << execTime <<
        ", StraightExpansions=" << 1.0*__StraightExpansions/__TotalExpansions <<
        ", SpeculativeUsefull=" << 1.0*__SpeculativeUsefull/__SpeculativeThreads <<
        ", AverageSpecLoopIterations=" << 1.0*__TotalSpecLoopIterations/__TotalSpecAttempts <<
        ", ActiveThreadsPerExpansion=" << 1.0*__TotalActiveThreads/__TotalExpansions <<
        ", NonSpeculativeActiveThreadsPerExpansion=" << 1.0*__TotalActiveNonSpecThreads/__TotalExpansions <<
        ", SpeculativeActiveThreadsPerExpansion=" << 1.0*__TotalActiveSpecThreads/__TotalExpansions <<
        std::endl;

    std::cout << 
        "DetailedResults" <<
        ": __StraightExpansions=" << __StraightExpansions <<
        ", __TotalExpansions=" << __TotalExpansions <<
        ", __SpeculativeUsefull=" << __SpeculativeUsefull <<
        ", __SpeculativeThreads=" << __SpeculativeThreads << 
        ", __TotalSpecLoopIterations=" << __TotalSpecLoopIterations <<
        ", __TotalSpecAttempts=" << __TotalSpecAttempts <<
        ", __TotalActiveThreads=" << __TotalActiveThreads <<
        ", __TotalActiveNonSpecThreads=" << __TotalActiveNonSpecThreads <<
        ", __TotalActiveSpecThreads=" << __TotalActiveSpecThreads<<
        std::endl;

    if (path.empty()) info("Couldn't find a path");

    // Write the path out output
    std::ofstream pathFile;
    pathFile.open(argv[5]);
    for (auto rit = path.rbegin(); rit!= path.rend(); ++rit) {
        // The output is printed consistent with the map files (i.e., 1-based indexing)
        pathFile << "<" << rit->first + 1 << ", " << rit->second + 1 << ">" << std::endl;
    }
    pathFile.close();


    // Write the CSV file
    std::ofstream csvFile;
    csvFile.open(argv[6]);
    csvFile << "Type,Time,X,Y" << std::endl;
    for (auto e : expansions) {
        auto [t, x, y] = e;
        csvFile << 0 /*Type = 0: Expansions*/ << "," << t << "," << x << "," << y << std::endl;
    }
    
    for (auto e : speculativeCollisionCheckings) {
        auto [t, x, y] = e;
        csvFile << 1/*Type = 1: Speclative Collision Checkings*/ << "," << t << "," << x << "," << y << std::endl;
    }
    
    for (auto e : usages) {
        auto [t, x, y] = e;
        csvFile << 2/*Type = 2: Usages*/ << "," << t << "," << x << "," << y << std::endl;
    }
    pathFile.close();

    return 0;
}

