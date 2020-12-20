#pragma once

#include "log.h"

class Environment {
    public:
        Environment(const char *fileName) {
            /*
             * The map file is parsed and the data structures are filled out. The map
             * files posit 1-based indexing (imported from MATLAB), but the C++ code
             * posits standard 0-based indexing. That's why the data structures are
             * fill out with values that are not necessarily the same as those in the
             * map files.
             */

            auto getPair = [](std::string str) {
                std::size_t pos = str.find(',');
                assert(pos != std::string::npos);
                int first = stoi(str.substr(0, pos));
                int second = stoi(str.substr(pos + 1));
                return std::make_pair(first, second);
            };

            std::ifstream file(fileName);
            assert(file.good());

            std::string line;
            std::pair<int, int> pair;

            getline(file, line);
            assert(line == "N");

            getline(file, line);
            pair = getPair(line);
            mapX = pair.first;
            mapY = pair.second;

            getline(file, line);
            assert(line == "C");

            getline(file, line);
            collisionThreshold = stoi(line);

            getline(file, line);
            assert(line == "R");

            getline(file, line);
            pair = getPair(line);
            robotX = pair.first - 1;    // 0-based indexing
            assert(robotX >= 0 && robotX < mapX);
            robotY = pair.second - 1;   // 0-based indexing
            assert(robotY >= 0 && robotY < mapY);

            getline(file, line);
            assert(line == "T");

            getline(file, line);
            pair = getPair(line);
            targetX = pair.first - 1;    // 0-based indexing
            assert(targetX >= 0 && targetY < mapX);
            targetY = pair.second - 1;   // 0-based indexing
            assert(targetY >= 0 && targetY < mapY);

            getline(file, line);
            assert(line == "M");

            map = new int*[mapX];
            for (int i = 0; i < mapX; i++) {
                map[i] = new int[mapY];
            }

            for (int i = 0; i < mapX; i++) {
                getline(file, line);
                std::stringstream row(line);
                for (int j = 0; j < mapY; j++) {
                    assert(row.good());
                    std::string substr;
                    getline(row, substr, ',');
                    map[i][j] = stoi(substr);
                    assert(map[i][j] >= 0);
                }
                assert(!row.good());
            }

            file.close();

            info("Map is created. Map: X=%d, Y=%d, Robot: X=%d, Y=%d, Target: X=%d, Y=%d, Collision Threshold: %d", 
                    mapX, mapY, robotX+1, robotY+1, targetX+1, targetY+1, collisionThreshold);
        }

        ~Environment() {
            for (int i = 0; i < mapX; i++) {
                delete[] map[i];
            }
            delete[] map;
        }

        int getMapX() const { return mapX; }
        int getMapY() const { return mapY; }
        int getRobotX() const { return robotX; }
        int getRobotY() const { return robotY; }
        int getCellCost(int x, int y) const { return map[x][y]; }
        int getTargetX() const { return targetX; }
        int getTargetY() const { return targetY; }

        bool isFree(int x, int y) const {
            volatile unsigned long long counter = 10000000;
            while (counter--);
            return (map[x][y] < collisionThreshold);
        }

        bool isValid(int x, int y) const {
            return ((x>=0) && (x<mapX) && (y>=0) && (y<mapY));
        }

    private:
        int mapX, mapY;
        int **map;
        int collisionThreshold;
        int robotX, robotY;
        int targetX, targetY;
};

