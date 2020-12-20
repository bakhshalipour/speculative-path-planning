CXX = g++
CFLAGS = -std=c++17
LFLAGS = -lpthread

.PHONY: clean

all: speculative_astar.out

%.out: %.cpp perf.c
	$(CXX) $(CFLAGS) $^ -o $@ $(LFLAGS)

clean:
	rm *.out
