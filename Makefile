CC = gcc
CXX = g++
MAKE = make

CXXFLAGS = -Wall -Ofast -fopenmp -march=native -mtune=native
CFLAGS = -Wall -Ofast -fopenmp -march=native -mtune=native
LDFLAGS = -lm
DBGFLAGS = -O0 -g -DDEBUG=1

RELTARGS = ligguf-cpp ligguf-cpp-distrib ligguf-cpp-qk ligguf-c ligguf-fast ligguf-fast-sdl
DBGTARGS = ligguf-cpp-debug ligguf-c-debug ligguf-c-profile

all: $(RELTARGS)
.PHONY: all

debug: $(DBGTARGS)
.PHONY: debug

clean:
	rm -vf $(RELTARGS) $(DBGTARGS)
	cd fast && $(MAKE) clean
.PHONY: clean

ligguf-cpp: cpp/ligguf.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

ligguf-cpp-debug: cpp/ligguf.cpp
	$(CXX) $(CXXFLAGS) $(DBGFLAGS) -o $@ $<

ligguf-cpp-distrib: cpp/ligguf_distrib.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

ligguf-cpp-qk: cpp/ligguf_qk.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

ligguf-c: c/ligguf.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

ligguf-c-debug: c/ligguf.c
	$(CC) $(CFLAGS) $(DBGFLAGS) -o $@ $< $(LDFLAGS)

ligguf-c-profile: c/ligguf.c
	$(CC) $(CFLAGS) -DDEBUG=1 -o $@ $< $(LDFLAGS)

ligguf-fast:
	cd fast && $(MAKE) APP=$@ NO_SDL=1 clean all
	mv fast/$@ ./

ligguf-fast-sdl:
	cd fast && $(MAKE) APP=$@ clean all
	mv fast/$@ ./
.PHONY: ligguf-fast ligguf-fast-sdl
