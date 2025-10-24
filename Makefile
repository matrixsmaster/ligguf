CC=gcc
CXX=g++

CXXFLAGS=-Wall -Ofast
CFLAGS=-Wall -Ofast -fopenmp
LDFLAGS=-lm
DBGFLAGS=-O0 -g -DDEBUG=1

RELTARGS=ligguf-cpp ligguf-c
DBGTARGS=ligguf-cpp-debug ligguf-c-debug ligguf-c-profile

all: $(RELTARGS)
.PHONY: all

debug: $(DBGTARGS)
.PHONY: debug

clean:
	rm -vf $(RELTARGS) $(DBGTARGS)
.PHONY: clean

ligguf-cpp: cpp/ligguf.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

ligguf-cpp-debug: cpp/ligguf_debug.cpp
	$(CXX) $(CXXFLAGS) $(DBGFLAGS) -o $@ $<

ligguf-c: c/ligguf.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

ligguf-c-debug: c/ligguf.c
	$(CXX) $(CFLAGS) $(DBGFLAGS) -o $@ $< $(LDFLAGS)

ligguf-c-profile: c/ligguf.c
	$(CXX) $(CFLAGS) -DDEBUG=1 -o $@ $< $(LDFLAGS)
