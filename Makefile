CC=gcc
CXX=g++

all: ligguf-cpp ligguf-c
.PHONY: all

debug: ligguf-debug
.PHONY: debug

clean:
	rm -f ligguf-cpp ligguf-debug ligguf-c
.PHONY: clean

CXXFLAGS=-Wall -Ofast
CFLAGS=-Wall -Wno-unused-variable -O3
LDFLAGS=-lm
DBGFLAGS=-O0 -g

ligguf-cpp: cpp/ligguf.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

ligguf-debug: cpp/ligguf_debug.cpp
	$(CXX) $(CXXFLAGS) $(DBGFLAGS) -o $@ $<

ligguf-c: c/ligguf.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
