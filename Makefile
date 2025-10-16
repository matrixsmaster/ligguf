CXX=g++

all: ligguf
.PHONY: all

debug: ligguf_debug
.PHONY: debug

clean:
	rm -f ligguf ligguf_debug
.PHONY: clean

CXXFLAGS=-Wall -Ofast
DBGFLAGS=-O0 -g

ligguf: ligguf.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

ligguf_debug: ligguf_debug.cpp
	$(CXX) $(CXXFLAGS) $(DBGFLAGS) -o $@ $<
