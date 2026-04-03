CXX = g++
ARM_CXX = aarch64-linux-gnu-g++

DEBUG ?= 0
ARCH ?= $(shell uname -m)

CXXFLAGS = -Wall -Ofast -DDEBUG=1 -fopenmp
AVX_FLAGS = -march=native -mtune=native -mavx2 -mfma
ARM_FLAGS = -march=armv8.2-a+dotprod+fp16 -mtune=cortex-a76

LDFLAGS = -flto
DBGFLAGS = -O0 -g -DDEBUG=1
DBGLDFLAGS =

APP ?= ligguf_fast_qk
OBJS = ligguf_fast_qk.o lil_gguf.o lil_math.o tokenize.o common.o vdb.o
SDL_CFLAGS =
SDL_LIBS =

HOSTARCH = $(shell uname -m)

ifneq ($(ARCH),$(HOSTARCH))
	ifeq ($(ARCH),aarch64)
		CXX = $(ARM_CXX)
	endif
endif

ifndef SCALAR
	ifeq ($(ARCH),aarch64)
		CXXFLAGS += $(ARM_FLAGS)
	else
		CXXFLAGS += $(AVX_FLAGS)
	endif
endif

ifeq ($(DEBUG),1)
	CXXFLAGS += $(DBGFLAGS)
	LDFLAGS = $(DBGLDFLAGS)
endif

ifneq ($(NO_SDL),1)
	CXXFLAGS += -DUSE_SDL=1 $(shell sdl2-config --cflags 2>/dev/null)
	SDL_LIBS += $(shell sdl2-config --libs 2>/dev/null)
	OBJS += viz_sdl.o
endif

all: $(APP)
.PHONY: all

clean:
	rm -vf *.o $(APP)
.PHONY: clean

$(APP): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(SDL_LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
