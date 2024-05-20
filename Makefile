CC = gcc
INTRINSICS = -mavx
CFLAGS = -march=native
DEBUG = -g -DDEBUG -v -save-temps -fno-strict-aliasing -fwrapv -fno-aggressive-loop-optimizations
OPTIMIZE = -O3 -funroll-loops -fopenmp
WARN = -Wall -Wextra -Wno-attributes
SEQ = -DSEQUENTIAL
SRC = $(wildcard src/*.c) main.c
OBJS = $(patsubst src/%.c, build/%.o, $(SRC))
INC = includes/
CMP = -DCOMPARE
ASAN = -fsanitize=address,undefined

BASE = $(CC) $(OPTIMIZE) $(WARN) $(CFLAGS) -I $(INC)

ifeq ($(shell uname -m),x86_64)
  BASE += $(INTRINSICS)
endif

EXE = main

all: build $(EXE)

build:
	@mkdir -p build/

build/%.o: src/%.c
	$(BASE) -c $< -o $@

$(EXE): $(OBJS)
	$(BASE) -o $(EXE) $(OBJS)

clean:
	rm -rf build $(EXE)

