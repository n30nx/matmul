CC = gcc
INTRINSICS = -mavx
CFLAGS = -march=native
DEBUG = -g -D DEBUG -v -save-temps -fno-strict-aliasing -fwrapv
OPTIMIZE = -O3 -funroll-loops
OPENMP = -fopenmp
DBG_86 = -fno-aggressive-loop-optimizations
WARN = -Wall -Wextra
SEQ = -D SEQUENTIAL
SRC = main.c
OUT = -o main
CMP = -D COMPARE
ASAN = -fsanitize=address,undefined

BASE = $(CC) $(OPTIMIZE) $(WARN) $(OUT) $(CFLAGS) $(SRC)

ifeq ($(shell uname -m),x86_64)
  BASE += $(INTRINSICS)
  BASE += $(OPENMP)
  BASE += $(DBG_86)
endif

build:
	$(BASE)

seq:
	$(BASE) $(SEQ)

cmp:
	$(BASE) $(SEQ) $(CMP)

debug:
	$(BASE) $(SEQ) $(CMP) $(DEBUG)

asan:
	$(BASE) $(SEQ) $(ASAN)

asandbg:
	$(BASE) $(SEQ) $(DEBUG) $(ASAN)
