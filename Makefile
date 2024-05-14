CC = gcc
INTRINSICS = -mavx
CFLAGS = -march=native
DEBUG = -g -D DEBUG
OPTIMIZE = -O3 -funroll-loops
WARN = -Wall -Wextra
SEQ = -D SEQUENTIAL
SRC = main.c
OUT = -o main
ASAN = -fsanitize=address,undefined

BASE = $(CC) $(OPTIMIZE) $(WARN) $(OUT) $(CFLAGS) $(SRC)

ifeq ($(shell uname -m),x86_64)
  BASE += $(INTRINSICS)
endif

build:
	$(BASE)

seq:
	$(BASE) $(SEQ)

debug:
	$(BASE) $(SEQ) $(DEBUG)

asan:
	$(BASE) $(SEQ) $(DEBUG) $(ASAN)
