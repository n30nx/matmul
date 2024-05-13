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

debug:
	$(BASE) $(DEBUG)

seq:
	$(BASE) $(SEQ)

asan:
	$(BASE) $(DEBUG) $(ASAN)
