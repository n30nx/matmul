CC = gcc
CFLAGS = -march=native -mavx
DEBUG = -g -D DEBUG
OPTIMIZE = -O0
#LIBS = -lm
WARN = -Wall
SRC = main.c
OUT = -o main
ASAN = -fsanitize=address,undefined

BASE = $(CC) $(OPTIMIZE) $(WARN) $(CFLAGS) $(OUT) $(SRC)

build:
	$(BASE)

debug:
	$(BASE) $(DEBUG)

asan:
	$(BASE) $(DEBUG) $(ASAN)
