CC = gcc
INTRINSICS = -mavx
CFLAGS = -march=native
DEBUG = -g -D DEBUG
OPTIMIZE = -O3
WARN = -Wall -Wextra -Werror
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

asan:
	$(BASE) $(DEBUG) $(ASAN)
