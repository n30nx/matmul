CC = gcc
INTRINSICS = -march=native -mavx
DEBUG = -g -D DEBUG
OPTIMIZE = -O0
WARN = -Wall
SRC = main.c
OUT = -o main
ASAN = -fsanitize=address,undefined

BASE = $(CC) $(OPTIMIZE) $(WARN) $(OUT) $(SRC)

ifeq ($(PLATFORM),x86_64)
  BASE += $(INTRINSICS)
endif

build:
	$(BASE)

debug:
	$(BASE) $(DEBUG)

asan:
	$(BASE) $(DEBUG) $(ASAN)
