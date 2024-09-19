#include <stdlib.h>
#include <stdint.h>

#include "./matrix.h"

#if defined(__x86_64__)
    #include <immintrin.h>
#endif

#pragma once

extern inline matrix_t matrix_new(uint16_t row, uint16_t col);
extern inline void matrix_free(matrix_t matrix);
