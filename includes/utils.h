#include <stdint.h>
#include <stdio.h>

#include "matrix.h"

#pragma once

extern inline uint16_t round_size(uint16_t size);
extern inline void matrix_assign_random(matrix_t matrix, uint16_t row, uint16_t col);
extern inline void matrix_print(FILE *file, matrix_t restrict matrix, uint16_t row, uint16_t col);
extern inline void matrix_pad(matrix_t matrix, matrix_t restrict src, uint16_t dest_size, uint16_t src_row, uint16_t src_col);
extern inline void matrix_unpad(matrix_t matrix, matrix_t restrict src, uint16_t dest_row, uint16_t dest_col, uint16_t src_size);
extern inline bool matrix_eq(matrix_t restrict a, matrix_t restrict b, uint16_t m, uint16_t l);
