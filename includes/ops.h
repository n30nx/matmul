#include "./matrix.h"

#pragma once

extern inline void matrix_add(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
extern inline void matrix_sub(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
