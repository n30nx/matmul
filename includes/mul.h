#include <stdint.h>

#include "./matrix.h"

#pragma once

void matrix_mult_normal(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
void matrix_multiply(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
void strassen(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
void matrix_prepare_and_mul(matrix_t c, matrix_t restrict a, matrix_t restrict b, uint16_t m, uint16_t n, uint16_t l);
