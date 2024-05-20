#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

// This part is for allowing me to work from my MacBook
#if defined(__APPLE__)
    #include <malloc/malloc.h>
    #define malloc_usable_size(x) malloc_size(x)
#else
    #include <malloc.h>
    #if defined(_WIN32)
        #define malloc_usable_size(x) _msize(x)
    #endif
#endif

#if defined(__x86_64__)
    #include <immintrin.h>
#endif
//

// Verbosed debug macro
#if defined(DEBUG)
    #define DBG(fmt, ...) do {                                  \
        fprintf(stdout, "%s:%d ", __FUNCTION__, __LINE__);      \
        fprintf(stdout, fmt, __VA_ARGS__);                      \
        fprintf(stdout, "\n");                                  \
    } while (0)
#else
    #define DBG(...)
#endif

// Didn't want to link maths library for just max and min functions
#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

#define BLOCK_SIZE 1024
#define LEAFSIZE 512

#pragma once

/*
 * I've used uint16_t because:
 * 1- If I need to do 1000x1000 * 1000x1000 matrix multiplication I'm
 * gonna need way more bits and it's not feasible
 * 2- I'd like to think the elements of the matrix \every M_{i} \in Z/2^16Z
 */
typedef uint16_t *matrix_t;

//static inline matrix_t matrix_new(uint16_t row, uint16_t col);
//static inline void matrix_free(matrix_t matrix);

//static inline uint16_t round_size(uint16_t size);
//static inline void matrix_assign_random(matrix_t matrix, uint16_t row, uint16_t col);
//static inline void matrix_print(FILE *file, matrix_t restrict matrix, uint16_t row, uint16_t col);
//static inline void matrix_pad(matrix_t matrix, matrix_t restrict src, uint16_t dest_size, uint16_t src_row, uint16_t src_col);
//static inline void matrix_unpad(matrix_t matrix, matrix_t restrict src, uint16_t dest_row, uint16_t dest_col, uint16_t src_size);
//static inline bool matrix_eq(matrix_t restrict a, matrix_t restrict b, uint16_t m, uint16_t l);

//static inline void matrix_add(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
//static inline void matrix_sub(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);

//static void matrix_mult_normal(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
//static void matrix_multiply(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
//static void strassen(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size);
//void matrix_prepare_and_mul(matrix_t c, matrix_t restrict a, matrix_t restrict b, uint16_t m, uint16_t n, uint16_t l);
