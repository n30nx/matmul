#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "matrix.h"
#include "utils.h"

#if defined(__x86_64__)
    #include <immintrin.h>
#endif

__attribute__((always_inline))
static inline bool is_power_of_two(uint16_t n)
{
    uint16_t i = 0;
    while (n > 0) {
        if ((n & 1) == 1) {
            i++;
        }
        n = n >> 1;
    }
 
    return i == 1;
}

static const int log_table[64] = {
    63,  0, 58,  1, 59, 47, 53,  2,
    60, 39, 48, 27, 54, 33, 42,  3,
    61, 51, 37, 40, 49, 18, 28, 20,
    55, 30, 34, 11, 43, 14, 22,  4,
    62, 57, 46, 52, 38, 26, 32, 41,
    50, 36, 17, 19, 29, 10, 13, 21,
    56, 45, 25, 31, 35, 16,  9, 12,
    44, 24, 15,  8, 23,  7,  6,  5
};

/*
 * de Brujin sequence
 * http://supertech.csail.mit.edu/papers/debruijn.pdf
 */
// Fast log2 algorithm
__attribute__((always_inline))
static inline uint16_t u16_log2(size_t size) {
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    return log_table[((size - (size >> 1)) * 0x07EDD5E59A4E28C2) >> 58];
}

__attribute__((always_inline))
inline uint16_t round_size(uint16_t size) {
    if (is_power_of_two(size)) {
        return size;
    }
    uint16_t cur = size;
    uint16_t log_cap = u16_log2(size);
    size = 1 << ((uint16_t)log_cap + 1); 
    DBG("log_cap, size = %hu, %hu", log_cap, size);

    if (cur >= size) {
        return 0;
    }

    return size;
}

__attribute__((always_inline))
inline void matrix_assign_random(matrix_t matrix, uint16_t row, uint16_t col) {
    // Pretty straight-forward, assigns random (mod 2^16) to all indexes
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            matrix[i * col + j] = rand() % UINT16_MAX;
        }
    }
}

__attribute__((always_inline))
inline void matrix_print(FILE *stream, matrix_t restrict matrix, uint16_t row, uint16_t col) {
    // Writes values to the given stream
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            fprintf(stream, "%hu ", matrix[i * col + j]);
        }
        fprintf(stream, "\n");
    }
    fprintf(stream, "\n\n\n");
}


__attribute__((always_inline))
inline void matrix_pad(matrix_t dest, matrix_t restrict src, uint16_t dest_size, uint16_t src_row, uint16_t src_col) {
#if defined(__x86_64__)
    __m256i zero = _mm256_setzero_si256();
    for (uint16_t i = 0; i < dest_size; i++) {
        for (uint16_t j = 0; j < dest_size; j += 16) {  // Using uint16_t, 16 values per 256-bit register
            _mm256_store_si256((__m256i*)(dest + i * dest_size + j), zero);
        }
    }

    // Now copy src to dest, taking care of possible misalignment
    for (uint16_t i = 0; i < src_row; i++) {
        uint16_t j = 0;
        for (; j < src_col - 15; j += 16) {  // Copy in blocks of 16, assuming src_col is large enough
            __m256i data = _mm256_loadu_si256((__m256i*)(src + i * src_col + j));
            _mm256_storeu_si256((__m256i*)(dest + i * dest_size + j), data);
        }
        for (; j < src_col; j++) {  // Handle remaining elements if any
            dest[i * dest_size + j] = src[i * src_col + j];
        }
    }
#else
    memset(dest, 0, dest_size * dest_size * sizeof(uint16_t)); // Write zeroes
    for (uint16_t i = 0; i < src_row; i++) {
        memcpy(dest + dest_size * i, src + src_col * i, src_col * sizeof(uint16_t));  // Copy original data
    }
#endif
}

__attribute__((always_inline))
inline void matrix_unpad(matrix_t dest, matrix_t restrict src, uint16_t dest_row, uint16_t dest_col, uint16_t src_size) {
    DBG("dest_start = %p, dest_end = %p, length = %hu", dest, dest + dest_row * dest_col, dest_row * dest_col);
    DBG("src_start = %p, src_end = %p, length = %hu", src, src + src_size * src_size, src_size * src_size);
    DBG("dest_row = %hu", dest_row);
    for (uint16_t i = 0; i < dest_row; i++) {
        DBG("copying to dest + %hu (%hu * %hu) from src + %hu (%hu * %hu), with the size %hu", dest_col * i, dest_col, i, src_size * i, src_size, i, dest_col);
        memcpy(dest + dest_col * i, src + src_size * i, sizeof(uint16_t) * dest_col); // Copy data from the padded matrix
    }
}

__attribute__((always_inline, unused))
inline bool matrix_eq(matrix_t restrict a, matrix_t restrict b, uint16_t m, uint16_t l) {
    // Check if matrixes are equal (used for validating multiplication)
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < l; j++) {
            if (a[i * l + j] != b[i * l + j]) {
                return false;
            }
        }
    }
    return true;
}
