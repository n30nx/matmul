#include <stdint.h>

#include "matrix.h"
#include "ops.h"

#if defined(__x86_64__)
    #include <immintrin.h>
#endif

__attribute__((always_inline))
inline void matrix_add(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) {
#if defined(__x86_64__)
    #pragma omp parallel for
    for (uint16_t i = 0; i < size; i++) {
        for (uint16_t j = 0; j < size; j += 16) {
            if (j + 16 <= size) {  // Ensure we do not go out of bounds
                __m256i vec_a = _mm256_load_si256((__m256i*)(a + i * a_size + j));  // Load 16 elements from a
                __m256i vec_b = _mm256_load_si256((__m256i*)(b + i * b_size + j));  // Load 16 elements from b
                __m256i vec_sub = _mm256_add_epi16(vec_a, vec_b);      // Subtract the elements
                _mm256_store_si256((__m256i*)(c + i * c_size + j), vec_sub);        // Store the result in c
            } else {
                // Handle the case where remaining elements are less than 16
                for (uint16_t k = j; k < size; k++) {
                    c[i * c_size + k] = a[i * a_size + k] + b[i * b_size + k]; // Non-intrinsic subtraction for remaining elements
                }
            }
        }
    }
#else
    for (uint16_t i = 0; i < size; i++) {
        for (uint16_t j = 0; j < size; j++) {
            c[i * c_size + j] = a[i * a_size + j] + b[i * b_size + j];
        }
    }
#endif
}

__attribute__((always_inline))
inline void matrix_sub(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) {
#if defined(__x86_64__)
    #pragma omp parallel for
    for (uint16_t i = 0; i < size; i++) {
        for (uint16_t j = 0; j < size; j += 16) {
            if (j + 16 <= size) {  // Ensure we do not go out of bounds
                __m256i vec_a = _mm256_load_si256((__m256i*)(a + i * a_size + j));  // Load 16 elements from a
                __m256i vec_b = _mm256_load_si256((__m256i*)(b + i * b_size + j));  // Load 16 elements from b
                __m256i vec_sub = _mm256_sub_epi16(vec_a, vec_b);      // Subtract the elements
                _mm256_store_si256((__m256i*)(c + i * c_size + j), vec_sub);        // Store the result in c
            } else {
                // Handle the case where remaining elements are less than 16
                for (uint16_t k = j; k < size; k++) {
                    c[i * c_size + k] = a[i * a_size + k] - b[i * b_size + k]; // Non-intrinsic subtraction for remaining elements
                }
            }
        }
    }
#else
    for (uint16_t i = 0; i < size; i++) {
        for (uint16_t j = 0; j < size; j++) {
            c[i * c_size + j] = a[i * a_size + j] - b[i * b_size + j];
        }
    }
#endif
}
