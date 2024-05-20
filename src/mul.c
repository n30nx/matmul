#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#if defined(__x86_64__)
    #include <immintrin.h>
#endif

#include "matrix.h"
#include "mul.h"
#include "utils.h"
#include "mem.h"
#include "ops.h"

__attribute__((unused))
void matrix_mult_normal(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) {
    uint16_t i, j, k, i0, j0, k0;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            c[i * c_size + j] = 0;
        }
    }

    // Cache-efficient matrix algorithm
    for (i0 = 0; i0 < size; i0 += BLOCK_SIZE) {
        for (j0 = 0; j0 < size; j0 += BLOCK_SIZE) {
            for (k0 = 0; k0 < size; k0 += BLOCK_SIZE) {
                for (i = i0; i < i0 + BLOCK_SIZE && i < size; i++) {
                    for (j = j0; j < j0 + BLOCK_SIZE && j < size; j++) {
                        for (k = k0; k < k0 + BLOCK_SIZE && k < size; k++) {
                            c[i * c_size + j] += a[i * a_size + k] * b[k * b_size + j];
                        }
                    }
                }
            }
        }
    }
}

// FrodoKEM Matrix Multiplication
// REFS: https://eprint.iacr.org/2021/711.pdf
//       https://github.com/microsoft/PQCrypto-LWEKE/blob/a2f9dec8917ccc3464b3378d46b140fa7353320d/FrodoKEM/src/frodo_macrify.c#L252
void matrix_multiply(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) {
#if defined(__x86_64__)
    __m256i b_vec, acc_vec;
    #pragma omp parallel for private(b_vec, acc_vec)
    for (uint16_t j = 0; j < size; j++) {
        for (uint16_t q = 0; q < size; q += 16) {
            acc_vec = _mm256_setzero_si256();  // Initialize acc_vec to zero

            for (uint16_t p = 0; p < size; p += 8) {  // Assuming size is a multiple of 8
                __m256i sp_vec[8];
                for (uint16_t k = 0; k < 8; k++) {
                    sp_vec[k] = _mm256_set1_epi16(a[j * a_size + p + k]);  // Broadcast elements from 'a'
                }

                for (uint16_t k = 0; k < 8; k++) {
                    b_vec = _mm256_load_si256((__m256i*)(b + (p + k) * b_size + q));  // Load 16 elements from 'b'
                    acc_vec = _mm256_add_epi16(acc_vec, _mm256_mullo_epi16(sp_vec[k], b_vec));  // Multiply and accumulate
                }
            }

            if (q + 16 <= size) {
                _mm256_store_si256((__m256i*)(c + j * c_size + q), acc_vec);  // Store the result back to 'c'
            } else {
                // Handle remaining columns that do not fill a full SIMD register width
                uint16_t temp[16];
                _mm256_storeu_si256((__m256i*)temp, acc_vec);
                for (uint16_t r = 0; r < size % 16; r++) {
                    c[j * c_size + q + r] = temp[r];
                }
            }
        }
    }
#else
    matrix_mult_normal(c, a, b, size, c_size, a_size, b_size);
#endif
}

// Strassen Algorithm
// REF: https://en.wikipedia.org/wiki/Strassen_algorithm
void strassen(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) { 
    // I know this looks shady but it's the best one for cache-misses, doesn't matter small or big numbers, 512 or 1024 does the trick.
    if (size <= LEAFSIZE) {
        matrix_multiply(c, a, b, size, c_size, a_size, b_size);
    } else {
        uint16_t new_size = size / 2;

        // Bootstrap submatrices
        matrix_t a11 = a;
        matrix_t a12 = a + new_size;
        matrix_t a21 = a + new_size * a_size;
        matrix_t a22 = a + new_size * a_size + new_size;
        
        matrix_t b11 = b;
        matrix_t b12 = b + new_size;
        matrix_t b21 = b + new_size * b_size;
        matrix_t b22 = b + new_size * b_size + new_size;

        matrix_t c11 = c;
        matrix_t c12 = c + new_size;
        matrix_t c21 = c + new_size * c_size;
        matrix_t c22 = c + new_size * c_size + new_size;

        matrix_t p1 = matrix_new(new_size, new_size);
        matrix_t p2 = matrix_new(new_size, new_size);
        matrix_t p3 = matrix_new(new_size, new_size);
        matrix_t p4 = matrix_new(new_size, new_size);
        matrix_t p5 = matrix_new(new_size, new_size);
        matrix_t p6 = matrix_new(new_size, new_size);
        matrix_t p7 = matrix_new(new_size, new_size);

        matrix_t tmp_a = matrix_new(new_size, new_size);
        matrix_t tmp_b = matrix_new(new_size, new_size);

        // p1 = a11 * (b12 - b22)
        matrix_sub(tmp_b, b12, b22, new_size, new_size, b_size, b_size);
        strassen(p1, a11, tmp_b, new_size, new_size, a_size, new_size);

        // p2 = (a11 + a12) * b22
        matrix_add(tmp_a, a11, a12, new_size, new_size, a_size, a_size);
        strassen(p2, tmp_a, b22, new_size, new_size, new_size, b_size);
       
        // p3 = (a21 + a22) * b11 
        matrix_add(tmp_a, a21, a22, new_size, new_size, a_size, a_size);
        strassen(p3, tmp_a, b11, new_size, new_size, new_size, b_size);

        // p4 = a22 * (b21 - b11)
        matrix_sub(tmp_b, b21, b11, new_size, new_size, b_size, b_size);
        strassen(p4, a22, tmp_b, new_size, new_size, a_size, new_size);

        // p5 = (a11 + a22) * (b11 + b22)
        matrix_add(tmp_a, a11, a22, new_size, new_size, a_size, a_size);
        matrix_add(tmp_b, b11, b22, new_size, new_size, b_size, b_size);
        strassen(p5, tmp_a, tmp_b, new_size, new_size, new_size, new_size);

        // p6 = (a12 - a22) * (b21 + b22)    
        matrix_sub(tmp_a, a12, a22, new_size, new_size, a_size, a_size);
        matrix_add(tmp_b, b21, b22, new_size, new_size, b_size, b_size);
        strassen(p6, tmp_a, tmp_b, new_size, new_size, new_size, new_size);

        // p7 = (a11 - a21) * (b11 + b12)
        matrix_sub(tmp_a, a11, a21, new_size, new_size, a_size, a_size);
        matrix_add(tmp_b, b11, b12, new_size, new_size, b_size, b_size);
        strassen(p7, tmp_a, tmp_b, new_size, new_size, new_size, new_size);

        // c11 = p5 + p4 - p2 + p6
        matrix_add(c11, p5, p4, new_size, c_size, new_size, new_size);
        matrix_sub(c11, c11, p2, new_size, c_size, c_size, new_size);
        matrix_add(c11, c11, p6, new_size, c_size, c_size, new_size);

        // c12 = p1 + p2
        matrix_add(c12, p1, p2, new_size, c_size, new_size, new_size);

        // c21 = p3 + p4
        matrix_add(c21, p3, p4, new_size, c_size, new_size, new_size);

        // c22 = p5 + p1 - p3 - p7
        matrix_add(c22, p5, p1, new_size, c_size, new_size, new_size);
        matrix_sub(c22, c22, p3, new_size, c_size, c_size, new_size);
        matrix_sub(c22, c22, p7, new_size, c_size, c_size, new_size);
        
        matrix_free(p1);
        matrix_free(p2);
        matrix_free(p3);
        matrix_free(p4);
        matrix_free(p5);
        matrix_free(p6);
        matrix_free(p7);

        matrix_free(tmp_a);
        matrix_free(tmp_b);
    }
}

void matrix_prepare_and_mul(matrix_t c, matrix_t restrict a, matrix_t restrict b, uint16_t m, uint16_t n, uint16_t l) {
    // Get the padding size
    uint16_t new_size = round_size(max(max(m, n), l));
    new_size = new_size < LEAFSIZE ? LEAFSIZE : new_size;
    DBG("new_size: %hu", new_size);

    bool pad_a = new_size != m || new_size != n;
    matrix_t padded_a;
    if (!pad_a) {
        padded_a = a;
    } else {
        padded_a = matrix_new(new_size, new_size);
        matrix_pad(padded_a, a, new_size, m, n);
    }

    bool pad_b = new_size != n || new_size != l;
    matrix_t padded_b;
    if (!pad_b) {
        padded_b = b;
    } else {
        padded_b = matrix_new(new_size, new_size);
        matrix_pad(padded_b, b, new_size, n, l);
    }

    bool pad_result = new_size != m || new_size != l;
    matrix_t padded_result;
    if (pad_result) {
        padded_result = matrix_new(new_size, new_size);
    } else {
        padded_result = c;
    }

    // Pad the matrix for the strassen algorithm to be able to divide them into equal 2^n lengthed parts
    strassen(padded_result, padded_a, padded_b, new_size, new_size, new_size, new_size);

    // Free the padded variables as they won't be necessary anymore
    if (pad_a) matrix_free(padded_a);
    if (pad_b) matrix_free(padded_b);

    // Remove the padding to print properly
    if (pad_result) {
        matrix_unpad(c, padded_result, m, l, new_size);
        matrix_free(padded_result);
    }
}
