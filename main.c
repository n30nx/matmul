#include <immintrin.h> // AVX2 intrinsics
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#define PARAMS_N 1344
#define PARAMS_NBAR 8

#define DBG(fmt, ...) do {                                  \
    fprintf(stdout, "%s:%d ", __FUNCTION__, __LINE__);      \
    fprintf(stdout, fmt, __VA_ARGS__);                      \
    fprintf(stdout, "\n");                                  \
} while (0)

typedef int16_t **matrix_t;

#pragma region mem
matrix_t matrix_new(uint16_t row, uint16_t col) {
    matrix_t new = (matrix_t)_mm_malloc(row * sizeof(int16_t*), 32);
    if (new == NULL) {
        printf("new = NULL!\n");
        exit(1);
    }
    for (uint16_t i = 0; i < row; i++) {
        new[i] = (int16_t*)_mm_malloc(col * sizeof(int16_t), 32);
        if (new[i] == NULL) {
            printf("new[%hu] = NULL!\n", i);
            exit(1);
        }
        memset(new[i], 0, col * sizeof(int16_t));
    }
    return new;
}

void matrix_free(matrix_t matrix, uint16_t row) {
    for (uint16_t i = 0; i < row; i++) {
        _mm_free(matrix[i]);
    }
    _mm_free(matrix);
}
#pragma endregion // mem

#pragma region utils
void matrix_assign_random(matrix_t matrix, uint16_t row, uint16_t col) {
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            matrix[i][j] = rand() % INT16_MAX;
        }
    }
}

void matrix_print(matrix_t matrix, uint16_t row, uint16_t col) {
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            printf("%hd ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n\n\n");
}
#pragma endregion // utils

// FrodoKEM Matrix Multiplication
// TODO: https://eprint.iacr.org/2021/711.pdf
//       https://github.com/microsoft/PQCrypto-LWEKE/blob/a2f9dec8917ccc3464b3378d46b140fa7353320d/FrodoKEM/src/frodo_macrify.c#L252
//
// Schönhage-Strassen Algorithm
// TODO: https://en.wikipedia.org/wiki/Sch%C3%B6nhage%E2%80%93Strassen_algorithm
//       https://www.sanfoundry.com/c-program-implement-schonhage-strassen-algorithm-multiplication-two-numbers/
void matrix_multiply(matrix_t a, matrix_t b, matrix_t c, uint16_t m, uint16_t n, uint16_t l) {
    // Initialize matrix C to zero
    for (uint16_t i = 0; i < m; i++) {
        memset(c[i], 0, l * sizeof(int16_t));
    }

    // Perform matrix multiplication using AVX2
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < l; j++) {
            __m256i sum_vec = _mm256_setzero_si256(); // Initialize the sum vector to zero

            for (uint16_t k = 0; k < n; k += 16) { // Process 16 elements at a time
                if (k + 16 <= n) {  // Ensure we do not go out of bounds
                    __m256i vec_a = _mm256_loadu_si256((__m256i*)&a[i][k]); // Load 16 elements from row i of A
                    __m256i vec_b = _mm256_loadu_si256((__m256i*)&b[k][j]); // Load 16 elements from column j of B, transposed for continuous access
                    __m256i vec_mul = _mm256_mullo_epi16(vec_a, vec_b);  // Multiply the elements
                    sum_vec = _mm256_add_epi16(sum_vec, vec_mul); // Add the results to the sum vector
                } else {
                    // Handle the case where n is not a multiple of 16
                    for (uint16_t x = k; x < n; x++) {
                        int16_t product = a[i][x] * b[x][j];
                        c[i][j] += product; // Scalar addition for remaining elements
                    }
                }
            }

            // Horizontal sum of sum_vec and store result in matrix C
            int16_t *elements = (int16_t*)&sum_vec;
            for (int idx = 0; idx < 16; idx++) {
                c[i][j] += elements[idx];
            }
        }
    }
}


/*void matrix_multiply(matrix_t a, matrix_t b, matrix_t c, uint16_t m, uint16_t n, uint16_t l) {
    // Perform matrix multiplication using AVX2
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < l; j++) {
            for (uint16_t j_bar = 0; j_bar < PARAMS_NBAR; j_bar++) {
                __m256i sp[8], acc;
                for (uint16_t p = 0; p < 8; p++) {
                    sp[p] = _mm256_set1_epi16(b[j_bar * PARAMS_N + i + p]);
                }
                for (uint16_t q = 0; q < PARAMS_N; q += 16) {
                    acc = _mm256_load_si256((__m256i*)&c[j_bar * PARAMS_N + q]);
                    for (uint16_t p = 0; p < 8; p++) {
                        __m256i b_vec = _mm256_load_si256((__m256i*)&a[p * PARAMS_N + q]);
                        b_vec = _mm256_mullo_epi16(b_vec, sp[p]);
                        acc = _mm256_add_epi16(acc, b_vec);
                    }
                    _mm256_store_si256((__m256i*)&c[j_bar * PARAMS_N + q], acc);
                }
            }
        }
    }
}*/

void matrix_multiply_normal(matrix_t a, matrix_t b, matrix_t c, uint16_t m, uint16_t n, uint16_t l) {
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < l; j++) {
            for (uint16_t k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main(int argc, char **argv) {
    uint16_t m, n, l;

    if (argc == 4) {
        sscanf(*(argv + 1), "%hu", &m);
        sscanf(*(argv + 2), "%hu", &n);
        sscanf(*(argv + 3), "%hu", &l);
    } else {
        scanf("%hu %hu %hu", &m, &n, &l);
    }

    matrix_t a_matrix = matrix_new(m, n);
    matrix_t b_matrix = matrix_new(n, l);
    matrix_t c_matrix = matrix_new(m, l);
    matrix_t d_matrix = matrix_new(m, l);

    srand(time(NULL));

    matrix_assign_random(a_matrix, m, n);
    printf("A matrix:\n");
    matrix_print(a_matrix, m, n);
    
    matrix_assign_random(b_matrix, n, l);
    printf("B matrix:\n");
    matrix_print(b_matrix, n, l);

    matrix_multiply(a_matrix, b_matrix, c_matrix, m, n, l);
    printf("C matrix:\n");
    matrix_print(c_matrix, m, l);

    matrix_multiply_normal(a_matrix, b_matrix, d_matrix, m, n, l);
    printf("D matrix:\n");
    matrix_print(d_matrix, m, l);

    matrix_free(a_matrix, m);
    matrix_free(b_matrix, n);
    matrix_free(c_matrix, m);
    matrix_free(d_matrix, m);

    return 0;   
}
