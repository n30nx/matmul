#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

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

/*
 * I've used uint16_t because:
 * 1- If I need to do 1000x1000 * 1000x1000 matrix multiplication I'm
 * gonna need way more bits and it's not feasible
 * 2- I'd like to think the elements of the matrix \every M_{i} \in Z/2^16Z
 */
typedef uint16_t *matrix_t;

#pragma region mem

__attribute__((always_inline))
static inline matrix_t matrix_new(uint16_t row, uint16_t col) {
#if defined(__x86_64__)
    matrix_t new = (matrix_t)_mm_malloc(sizeof(uint16_t) * row * col, 64);
#else
    matrix_t new = (matrix_t)malloc(sizeof(uint16_t) * row * col);
#endif
    assert(new);
    DBG("allocated %hux%hu matrix with the size %lu", row, col, malloc_usable_size(new));
    return new;
}

__attribute__((always_inline))
static inline void matrix_free(matrix_t matrix) {
#if defined(__x86_64__)
    _mm_free(matrix);
#else
    free(matrix);
#endif
}

#pragma endregion // mem

#pragma region utils

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
static inline uint16_t round_size(uint16_t size) {
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
static inline void matrix_assign_random(matrix_t matrix, uint16_t row, uint16_t col) {
    // Pretty straight-forward, assigns random (mod 2^16) to all indexes
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            matrix[i * col + j] = rand() % UINT16_MAX;
        }
    }
}

__attribute__((always_inline))
static inline void matrix_print(FILE *stream, matrix_t __restrict matrix, uint16_t row, uint16_t col) {
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
static inline void matrix_add(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) {
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
static inline void matrix_sub(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) {
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

__attribute__((always_inline))
static inline void matrix_pad(matrix_t dest, matrix_t __restrict src, uint16_t dest_row, uint16_t dest_col, uint16_t src_row, uint16_t src_col) {
#if defined(__x86_64__)
    __m256i zero = _mm256_setzero_si256();
    for (uint16_t i = 0; i < dest_row; i++) {
        for (uint16_t j = 0; j < dest_col; j += 16) {  // Using uint16_t, 16 values per 256-bit register
            _mm256_store_si256((__m256i*)(dest + i * dest_col + j), zero);
        }
    }

    // Now copy src to dest, taking care of possible misalignment
    for (uint16_t i = 0; i < src_row; i++) {
        uint16_t j = 0;
        for (; j < src_col - 15; j += 16) {  // Copy in blocks of 16, assuming src_col is large enough
            __m256i data = _mm256_loadu_si256((__m256i*)(src + i * src_col + j));
            _mm256_storeu_si256((__m256i*)(dest + i * dest_col + j), data);
        }
        for (; j < src_col; j++) {  // Handle remaining elements if any
            dest[i * dest_col + j] = src[i * src_col + j];
        }
    }
#else
    memset(dest, 0, dest_row * dest_col * sizeof(uint16_t)); // Write zeroes
    for (uint16_t i = 0; i < src_row; i++) {
        memcpy(dest + dest_col * i, src + src_col * i, src_col * sizeof(uint16_t));  // Copy original data
    }
#endif
}

__attribute__((always_inline))
static inline void matrix_unpad(matrix_t dest, matrix_t __restrict src, uint16_t dest_row, uint16_t dest_col, uint16_t src_size) {
    DBG("dest_start = %p, dest_end = %p, length = %hu", dest, dest + dest_row * dest_col, dest_row * dest_col);
    DBG("src_start = %p, src_end = %p, length = %hu", src, src + src_size * src_size, src_size * src_size);
    DBG("dest_row = %hu", dest_row);
    for (uint16_t i = 0; i < dest_row; i++) {
        DBG("copying to dest + %hu (%hu * %hu) from src + %hu (%hu * %hu), with the size %hu", dest_col * i, dest_col, i, src_size * i, src_size, i, dest_col);
        memcpy(dest + dest_col * i, src + src_size * i, sizeof(uint16_t) * dest_col); // Copy data from the padded matrix
    }
}

__attribute__((always_inline, unused))
static inline bool matrix_eq(matrix_t __restrict a, matrix_t __restrict b, uint16_t m, uint16_t l) {
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

#pragma endregion // utils

static void matrix_mult_normal(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) {
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
static void matrix_multiply(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) {
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
static void strassen(matrix_t c, matrix_t a, matrix_t b, uint16_t size, uint16_t c_size, uint16_t a_size, uint16_t b_size) { 
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

void matrix_prepare_and_mul(matrix_t c, matrix_t __restrict a, matrix_t __restrict b, uint16_t m, uint16_t n, uint16_t l) {
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
        matrix_pad(padded_a, a, new_size, new_size, m, n);
    }

    bool pad_b = new_size != n || new_size != l;
    matrix_t padded_b;
    if (!pad_b) {
        padded_b = b;
    } else {
        padded_b = matrix_new(new_size, new_size);
        matrix_pad(padded_b, b, new_size, new_size, n, l);
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

int main(int argc, char **argv) {
    int ret_val = 0;
    uint16_t m, n, l;

    // Read the params from stdin if supplied
    if (argc == 4) {
        sscanf(*(argv + 1), "%hu", &m);
        sscanf(*(argv + 2), "%hu", &n);
        sscanf(*(argv + 3), "%hu", &l);
    } else {
        // Read the params
        scanf("%hu %hu %hu", &m, &n, &l);
    }

    // Create matrixes
    matrix_t a_matrix = matrix_new(m, n);
    matrix_t b_matrix = matrix_new(n, l);
    matrix_t c_matrix = matrix_new(m, l);

    srand(time(NULL));

#if defined(SEQUENTIAL)
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < n; j++) {
            a_matrix[i * n + j] = i * n + j + 1;
        }
    }

    for (uint16_t i = 0; i < n; i++) {
        for (uint16_t j = 0; j < l; j++) {
            b_matrix[i * l + j] = i * l + j + 2;
        }
    }
#else
    matrix_assign_random(a_matrix, m, n);
    matrix_assign_random(b_matrix, n, l);
#endif

#if defined(COMPARE)
    matrix_t d_matrix = matrix_new(m, l);

    uint16_t new_size = round_size(max(max(m, n), l));
    new_size = new_size < LEAFSIZE ? LEAFSIZE : new_size;

    bool pad_a = new_size != m || new_size != n;
    matrix_t padded_a;
    if (!pad_a) {
        padded_a = a_matrix;
    } else {
        padded_a = matrix_new(new_size, new_size);
        matrix_pad(padded_a, a_matrix, new_size, new_size, m, n);
    }

    bool pad_b = new_size != n || new_size != l;
    matrix_t padded_b;
    if (!pad_b) {
        padded_b = b_matrix;
    } else {
        padded_b = matrix_new(new_size, new_size);
        matrix_pad(padded_b, b_matrix, new_size, new_size, n, l);
    }

    bool pad_result = new_size != m || new_size != l;
    matrix_t padded_d;
    if (!pad_result) {
        padded_d = d_matrix;
    } else {
        padded_d = matrix_new(new_size, new_size);
    }

    matrix_mult_normal(padded_d, padded_a, padded_b, new_size, new_size, new_size, new_size);
    // Free the padded variables as they won't be necessary anymore
    if (pad_a) matrix_free(padded_a);
    if (pad_b) matrix_free(padded_b);

    // Remove the padding to print properly
    if (pad_result) {
        matrix_unpad(d_matrix, padded_d, m, l, new_size);
        matrix_free(padded_d);
    }
#endif
    matrix_prepare_and_mul(c_matrix, a_matrix, b_matrix, m, n, l);

    // Write the matrix to a file in order to keep the stdout clean.
    FILE *matrix_out = fopen("matrix_output_main.txt", "w");
    matrix_print(matrix_out, c_matrix, m, l);
    fclose(matrix_out);

#if defined(COMPARE)
    //matrix_print(stdout, c_matrix, m, l);
    matrix_print(stdout, d_matrix, m, l);

    ret_val = !(matrix_eq(c_matrix, d_matrix, m, l));
    if (ret_val == 0) printf("Function is correct!\n");
    else printf("Function is not correct!\n");
    matrix_free(d_matrix);
#endif

    // Free used memory
    matrix_free(a_matrix);
    matrix_free(b_matrix);
    matrix_free(c_matrix);

    return ret_val;
}
