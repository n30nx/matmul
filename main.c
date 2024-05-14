#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <limits.h>
#include <time.h>

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

/*
 * I've used uint16_t because:
 * 1- If I need to do 1000x1000 * 1000x1000 matrix multiplication I'm
 * gonna need way more bits and it's not feasible
 * 2- I'd like to think the elements of the matrix \every M_{i} \in Z/2^16Z
 */
typedef uint16_t *matrix_t;

#pragma region mem




/*
 * Matrix New
 *
 * This function allocates memory for a new matrix with the specified number of rows and columns.
 *
 * @param row: Number of rows in the new matrix
 * @param col: Number of columns in the new matrix
 *
 * @return: Pointer to the newly allocated matrix
 *
 * @pre row and col should be greater than 0.
 *
 * @post Memory is allocated for a new matrix with the specified dimensions.
 * @post The returned pointer points to the allocated memory block.
 * @post If preconditions are not met or memory allocation fails, the behavior is undefined.
 */
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



/*
 * Matrix Free
 *
 * This function frees the memory allocated for a matrix.
 *
 * @param matrix: Pointer to the matrix to be freed
 *
 * @pre matrix must point to a valid memory block allocated for a matrix.
 *
 * @post The memory allocated for the matrix pointed to by matrix is freed.
 * @post If preconditions are not met, the behavior is undefined.
 */
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
 * Fast log2 algorithm
*/

/*
 * Unsigned 16-bit Logarithm Base 2
 *
 * This function calculates the logarithm base 2 of a 16-bit unsigned integer.
 *
 * @param size: Size for which the logarithm base 2 will be calculated
 *
 * @return: Logarithm base 2 of the input size
 *
 * @pre None
 *
 * @post The returned value is the logarithm base 2 of the input size.
 */
__attribute__((always_inline))
static inline int u16_log2(size_t size) {
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    return log_table[((size - (size >> 1)) * 0x07EDD5E59A4E28C2) >> 58];
}


/*
 * Round Size
 *
 * This function rounds up a given size to the nearest power of 2.
 *
 * @param size: Size to be rounded up
 *
 * @return: Rounded up size to the nearest power of 2
 *
 * @pre None
 *
 * @post The returned value is the input size rounded up to the nearest power of 2.
 */
__attribute__((always_inline))
static inline uint16_t round_size(uint16_t size) {
    uint16_t cur = size;
    uint16_t log_cap = u16_log2(size);
    size = 1 << ((uint16_t)log_cap + 1); 
    DBG("log_cap, size = %hu, %hu", log_cap, size);

    if (cur >= size) {
        return 0;
    }

    return size;
}



/*
 * Matrix Random Assignment
 *
 * This function assigns random values (modulo 2^16) to all elements of a matrix.
 *
 * @param matrix: Matrix to which random values will be assigned (2D array of integers, stored in row-major order)
 * @param row: Number of rows in the matrix
 * @param col: Number of columns in the matrix
 *
 * @pre matrix must not be NULL.
 * @pre row and col should be greater than 0.
 *
 * @post All elements of the matrix are assigned random values modulo 2^16.
 * @post If preconditions are not met, the behavior is undefined.
 */
__attribute__((always_inline))
static inline void matrix_assign_random(matrix_t matrix, uint16_t row, uint16_t col) {
    // Pretty straight-forward, assigns random (mod 2^16) to all indexes
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            matrix[i * col + j] = rand() % UINT16_MAX;
        }
    }
}


/*
 * Matrix Print
 *
 * This function prints the elements of a matrix to the specified stream.
 *
 * @param stream: Pointer to the file stream where the matrix will be printed
 * @param matrix: Matrix to be printed (2D array of integers, stored in row-major order)
 * @param row: Number of rows in the matrix
 * @param col: Number of columns in the matrix
 *
 * @pre stream must not be NULL.
 * @pre matrix must not be NULL.
 * @pre row and col should be greater than 0.
 *
 * @post The matrix elements are printed to the specified stream.
 * @post If preconditions are not met, the behavior is undefined.
 */
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


/*
 * Matrix Addition
 *
 * This function adds matrix b to matrix a and stores the result in matrix c.
 *
 * @param c: Resultant matrix (2D array of integers, stored in row-major order)
 * @param a: Matrix operand A (2D array of integers, stored in row-major order)
 * @param b: Matrix operand B (2D array of integers, stored in row-major order)
 * @param size: Size of the matrices (number of rows or columns)
 *
 * @pre c, a, and b must not be NULL.
 * @pre size should be greater than 0.
 * @pre Matrices a, b, and c must be properly allocated to hold size*size elements.
 *
 * @post Matrix c contains the result of the addition of matrix b to matrix a.
 * @post If preconditions are not met, the behavior is undefined.
 */
 __attribute__((always_inline))
static inline void matrix_add(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size) {
    uint32_t num_elements = size * size;
#if defined(__x86_64__)
    for (uint32_t i = 0; i < num_elements; i += 16) {
        if (i + 16 <= num_elements) {  // So we do not go out of bounds
            __m256i vec_a = _mm256_load_si256((__m256i*)(a + i));  // Load 16 elements from a
            __m256i vec_b = _mm256_load_si256((__m256i*)(b + i));  // Load 16 elements from b
            __m256i vec_sum = _mm256_add_epi16(vec_a, vec_b);      // Add the elements
            _mm256_store_si256((__m256i*)(c + i), vec_sum);        // Store the result in c
        } else {
            // Handle the case where remaining elements are less than 16
            for (uint32_t j = i; j < num_elements; j++) {
                c[j] = a[j] + b[j]; // Non-intrinsic addition for remaining elements
            }
            return; // If we've got this far, there are no more elements left
        }
    }
#else
    for (uint32_t i = 0; i < num_elements; i++) {
        c[i] = a[i] + b[i]; // Normal matrix addition
    }
#endif
}




/*
 * Matrix Subtraction
 *
 * This function subtracts matrix b from matrix a and stores the result in matrix c.
 *
 * @param c: Resultant matrix (2D array of integers, stored in row-major order)
 * @param a: Matrix operand A (2D array of integers, stored in row-major order)
 * @param b: Matrix operand B (2D array of integers, stored in row-major order)
 * @param size: Size of the matrices (number of rows or columns)
 *
 * @pre c, a, and b must not be NULL.
 * @pre size should be greater than 0.
 * @pre Matrices a, b, and c must be properly allocated to hold size*size elements.
 *
 * @post Matrix c contains the result of the subtraction of matrix b from matrix a.
 * @post If preconditions are not met, the behavior is undefined.
 */
__attribute__((always_inline))
static inline void matrix_sub(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size) {
    uint32_t num_elements = size * size;
#if defined(__x86_64__)
    for (uint32_t i = 0; i < num_elements; i += 16) {
        if (i + 16 <= num_elements) {  // So we do not go out of bounds
            __m256i vec_a = _mm256_load_si256((__m256i*)(a + i));  // Load 16 elements from a
            __m256i vec_b = _mm256_load_si256((__m256i*)(b + i));  // Load 16 elements from b
            __m256i vec_sum = _mm256_sub_epi16(vec_a, vec_b);      // Subtract the elements
            _mm256_store_si256((__m256i*)(c + i), vec_sum);        // Store the result in c
        } else {
            // Handle the case where remaining elements are less than 16
            for (uint32_t j = i; j < num_elements; j++) {
                c[j] = a[j] - b[j]; // Non-intrinsic subtraction for remaining elements
            }
            return; // If we've got this far, there are no more elements left
        }
    }
#else
    for (uint32_t i = 0; i < num_elements; i++) {
        c[i] = a[i] - b[i]; // Normal matrix subraction
    }
#endif
}

/*
 * Matrix Padding
 *
 * This function pads the source matrix with zeroes to match the dimensions of the destination matrix
 * and then copies the source matrix into the padded area of the destination matrix.
 *
 * @param dest: Destination matrix (2D array of integers, stored in row-major order)
 * @param src: Source matrix (2D array of integers, stored in row-major order)
 * @param dest_row: Number of rows in the destination matrix
 * @param dest_col: Number of columns in the destination matrix
 * @param src_row: Number of rows in the source matrix
 * @param src_col: Number of columns in the source matrix
 *
 * @pre dest and src must not be NULL.
 * @pre dest_row, dest_col, src_row, and src_col should be greater than 0.
 * @pre The size of dest should be large enough to accommodate src after padding.
 *
 * @post The destination matrix dest contains the padded source matrix src.
 * @post If preconditions are not met, the behavior is undefined.
 */
void matrix_pad(matrix_t dest, matrix_t __restrict src, uint16_t dest_row, uint16_t dest_col, uint16_t src_row, uint16_t src_col) {
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



/*
 * Matrix Unpadding
 *
 * This function copies data from a padded matrix to a destination matrix, removing padding.
 *
 * @param dest: Destination matrix to which data will be copied (2D array of integers, stored in row-major order)
 * @param src: Padded source matrix from which data will be copied (2D array of integers, stored in row-major order)
 * @param dest_row: Number of rows in the destination matrix
 * @param dest_col: Number of columns in the destination matrix
 * @param src_size: Size of the padded source matrix (number of rows or columns)
 *
 * @pre dest and src must not be NULL.
 * @pre dest_row and dest_col should be greater than 0.
 * @pre src_size should be greater than or equal to dest_row and dest_col.
 *
 * @post Data is copied from the padded source matrix to the destination matrix, removing padding.
 * @post If preconditions are not met, the behavior is undefined.
 */
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



/*
 * Matrix Equality Check (Unused)
 *
 * This function checks if two matrices are equal element-wise.
 * It is intended for validating matrix multiplication, but it is currently unused.
 *
 * @param a: First matrix to be compared (2D array of integers, stored in row-major order)
 * @param b: Second matrix to be compared (2D array of integers, stored in row-major order)
 * @param m: Number of rows in the matrices
 * @param l: Number of columns in the matrices
 *
 * @pre a and b must not be NULL.
 * @pre m and l should be greater than 0.
 *
 * @post If the matrices are equal element-wise, "Function is correct!" is printed; otherwise, "Function is wrong!" is printed.
 * @post If preconditions are not met, the behavior is undefined.
 */
__attribute__((always_inline, unused))
static inline void matrix_eq(matrix_t __restrict a, matrix_t __restrict b, uint16_t m, uint16_t l) {
    // Check if matrixes are equal (used for validating multiplication)
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < l; j++) {
            if (a[i * l + j] != b[i * l + j]) {
                printf("Function is wrong!\n");
                return;
            }
        }
    }
    printf("Function is correct!\n");
}
#pragma endregion // utils

// FrodoKEM Matrix Multiplication
// REFS: https://eprint.iacr.org/2021/711.pdf
//       https://github.com/microsoft/PQCrypto-LWEKE/blob/a2f9dec8917ccc3464b3378d46b140fa7353320d/FrodoKEM/src/frodo_macrify.c#L252
// TODO: try _mm_prefetch


/*
 * Matrix Multiplication
 *
 * This function multiplies matrix a and matrix b and stores the result in matrix c.
 * It uses optimized SIMD instructions for x86_64 architectures and a cache-efficient
 * algorithm for other architectures.
 *
 * @param c: Resultant matrix (2D array of integers, stored in row-major order)
 * @param a: Matrix operand A (2D array of integers, stored in row-major order)
 * @param b: Matrix operand B (2D array of integers, stored in row-major order)
 * @param m: Number of rows in matrix a and matrix c
 * @param n: Number of columns in matrix a and number of rows in matrix b
 * @param l: Number of columns in matrix b and matrix c
 *
 * @pre c, a, and b must not be NULL.
 * @pre m, n, and l should be greater than 0.
 * @pre Matrices a, b, and c must be properly allocated to hold m*n, n*l, and m*l elements respectively.
 * @pre Matrix a must have dimensions m x n.
 * @pre Matrix b must have dimensions n x l.
 * @pre Matrix c must have dimensions m x l.
 *
 * @post Matrix c contains the result of the multiplication of matrix a and matrix b.
 * @post If preconditions are not met, the behavior is undefined.
 */
 void matrix_multiply(matrix_t c, matrix_t a, matrix_t b, uint16_t m, uint16_t n, uint16_t l) 
 {
#if defined(__x86_64__)
    __m256i b_vec, acc_vec;
    for (uint16_t j = 0; j < m; j++) {
        for (uint16_t q = 0; q < l; q += 16) {
            acc_vec = _mm256_setzero_si256();  // Initialize acc_vec to zero

            for (uint16_t p = 0; p < n; p += 8) {  // Assuming n is a multiple of 8
                __m256i sp_vec[8];
                for (uint16_t k = 0; k < 8; k++) {
                    sp_vec[k] = _mm256_set1_epi16(a[j * n + p + k]);  // Broadcast elements from 'a'
                }

                for (uint16_t k = 0; k < 8; k++) {
                    b_vec = _mm256_load_si256((__m256i*)(b + (p + k) * l + q));  // Load 16 elements from 'b'
                    acc_vec = _mm256_add_epi16(acc_vec, _mm256_mullo_epi16(sp_vec[k], b_vec));  // Multiply and accumulate
                }
            }

            if (q + 16 <= l) {
                _mm256_store_si256((__m256i*)(c + j * l + q), acc_vec);  // Store the result back to 'c'
            } else {
                // Handle remaining columns that do not fill a full SIMD register width
                uint16_t temp[16];
                _mm256_storeu_si256((__m256i*)temp, acc_vec);
                for (uint16_t r = 0; r < l % 16; r++) {
                    c[j * l + q + r] = temp[r];
                }
            }
        }
    }
#else
#define BLOCK_SIZE 64
    // Zero out the result matrix
    uint16_t i, j, k, i0, j0, k0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < l; j++) {
            c[i * l + j] = 0;
        }
    }

    // Cache-efficient matrix algorithm
    for (i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        for (j0 = 0; j0 < l; j0 += BLOCK_SIZE) {
            for (k0 = 0; k0 < m; k0 += BLOCK_SIZE) {
                for (i = i0; i < i0 + BLOCK_SIZE && i < n; i++) {
                    for (j = j0; j < j0 + BLOCK_SIZE && j < l; j++) {
                        for (k = k0; k < k0 + BLOCK_SIZE && k < m; k++) {
                            c[i * l + j] += a[i * n + k] * b[k * l + j];
                        }
                    }
                }
            }
        }
    }
#endif
}

// Strassen Algorithm
// REF: https://en.wikipedia.org/wiki/Strassen_algorithm

/*
 * Strassen Algorithm for Matrix Multiplication
 *
 * This function multiplies matrix a and matrix b using the Strassen algorithm and stores the result in matrix c.
 * It switches to a conventional matrix multiplication algorithm if the size of the matrices falls below a certain threshold.
 *
 * @param c: Resultant matrix (2D array of integers, stored in row-major order)
 * @param a: Matrix operand A (2D array of integers, stored in row-major order)
 * @param b: Matrix operand B (2D array of integers, stored in row-major order)
 * @param size: Size of the matrices (number of rows or columns)
 *
 * @pre c, a, and b must not be NULL.
 * @pre size should be greater than 0.
 * @pre Matrices a, b, and c must be properly allocated to hold size*size elements.
 * @pre Matrices a and b must have the same dimensions.
 * @pre Size should be a power of 2.
 *
 * @post Matrix c contains the result of the multiplication of matrix a and matrix b using the Strassen algorithm.
 * @post If preconditions are not met, the behavior is undefined.
 */
static void strassen(matrix_t c, matrix_t __restrict a, matrix_t __restrict b, uint16_t size) {
    // I know this looks shady but it's the best one for cache-misses, doesn't matter small or big numbers, 512 does the trick.
    if (size <= 512) {
        matrix_multiply(c, a, b, size, size, size);
    } else {
        uint16_t new_size = size / 2;

        // Bootstrap submatrices
        matrix_t a11 = matrix_new(new_size, new_size);
        matrix_t a12 = matrix_new(new_size, new_size);
        matrix_t a21 = matrix_new(new_size, new_size);
        matrix_t a22 = matrix_new(new_size, new_size);
        
        matrix_t b11 = matrix_new(new_size, new_size);
        matrix_t b12 = matrix_new(new_size, new_size);
        matrix_t b21 = matrix_new(new_size, new_size);
        matrix_t b22 = matrix_new(new_size, new_size);

        matrix_t c11 = matrix_new(new_size, new_size);
        matrix_t c12 = matrix_new(new_size, new_size);
        matrix_t c21 = matrix_new(new_size, new_size);
        matrix_t c22 = matrix_new(new_size, new_size);

        matrix_t p1 = matrix_new(new_size, new_size);
        matrix_t p2 = matrix_new(new_size, new_size);
        matrix_t p3 = matrix_new(new_size, new_size);
        matrix_t p4 = matrix_new(new_size, new_size);
        matrix_t p5 = matrix_new(new_size, new_size);
        matrix_t p6 = matrix_new(new_size, new_size);
        matrix_t p7 = matrix_new(new_size, new_size);

        matrix_t tmp_a = matrix_new(new_size, new_size);
        matrix_t tmp_b = matrix_new(new_size, new_size);

        // Divide to submatrices
        for (uint16_t i = 0; i < new_size; i++) {
            for (uint16_t j = 0; j < new_size; j++) {
                a11[i * new_size + j] = a[i * size + j];
                a12[i * new_size + j] = a[i * size + j + new_size];
                a21[i * new_size + j] = a[(i + new_size) * size + j];
                a22[i * new_size + j] = a[(i + new_size) * size + j + new_size];
                
                b11[i * new_size + j] = b[i * size + j];
                b12[i * new_size + j] = b[i * size + j + new_size];
                b21[i * new_size + j] = b[(i + new_size) * size + j];
                b22[i * new_size + j] = b[(i + new_size) * size + j + new_size];
            }
        }
#if defined(DEBUG)
        matrix_print(stdout, a11, new_size, new_size);
        matrix_print(stdout, a12, new_size, new_size);
        matrix_print(stdout, a21, new_size, new_size);
        matrix_print(stdout, a22, new_size, new_size);
        
        matrix_print(stdout, b11, new_size, new_size);
        matrix_print(stdout, b12, new_size, new_size);
        matrix_print(stdout, b21, new_size, new_size);
        matrix_print(stdout, b22, new_size, new_size);
#endif

        // p1 = a11 * (b12 - b22)
        matrix_sub(tmp_b, b12, b22, new_size);
        strassen(p1, a11, tmp_b, new_size);

        // p2 = (a11 + a12) * b22
        matrix_add(tmp_a, a11, a12, new_size);
        strassen(p2, tmp_a, b22, new_size);
       
        // p3 = (a21 + a22) * b11 
        matrix_add(tmp_a, a21, a22, new_size);
        strassen(p3, tmp_a, b11, new_size);

        // p4 = a22 * (b21 - b11)
        matrix_sub(tmp_b, b21, b11, new_size);
        strassen(p4, a22, tmp_b, new_size);

        // p5 = (a11 + a22) * (b11 + b22)
        matrix_add(tmp_a, a11, a22, new_size);
        matrix_add(tmp_b, b11, b22, new_size);
        strassen(p5, tmp_a, tmp_b, new_size);

        // p6 = (a12 - a22) * (b21 + b22)    
        matrix_sub(tmp_a, a12, a22, new_size);
        matrix_add(tmp_b, b21, b22, new_size);
        strassen(p6, tmp_a, tmp_b, new_size);

        // p7 = (a11 - a21) * (b11 + b12)
        matrix_sub(tmp_a, a11, a21, new_size);
        matrix_add(tmp_b, b11, b12, new_size);
        strassen(p7, tmp_a, tmp_b, new_size);

        // c11 = p5 + p4 - p2 + p6
        matrix_add(c11, p5, p4, new_size);
        matrix_sub(c11, c11, p2, new_size);
        matrix_add(c11, c11, p6, new_size);

        // c12 = p1 + p2
        matrix_add(c12, p1, p2, new_size);

        // c21 = p3 + p4
        matrix_add(c21, p3, p4, new_size);

        // c22 = p5 + p1 - p3 - p7
        matrix_add(c22, p5, p1, new_size);
        matrix_sub(c22, c22, p3, new_size);
        matrix_sub(c22, c22, p7, new_size);
        
        // Place the output to the relevant quadrants
        for (uint16_t i = 0; i < new_size; i++) {
            for (uint16_t j = 0; j < new_size; j++) {
                c[i * size + j] = c11[i * new_size + j]; 
                c[i * size + (j + new_size)] = c12[i * new_size + j]; 
                c[(i + new_size) * size + j] = c21[i * new_size + j]; 
                c[(i + new_size) * size + (j + new_size)] = c22[i * new_size + j]; 
            }
        }

        // Free used memory
        matrix_free(a11);
        matrix_free(a12);
        matrix_free(a21);
        matrix_free(a22);
        
        matrix_free(b11);
        matrix_free(b12);
        matrix_free(b21);
        matrix_free(b22);
        
        matrix_free(c11);
        matrix_free(c12);
        matrix_free(c21);
        matrix_free(c22);
       
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
/*
 * Matrix Preparation and Multiplication
 *
 * This function prepares matrices for multiplication by padding them and then performs matrix multiplication.
 * It uses the Strassen algorithm if the size of the matrices exceeds a certain threshold.
 *
 * @param c: Resultant matrix (2D array of integers, stored in row-major order)
 * @param a: Matrix operand A (2D array of integers, stored in row-major order)
 * @param b: Matrix operand B (2D array of integers, stored in row-major order)
 * @param m: Number of rows in matrix a and matrix c
 * @param n: Number of columns in matrix a and number of rows in matrix b
 * @param l: Number of columns in matrix b and matrix c
 *
 * @pre c, a, and b must not be NULL.
 * @pre m, n, and l should be greater than 0.
 * @pre Matrices a, b, and c must be properly allocated to hold m*n, n*l, and m*l elements respectively.
 *
 * @post Matrix c contains the result of the multiplication of matrix a and matrix b after proper padding and unpadding.
 * @post If preconditions are not met, the behavior is undefined.
 */
void matrix_prepare_and_mul(matrix_t c, matrix_t __restrict a, matrix_t __restrict b, uint16_t m, uint16_t n, uint16_t l) {
    // Get the padding size
    uint16_t new_size = round_size(max(max(m, n), l));
    new_size = new_size < 512 ? 512 : new_size;
    DBG("new_size: %hu", new_size);

    matrix_t padded_a = matrix_new(new_size, new_size);
    matrix_t padded_b = matrix_new(new_size, new_size);
    matrix_t padded_result = matrix_new(new_size, new_size);

    // Pad the matrix for the strassen algorithm to be able to divide them into equal 2^n lengthed parts
    matrix_pad(padded_a, a, new_size, new_size, m, n);
    matrix_pad(padded_b, b, new_size, new_size, n, l);

    strassen(padded_result, padded_a, padded_b, new_size);

    // Free the padded variables as they won't be necessary anymore
    matrix_free(padded_a);
    matrix_free(padded_b);

    // Remove the padding to print properly
    matrix_unpad(c, padded_result, m, l, new_size);
    matrix_free(padded_result);
}

int main(int argc, char **argv) {
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
    matrix_t d_matrix = matrix_new(m, l);

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

    matrix_prepare_and_mul(c_matrix, a_matrix, b_matrix, m, n, l);

    // Write the matrix to a file in order to keep the stdout clean.
    FILE *matrix_out = fopen("matrix_output_main.txt", "w");
    matrix_print(matrix_out, c_matrix, m, l);
    fclose(matrix_out);

    // Free used memory
    matrix_free(a_matrix);
    matrix_free(b_matrix);
    matrix_free(c_matrix);
    matrix_free(d_matrix);

    return 0;
}
