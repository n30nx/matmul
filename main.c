#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <limits.h>
#include <time.h>

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

#if defined(DEBUG)
    #define DBG(fmt, ...) do {                                  \
        fprintf(stdout, "%s:%d ", __FUNCTION__, __LINE__);      \
        fprintf(stdout, fmt, __VA_ARGS__);                      \
        fprintf(stdout, "\n");                                  \
    } while (0)
#else
    #define DBG(...)
#endif

#define max(a, b) a > b ? a : b

typedef int16_t *matrix_t;

#pragma region mem

/*
 * Create a New Matrix
 *
 * Belirtilen satır ve sütun sayısına sahip yeni bir matris oluşturur ve bellek tahsis eder.
 *
 * @param row: Oluşturulacak matrisin satır sayısı
 * @param col: Oluşturulacak matrisin sütun sayısı
 *
 * @pre row ve col sıfırdan farklı olmalıdır.
 *
 * @post Oluşturulan matris, her bir elemanı sıfır olan bir matris olmalıdır.
 * @post Matris belleği dinamik olarak tahsis edilmelidir.
 */
__attribute__((always_inline))
static inline matrix_t matrix_new(uint16_t row, uint16_t col) {
#if defined(__x86_64__)
    matrix_t new = (matrix_t)_mm_malloc(sizeof(int16_t) * row * col, 32);
#else
    matrix_t new = (matrix_t)malloc(sizeof(int16_t) * row * col);
#endif
    assert(new);
    DBG("allocated %hux%hu matrix with the size %lu", row, col, malloc_usable_size(new));
    return new;
}

/*
 * Free Matrix Memory
 *
 * Bu işlev, dinamik olarak tahsis edilmiş belleği serbest bırakır ve bir matrisin bellek sızıntısını önler.
 *
 * @param matrix: Serbest bırakılacak matris
 * @param row: Matrisin satır sayısı
 *
 * @pre matrix NULL olmamalıdır.
 * @pre row sıfırdan farklı olmalıdır.
 *
 * @post Matrisin belleği serbest bırakılmalıdır.
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
/*
 * Assign Random Values to Matrix
 *
 * Bu işlev, belirtilen satır ve sütun sayısına sahip bir matrise rastgele tam sayı değerleri atar.
 *
 * @param matrix: Değerlerin atanacağı matris
 * @param row: Matrisin satır sayısı
 * @param col: Matrisin sütun sayısı
 *
 * @pre matrix NULL olmamalıdır.
 * @pre row ve col sıfırdan farklı olmalıdır.
 *
 * @post Matris, her bir elemanı 0 ile INT16_MAX (hariç) arasında rastgele bir tam sayı ile doldurulmalıdır.
 */
__attribute__((always_inline))
static inline void matrix_assign_random(matrix_t matrix, uint16_t row, uint16_t col) {
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            matrix[i * col + j] = rand() % INT16_MAX;
        }
    }
}

/*
 * Print Matrix
 *
 * This function prints the elements of the matrix to the standard output.
 *
 * @param matrix: Matrix to be printed
 * @param row: Number of rows in the matrix
 * @param col: Number of columns in the matrix
 *
 * @pre matrix must not be NULL.
 * @pre row and col should be greater than 0.
 *
 * @post The elements of the matrix are printed to the standard output.
 */
__attribute__((always_inline))
static inline void matrix_print(matrix_t __restrict matrix, uint16_t row, uint16_t col) {
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            printf("%hd ", matrix[i * col + j]);
        }
        printf("\n");
    }
    printf("\n\n\n");
}

/*
 * Matrix Addition
 *
 * This function adds matrix a and matrix b element-wise and stores the result in matrix c.
 *
 * @param c: Resultant matrix
 * @param a: Matrix operand A
 * @param b: Matrix operand B
 * @param size: Size of the matrices (number of rows or columns)
 *
 * @pre c, a, and b must not be NULL.
 * @pre size should be greater than 0.
 *
 * @post Matrix c contains the result of the addition of matrix a and matrix b.
 */
__attribute__((always_inline))
static inline void matrix_add(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size) {
    uint32_t num_elements = size * size;
#if defined(__x86_64__)
    for (uint32_t i = 0; i < num_elements; i += 16) {
        if (i + 16 <= num_elements) {  // ensure we do not go out of bounds
            __m256i vec_a = _mm256_load_si256((__m256i*)&a[i]);  // load 8 elements from a
            __m256i vec_b = _mm256_load_si256((__m256i*)&b[i]);  // load 8 elements from b
            __m256i vec_sum = _mm256_add_epi16(vec_a, vec_b);     // add the elements
            _mm256_store_si256((__m256i*)&c[i], vec_sum);        // store the result in c
        } else {
            // handle the case where remaining elements are less than 8
            for (uint32_t j = i; j < num_elements; j++) {
                c[j] = a[j] + b[j]; // scalar addition for remaining elements
            }
            break; // exit the loop after handling remaining elements
        }
    }
#else
    for (uint32_t i = 0; i < num_elems; i++) {
        c[i] = a[i] + b[i];
    }
#endif
}

/*
 * Matrix Subtraction
 *
 * This function subtracts matrix b from matrix a element-wise and stores the result in matrix c.
 *
 * @param c: Resultant matrix
 * @param a: Matrix operand A
 * @param b: Matrix operand B
 * @param size: Size of the matrices (number of rows or columns)
 *
 * @pre c, a, and b must not be NULL.
 * @pre size should be greater than 0.
 *
 * @post Matrix c contains the result of the subtraction of matrix b from matrix a.
 */
__attribute__((always_inline))
static inline void matrix_sub(matrix_t c, matrix_t a, matrix_t __restrict b, uint16_t size) {
    uint32_t num_elements = size * size;
#if defined(__x86_64__)
    for (uint32_t i = 0; i < num_elements; i += 16) {
        if (i + 16 <= num_elements) {  // ensure we do not go out of bounds
            __m256i vec_a = _mm256_load_si256((__m256i*)&a[i]);  // load 8 elements from a
            __m256i vec_b = _mm256_load_si256((__m256i*)&b[i]);  // load 8 elements from b
            __m256i vec_sum = _mm256_sub_epi16(vec_a, vec_b);     // add the elements
            _mm256_store_si256((__m256i*)&c[i], vec_sum);        // store the result in c
        } else {
            // handle the case where remaining elements are less than 8
            for (uint32_t j = i; j < num_elements; j++) {
                c[j] = a[j] - b[j]; // scalar addition for remaining elements
            }
            break; // exit the loop after handling remaining elements
        }
    }
#else
    for (uint32_t i = 0; i < num_elems; i++) {
        c[i] = a[i] - b[i];
    }
#endif
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
__attribute__((always_inline))
static inline int u16_log2(size_t size) {
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    return log_table[((size - (size >> 1)) * 0x07EDD5E59A4E28C2) >> 58];
}

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
 * Pad Matrix
 *
 * This function performs padding to resize a matrix to a specified dimension.
 *
 * @param dest: Destination address of the padded matrix
 * @param src: Address of the original matrix to be padded
 * @param dest_row: Number of rows in the padded matrix
 * @param dest_col: Number of columns in the padded matrix
 * @param src_row: Number of rows in the original matrix
 * @param src_col: Number of columns in the original matrix
 *
 * @pre dest and src must not be NULL.
 * @pre dest_row and dest_col must be non-zero.
 * @pre src_row and src_col must be non-zero.
 * @pre The difference between dest_row and src_row should be less than or equal to the difference between src_col and dest_col.
 * @pre src_col value must be greater than or equal to dest_col value.
 *
 * @post The padded matrix should be filled with the data from the original matrix.
 * @post The empty part of the padded matrix should be filled with zeros.
 */
void matrix_pad(matrix_t dest, matrix_t __restrict src, uint16_t dest_row, uint16_t dest_col, uint16_t src_row, uint16_t src_col) {
    for (uint16_t i = 0; i < src_row; i++) {
        memcpy(dest + dest_col * i, src + src_col * i, src_col * sizeof(int16_t));  // Copy original data
        memset(dest + (dest_col * i) + src_col, 0, (dest_col - src_col) * sizeof(int16_t));  // Pad remaining columns with zeros
    }
    // Pad remaining rows with zeros
    for (int16_t i = src_row; i < dest_row; i++) {
        memset(dest + (dest_col * i), 0, dest_col * sizeof(int16_t));
    }
}

/*
 * Unpad Matrix
 *
 * This function performs unpadding to resize a matrix to a specified dimension.
 *
 * @param dest: Destination address of the unpadded matrix
 * @param src: Address of the original matrix to be unpadded
 * @param dest_row: Number of rows in the unpadded matrix
 * @param dest_col: Number of columns in the unpadded matrix
 * @param src_size: Size of the original square matrix
 *
 * @pre dest and src must not be NULL.
 * @pre dest_row and dest_col must be non-zero.
 * @pre src_size must be non-zero.
 *
 * @post The unpadded matrix should be filled with the data from the original matrix.
 */
__attribute__((always_inline))
static inline void matrix_unpad(matrix_t dest, matrix_t __restrict src, uint16_t dest_row, uint16_t dest_col, uint16_t src_size) {
    DBG("dest_start = %p, dest_end = %p, length = %hu", dest, dest + dest_row * dest_col, dest_row * dest_col);
    DBG("src_start = %p, src_end = %p, length = %hu", src, src + src_size * src_size, src_size * src_size);
    DBG("dest_row = %hu", dest_row);
    for (uint16_t i = 0; i < dest_row; i++) {
        DBG("copying to dest + %hu (%hu * %hu) from src + %hu (%hu * %hu), with the size %hu", dest_col * i, dest_col, i, src_size * i, src_size, i, dest_col);
        memcpy(dest + dest_col * i, src + src_size * i, sizeof(int16_t) * dest_col);
    }
}

__attribute__((always_inline))
static inline void matrix_eq(matrix_t __restrict a, matrix_t __restrict b, uint16_t m, uint16_t l) {
    for (uint32_t i = 0; i < m * l; i++) {
        if (a[i] != b[i]) {
            printf("Function is wrong!\n");
            return;
        }
    }
    printf("Function is correct!\n");
}
#pragma endregion // utils

// FrodoKEM Matrix Multiplication
// TODO: https://eprint.iacr.org/2021/711.pdf
//       https://github.com/microsoft/PQCrypto-LWEKE/blob/a2f9dec8917ccc3464b3378d46b140fa7353320d/FrodoKEM/src/frodo_macrify.c#L252
//
/*void matrix_multiply(matrix_t c, matrix_t __restrict a, matrix_t __restrict b, uint16_t m, uint16_t n, uint16_t l) {
    // Initialize the result matrix to zero
    memset(c, 0, sizeof(int16_t) * m * l);

    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t k = 0; k < n; k++) {
            // Broadcast a single element of A across the entire 256-bit register
            __m256i a_element = _mm256_set1_epi16(a[i * n + k]);

            for (uint16_t j = 0; j < l; j += 16) {  // Process 16 elements of B at a time
                // Load 16 elements of B
                __m256i b_elements = _mm256_load_si256((__m256i *)(b + k * l + j));
                // Load 16 elements of C to accumulate the result
                __m256i c_elements = _mm256_load_si256((__m256i *)(c + i * l + j));

                // Perform element-wise multiplication and add to the accumulator
                c_elements = _mm256_add_epi16(c_elements, _mm256_mullo_epi16(a_element, b_elements));

                // Store the result back to C
                _mm256_store_si256((__m256i *)(c + i * l + j), c_elements);
            }
        }
    }
}*/

void matrix_multiply(matrix_t c, matrix_t a, matrix_t b, uint16_t m, uint16_t n, uint16_t l) {
    // Assuming m corresponds to PARAMS_NBAR and n corresponds to PARAMS_N in the given context
    __m256i b_vec, acc_vec;
    for (uint16_t j = 0; j < m; j++) {
        for (uint16_t q = 0; q < l; q += 16) { // Assuming l is a multiple of 16
            acc_vec = _mm256_load_si256((__m256i*)(c + j * l + q)); // Load 16 elements from c (matrix 'e' in the original snippet)

            for (uint16_t p = 0; p < n; p += 8) { // Assuming n is a multiple of 8 and we use 8 for step in 'p'
                __m256i sp_vec[8];
                for (int k = 0; k < 8; k++) {
                    sp_vec[k] = _mm256_set1_epi16(a[j * n + p + k]); // Broadcast elements from 'a'
                }

                for (int k = 0; k < 8; k++) {
                    b_vec = _mm256_load_si256((__m256i*)(b + (p + k) * l + q)); // Load 16 elements from 'b'
                    acc_vec = _mm256_add_epi16(acc_vec, _mm256_mullo_epi16(sp_vec[k], b_vec)); // Multiply and accumulate
                }
            }

            _mm256_store_si256((__m256i*)(c + j * l + q), acc_vec); // Store the result back to 'c'
        }
    }
}


// Strassen Algorithm
// TODO: https://en.wikipedia.org/wiki/Strassen_algorithm
//       
/*
 * Strassen Matrix Multiplication
 *
 * This function implements the Strassen algorithm for matrix multiplication.
 *
 * @param c: Destination matrix for the result of multiplication
 * @param a: First matrix to be multiplied
 * @param b: Second matrix to be multiplied
 * @param size: Size of the matrices (assumed to be a power of 2)
 *
 * @pre c, a, and b must not be NULL.
 * @pre size must be a power of 2.
 *
 * @post The destination matrix c will contain the result of multiplying matrices a and b using the Strassen algorithm.
 */
//uint64_t recursion_count = 0;
static void strassen(matrix_t c, matrix_t __restrict a, matrix_t __restrict b, uint16_t size) {
    //++recursion_count;
    //printf("%lu\r", recursion_count);
    if (size == 1) {
       c[0] = a[0] * b[0];
    } else if (size == 32) {
       matrix_multiply(c, a, b, size, size, size); 
    } else {
        uint16_t new_size = size / 2;

        // bootstrapping submatrices
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

        // dividing to submatrices
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
        matrix_print(a11, new_size, new_size);
        matrix_print(a12, new_size, new_size);
        matrix_print(a21, new_size, new_size);
        matrix_print(a22, new_size, new_size);
        
        matrix_print(b11, new_size, new_size);
        matrix_print(b12, new_size, new_size);
        matrix_print(b21, new_size, new_size);
        matrix_print(b22, new_size, new_size);
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
        
        for (uint16_t i = 0; i < new_size; i++) {
            for (uint16_t j = 0; j < new_size; j++) {
                c[i * size + j] = c11[i * new_size + j]; 
                c[i * size + (j + new_size)] = c12[i * new_size + j]; 
                c[(i + new_size) * size + j] = c21[i * new_size + j]; 
                c[(i + new_size) * size + (j + new_size)] = c22[i * new_size + j]; 
            }
        }

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
 * Matrix Preparation and Multiplication using Strassen Algorithm
 *
 * This function prepares the input matrices by padding them to a suitable size for the Strassen algorithm
 * and then performs matrix multiplication using the Strassen algorithm.
 *
 * @param c: Destination matrix for the result of multiplication
 * @param a: First matrix to be multiplied
 * @param b: Second matrix to be multiplied
 * @param m: Number of rows in matrix a
 * @param n: Number of columns in matrix a (and number of rows in matrix b)
 * @param l: Number of columns in matrix b
 *
 * @pre c, a, and b must not be NULL.
 *
 * @post The destination matrix c will contain the result of multiplying matrices a and b.
 */
void matrix_prepare_and_mul(matrix_t c, matrix_t __restrict a, matrix_t __restrict b, uint16_t m, uint16_t n, uint16_t l) {
    uint16_t new_size = round_size(max(max(m, n), l));
    
    matrix_t padded_a = matrix_new(new_size, new_size);
    matrix_t padded_b = matrix_new(new_size, new_size);
    matrix_t padded_result = matrix_new(new_size, new_size);

    // Pad the matrix for the strassen algorithm to be able to divide them into equal 2^n lengthed parts
    matrix_pad(padded_a, a, new_size, new_size, m, n);
    matrix_pad(padded_b, b, new_size, new_size, n, l);

    printf("matrix a:\n");
    matrix_print(padded_a, new_size, new_size);
    printf("matrix b:\n");
    matrix_print(padded_b, new_size, new_size);

    strassen(padded_result, padded_a, padded_b, new_size);

    // Free the padded variables as they won't be necessary anymore
    matrix_free(padded_a);
    matrix_free(padded_b);

    // Remove the padding to print properly
    matrix_unpad(c, padded_result, m, l, new_size);
    matrix_free(padded_result);
}

/*
 * Matrix Multiplication
 *
 * This function performs matrix multiplication of matrices a and b and stores the result in matrix c.
 *
 * @param c: Destination matrix for the result of multiplication
 * @param a: First matrix to be multiplied
 * @param b: Second matrix to be multiplied
 * @param m: Number of rows in matrix a
 * @param n: Number of columns in matrix a (and number of rows in matrix b)
 * @param l: Number of columns in matrix b
 *
 * @pre c, a, and b must not be NULL.
 *
 * @post The destination matrix c will contain the result of multiplying matrices a and b.
 */

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
    /*for (uint16_t i = 0; i < m * n; i++) {
        a_matrix[i] = i + 1;
    }*/
    //printf("A matrix:\n");
    //matrix_print(a_matrix, m, n);
    
    matrix_assign_random(b_matrix, n, l);
    /*for (uint16_t i = 0; i < n * l; i++) {
        b_matrix[i] = i + 2;
    }*/
    //printf("B matrix:\n");
    //matrix_print(b_matrix, n, l);

    matrix_prepare_and_mul(c_matrix, a_matrix, b_matrix, m, n, l);
    printf("C matrix:\n");
    matrix_print(c_matrix, m, l);

    //matrix_multiply(d_matrix, a_matrix, b_matrix, m, n, l);
    //printf("D matrix:\n");
    //matrix_print(d_matrix, m, l);

    //matrix_eq(c_matrix, d_matrix, m, l);

    matrix_free(a_matrix);
    matrix_free(b_matrix);
    matrix_free(c_matrix);
    matrix_free(d_matrix);

    return 0;
}
