#include <immintrin.h> // AVX2 intrinsics
#include <time.h>
#include <assert.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#define PARAMS_N 1344
#define PARAMS_NBAR 8

#ifdef DEBUG
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
matrix_t matrix_new(uint16_t row, uint16_t col) {
    matrix_t new = (matrix_t)malloc(sizeof(int16_t) * row * col);
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
void matrix_free(matrix_t matrix) {
    free(matrix);
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
void matrix_assign_random(matrix_t matrix, uint16_t row, uint16_t col) {
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            matrix[i * col + j] = rand() % INT16_MAX;
        }
    }
}

/*
 * Print Matrix
 *
 * Bu işlev, belirtilen satır ve sütun sayısına sahip bir matrisi ekrana yazdırır.
 *
 * @param matrix: Yazdırılacak matris
 * @param row: Matrisin satır sayısı
 * @param col: Matrisin sütun sayısı
 *
 * @pre matrix NULL olmamalıdır.
 * @pre row ve col sıfırdan farklı olmalıdır.
 *
 * @post Matris, terminalde yazdırılmalıdır.
 */
void matrix_print(matrix_t matrix, uint16_t row, uint16_t col) {
    for (uint16_t i = 0; i < row; i++) {
        for (uint16_t j = 0; j < col; j++) {
            printf("%hd ", matrix[i * col + j]);
        }
        printf("\n");
    }
    printf("\n\n\n");
}

void matrix_add(matrix_t c, matrix_t a, matrix_t b, uint16_t size) {
    for (uint16_t i = 0; i < size; i++) {
        for (uint16_t j = 0; j < size; j++)
            c[i * size + j] = a[i * size + j] + b[i * size + j];
    }
}

void matrix_sub(matrix_t c, matrix_t a, matrix_t b, uint16_t size) {
    for (uint16_t i = 0; i < size; i++) {
        for (uint16_t j = 0; j < size; j++)
            c[i * size + j] = a[i * size + j] - b[i * size + j];
    }
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
void matrix_pad(matrix_t src, matrix_t dest, uint16_t src_row, uint16_t src_col, uint16_t dest_row, uint16_t dest_col) {
    for (uint16_t i = 0; i < src_row; i++) {
        memcpy(dest, src + (i * src_col), sizeof(int16_t) * src_row);
        memset(dest + (dest_col * i) + src_col, 0, sizeof(int16_t) * (dest_col - src_col));
    }

    memset(dest + (src_row * dest_col), 0, sizeof(int16_t) * ((dest_row * dest_col) - (src_row * dest_col)));
}*/

void matrix_pad(matrix_t dest, matrix_t src, uint16_t dest_row, uint16_t dest_col, uint16_t src_row, uint16_t src_col) {
    /*for (uint16_t i = 0; i < src_row; i++) {
        memcpy(dest + (i * dest_col), src + (i * src_col), src_col * sizeof(int16_t));
    }*/

    for (uint16_t i = 0; i < src_row; i++) {
        memcpy(dest + dest_col * i, src + src_col * i, src_col * sizeof(int16_t));  // Copy original data
        memset(dest + (dest_col * i) + src_col, 0, (dest_col - src_col) * sizeof(int16_t));  // Pad remaining columns with zeros
    }
    // Pad remaining rows with zeros
    for (int16_t i = src_row; i < dest_row; i++) {
        memset(dest + (dest_col * i), 0, dest_col * sizeof(int16_t));
    }
}

void matrix_unpad(matrix_t dest, matrix_t src, uint16_t dest_row, uint16_t dest_col, uint16_t src_size) {
    DBG("dest_start = %p, dest_end = %p, length = %hu", dest, dest + dest_row * dest_col, dest_row * dest_col);
    DBG("src_start = %p, src_end = %p, length = %hu", src, src + src_size * src_size, src_size * src_size);
    DBG("dest_row = %hu", dest_row);
    for (uint16_t i = 0; i < dest_row; i++) {
        DBG("copying to dest + %hu (%hu * %hu) from src + %hu (%hu * %hu), with the size %hu", dest_col * i, dest_col, i, src_size * i, src_size, i, dest_col);
        memcpy(dest + dest_col * i, src + src_size * i, sizeof(int) * dest_col);
    }
}
#pragma endregion // utils

// FrodoKEM Matrix Multiplication
// TODO: https://eprint.iacr.org/2021/711.pdf
//       https://github.com/microsoft/PQCrypto-LWEKE/blob/a2f9dec8917ccc3464b3378d46b140fa7353320d/FrodoKEM/src/frodo_macrify.c#L252
//
// Strassen Algorithm
// TODO: https://en.wikipedia.org/wiki/Strassen_algorithm
//       
/*
 * Matrix Multiplication Using AVX2 Intrinsics
 *
 * Çarpma işlemi sırasında kullanılan vektör işlemleri AVX2 SIMD (Single Instruction, Multiple Data) komutları kullanılarak gerçekleştirilir.
 * Bu işlev, matris a ve b'nin çarpımını hesaplar ve sonucu matris c'ye yazar.
 *
 * @param a: Çarpan matris A
 * @param b: Çarpan matris B
 * @param c: Sonuç matris C
 * @param m: A matrisinin satır sayısı
 * @param n: A matrisinin sütun sayısı ve B matrisinin satır sayısı
 * @param l: B matrisinin sütun sayısı ve C matrisinin sütun sayısı
 *
 * @pre a, b ve c NULL olmamalıdır.
 * @pre m, n ve l sıfırdan farklı olmalıdır.
 * @pre A matrisinin sütun sayısı (n) B matrisinin satır sayısına (n) eşit olmalıdır.
 * @pre A matrisinin satır sayısı (m), C matrisinin satır sayısına (m) eşit olmalıdır.
 * @pre B matrisinin sütun sayısı (l), C matrisinin sütun sayısına (l) eşit olmalıdır.
 *
 * @post C matrisi, A ve B matrislerinin çarpımını içermelidir.
 */
void strassen(matrix_t c, matrix_t a, matrix_t b, uint16_t size) {
    if (size == 1) {
        c[0] = a[0] * b[0];
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

        matrix_print(a11, new_size, new_size);
        matrix_print(a12, new_size, new_size);
        matrix_print(a21, new_size, new_size);
        matrix_print(a22, new_size, new_size);
        
        matrix_print(b11, new_size, new_size);
        matrix_print(b12, new_size, new_size);
        matrix_print(b21, new_size, new_size);
        matrix_print(b22, new_size, new_size);

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

void matrix_prepare_and_mul(matrix_t c, matrix_t a, matrix_t b, uint16_t m, uint16_t n, uint16_t l) {
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
 * Normal Matrix Multiplication
 *
 * Bu işlev, matris a ve b'nin çarpımını hesaplar ve sonucu matris c'ye yazar.
 * Çarpma işlemi, basit üçlü döngülerle gerçekleştirilir.
 *
 * @param a: Çarpan matris A
 * @param b: Çarpan matris B
 * @param c: Sonuç matris C
 * @param m: A matrisinin satır sayısı
 * @param n: A matrisinin sütun sayısı ve B matrisinin satır sayısı
 * @param l: B matrisinin sütun sayısı ve C matrisinin sütun sayısı
 *
 * @pre a, b ve c NULL olmamalıdır.
 * @pre m, n ve l sıfırdan farklı olmalıdır.
 * @pre A matrisinin sütun sayısı (n) B matrisinin satır sayısına (n) eşit olmalıdır.
 * @pre A matrisinin satır sayısı (m), C matrisinin satır sayısına (m) eşit olmalıdır.
 * @pre B matrisinin sütun sayısı (l), C matrisinin sütun sayısına (l) eşit olmalıdır.
 *
 * @post C matrisi, A ve B matrislerinin çarpımını içermelidir.
 */
void matrix_multiply(matrix_t c, matrix_t a, matrix_t b, uint16_t m, uint16_t n, uint16_t l) {
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < l; j++) {
            for (uint16_t k = 0; k < n; k++) {
                c[i * l + j] += a[i * n + k] * b[k * l + j];
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

    //matrix_assign_random(a_matrix, m, n);
    for (uint16_t i = 0; i < m * n; i++) {
        a_matrix[i] = i + 1;
    }
    printf("A matrix:\n");
    matrix_print(a_matrix, m, n);
    
    //matrix_assign_random(b_matrix, n, l);
    for (uint16_t i = 0; i < n * l; i++) {
        b_matrix[i] = i + 2;
    }
    printf("B matrix:\n");
    matrix_print(b_matrix, n, l);

    matrix_prepare_and_mul(c_matrix, a_matrix, b_matrix, m, n, l);
    printf("C matrix:\n");
    matrix_print(c_matrix, m, l);

    matrix_multiply(d_matrix, a_matrix, b_matrix, m, n, l);
    printf("D matrix:\n");
    matrix_print(d_matrix, m, l);

    matrix_free(a_matrix);
    matrix_free(b_matrix);
    matrix_free(c_matrix);
    matrix_free(d_matrix);

    return 0;   
}
