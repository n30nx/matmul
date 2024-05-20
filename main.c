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

#include "matrix.h"
#include "mem.h"
#include "utils.h"
#include "mul.h"

#define SEQUENTIAL 1

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
        (void)scanf("%hu %hu %hu", &m, &n, &l);
    }

    // Create matrixes
    matrix_t a_matrix = matrix_new(m, n);
    matrix_t b_matrix = matrix_new(n, l);
    matrix_t c_matrix = matrix_new(m, l);

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
        matrix_pad(padded_a, a_matrix, new_size, m, n);
    }

    bool pad_b = new_size != n || new_size != l;
    matrix_t padded_b;
    if (!pad_b) {
        padded_b = b_matrix;
    } else {
        padded_b = matrix_new(new_size, new_size);
        matrix_pad(padded_b, b_matrix, new_size, n, l);
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
    //matrix_print(stdout, d_matrix, m, l);

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
