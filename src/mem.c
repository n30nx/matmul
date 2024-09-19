#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "mem.h"
#include "matrix.h"

#if defined(__x86_64__)
    #include <immintrin.h>
#endif

__attribute__((always_inline))
inline matrix_t matrix_new(uint16_t row, uint16_t col) {
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
inline void matrix_free(matrix_t matrix) {
#if defined(__x86_64__)
    _mm_free(matrix);
#else
    free(matrix);
#endif
}
