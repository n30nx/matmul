#include <stdlib.h>
#include <time.h>

#include "matrix.h"

__attribute__((constructor, always_inline))
static inline void _initalize_srand(void) {
    srand(time(NULL));
}
