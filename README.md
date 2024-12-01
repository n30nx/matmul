In this repository, I have integrated Strassen's Matrix Multiplication with FrodoKEM's AVX2-based Matrix Multiplication, adding a size switch to optimize execution time and reduce cache miss rates.

Project structure
- matrix.c: redundant file, just initializes the rand.
- mul.c: matrix multiplication functions (includes strassen and FrodoKEM's algorithm)
- mem.c: matrix allocation and deallocation functions
- ops.c: simple matrix operations (addition and subtraction)
- utils.c: handles matrix padding, unpadding, filling with random data, equality checks, and includes a non-trivial logarithm function (using a de Bruijn sequence).
