#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

// These fcns are meant to make it easier to deal with
// matrices in C on the Beaglebone.

// Macros to extract matrix index and element.
// This assumes column major -- all rows of one column comes before the next column.
// n = ROWS*COLS.  Assume: for i=1:COLS; for j=1:ROWS
#define MATRIX_IDX(n, i, j) i*n + j
#define MATRIX_ELEMENT(A, m, n, i, j) A[ MATRIX_IDX(n, i, j) ]

// Min and max macros for scalars.
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// Function prototypes
void print_matrix(const float* A, int m, int n);
void print_matrix_linear(const float* A, int m, int n);
int lindex(int m, int n, int i, int j);
void zeros(int m, int n, float *A);
void eye(int m, int n, float *A);
void linspace(float x0, float x1, int N, float *v);
int maxeltf(int N, float *u);

#endif
