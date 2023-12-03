

void copy_matrix(int m, int n, double *a, int lda, double *b, int ldb) {
    #define A(i, j) a[(j)*lda + (i)]
    #define B(i, j) b[(j)*ldb + (i)]

    int i, j;

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            B(i, j) = A(i, j);
        }
    }
}

