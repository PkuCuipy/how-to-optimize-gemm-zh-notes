/* Create macros so that the matrices are stored in column-major order */
// 所以列向量的元素间距为 1, 而行向量的元素间距为 ldx
#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

/* Routine for computing C = A * B + C */

void AddDot1x4(int, double*, int, double*, int, double*, int);

void MY_MMult(int m, int n, int k, double *a, int lda, 
                                   double *b, int ldb, 
                                   double *c, int ldc) {
    int i, j;

    for (j = 0; j < n; j += 4) {     /* Loop over the columns of C, 并进行 4 层循环展开 */
        for (i = 0; i < m; i += 1) {         /* Loop over the rows of C */
            // 一次更新 C(i, j), C(i, j + 1), C(i, j + 2) 和 C(i, j + 3) 四个元素.
            AddDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void AddDot1x4(int k, double* a, int lda,
                      double* b, int ldb,
                      double* c, int ldc) {
    // 计算矩阵 C 的 C(0, 0), C(0, 1), C(0, 2) 和 C(0, 3) 四个元素.
    // 注意到在调用这个例程时, 实际传入的 c 是 C(i, j) 的地址,
    // 因此此时计算的是原矩阵的 C(i, j), C(i, j + 1), C(i, j + 2) 和 C(i, j + 3).

    // 这个版本把 4 个循环写到同一个循环中
    int p;

    // AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    // AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    // AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    // AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));
    for (p = 0; p < k; p++) {
        C(0, 0) += A(0, p) * B(p, 0);
        C(0, 1) += A(0, p) * B(p, 1);
        C(0, 2) += A(0, p) * B(p, 2);
        C(0, 3) += A(0, p) * B(p, 3);
    }

}
