/* Create macros so that the matrices are stored in column-major order */
// 所以列向量的元素间距为 1, 而行向量的元素间距为 ldx
#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

/* Routine for computing C = A * B + C */

void AddDot(int, double*, int, double*, double*);
void AddDot4x4(int k, double* a, int lda, double* b, int ldb, double* c, int ldc);

void MY_MMult(int m, int n, int k, double *a, int lda, 
                                   double *b, int ldb, 
                                   double *c, int ldc) {
    int i, j;

    for (j = 0; j < n; j += 4) {     /* Loop over the columns of C, 并进行 4 层循环展开 */
        for (i = 0; i < m; i += 4) {         /* Loop over the rows of C, 进行 4 层循环展开 */
            // 更新 C(i, j) 为左上角, C(i+3, j+3) 为右下角, 矩阵 C 的共 4x4 = 16 个元素
            AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}


void AddDot4x4(int k, double* a, int lda, double* b, int ldb, double* c, int ldc) {
    //
    // 计算矩阵 C 的如下共 4 x 4 = 16 个元素:
    //
    //   C(0, 0) ... C(0, 3)
    //      :    ...    :
    //   C(3, 0) ... C(3, 3)
    //
    // 注意到调用时传入的 c 其实是 C(i, j), 因此实际计算的是原矩阵的:
    //
    //   C(i, j)      ...    C(i, j + 3)
    //      :         ...         :
    //  C(i + 3, 0)   ...   C(i + 3, j + 3)
    //

    // 第一行
    AddDot(k, &A(0, 0), lda, &B(0, 0), &C(0, 0));
    AddDot(k, &A(0, 0), lda, &B(0, 1), &C(0, 1));
    AddDot(k, &A(0, 0), lda, &B(0, 2), &C(0, 2));
    AddDot(k, &A(0, 0), lda, &B(0, 3), &C(0, 3));

    // 第二行
    AddDot(k, &A(1, 0), lda, &B(0, 0), &C(1, 0));
    AddDot(k, &A(1, 0), lda, &B(0, 1), &C(1, 1));
    AddDot(k, &A(1, 0), lda, &B(0, 2), &C(1, 2));
    AddDot(k, &A(1, 0), lda, &B(0, 3), &C(1, 3));

    // 第三行
    AddDot(k, &A(2, 0), lda, &B(0, 0), &C(2, 0));
    AddDot(k, &A(2, 0), lda, &B(0, 1), &C(2, 1));
    AddDot(k, &A(2, 0), lda, &B(0, 2), &C(2, 2));
    AddDot(k, &A(2, 0), lda, &B(0, 3), &C(2, 3));

    // 第四行
    AddDot(k, &A(3, 0), lda, &B(0, 0), &C(3, 0));
    AddDot(k, &A(3, 0), lda, &B(0, 1), &C(3, 1));
    AddDot(k, &A(3, 0), lda, &B(0, 2), &C(3, 2));
    AddDot(k, &A(3, 0), lda, &B(0, 3), &C(3, 3));

}


void AddDot(int len, double* x, int incx, double* y, double* gamma) {
    // 计算 gamma := x' * y + gamma 
    // 这里 x, y 是长度为 len 的向量, 而 gamma 是一个数
    // x 的步长为 incx, 而 y 的步长为 1

    /* 行向量 x 的第 i 个元素 */
    #define X(i) x[i * incx]

    int p;
    for (p = 0; p < len; p++) {
        *gamma += X(p) * y[p];
    }   
    
}