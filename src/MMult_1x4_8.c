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

    // 这个版本的变化: 进行 4x1 循环展开

    int p;
    register double c_00_reg, c_01_reg, c_02_reg, c_03_reg;     // C(0, 0) ... C(0, 3)
    register double a_0p_reg;                                   // A(0, p)

    c_00_reg = 0.0;
    c_01_reg = 0.0;
    c_02_reg = 0.0;
    c_03_reg = 0.0;

    double *b_p0_ptr, *b_p1_ptr, *b_p2_ptr, *b_p3_ptr;
    b_p0_ptr = &B(0, 0);
    b_p1_ptr = &B(0, 1);
    b_p2_ptr = &B(0, 2);
    b_p3_ptr = &B(0, 3);

    // 在寄存器中累加
    for (p = 0; p < k; p += 4) {
        a_0p_reg = A(0, p);
        c_00_reg += a_0p_reg * (*(b_p0_ptr++));
        c_01_reg += a_0p_reg * (*(b_p1_ptr++));
        c_02_reg += a_0p_reg * (*(b_p2_ptr++));
        c_03_reg += a_0p_reg * (*(b_p3_ptr++));

        a_0p_reg = A(0, p + 1);
        c_00_reg += a_0p_reg * (*(b_p0_ptr++));
        c_01_reg += a_0p_reg * (*(b_p1_ptr++));
        c_02_reg += a_0p_reg * (*(b_p2_ptr++));
        c_03_reg += a_0p_reg * (*(b_p3_ptr++));

        a_0p_reg = A(0, p + 2);
        c_00_reg += a_0p_reg * (*(b_p0_ptr++));
        c_01_reg += a_0p_reg * (*(b_p1_ptr++));
        c_02_reg += a_0p_reg * (*(b_p2_ptr++));
        c_03_reg += a_0p_reg * (*(b_p3_ptr++));

        a_0p_reg = A(0, p + 3);
        c_00_reg += a_0p_reg * (*(b_p0_ptr++));
        c_01_reg += a_0p_reg * (*(b_p1_ptr++));
        c_02_reg += a_0p_reg * (*(b_p2_ptr++));
        c_03_reg += a_0p_reg * (*(b_p3_ptr++));
    }

    // 最后统一写回
    C(0, 0) += c_00_reg;
    C(0, 1) += c_01_reg;
    C(0, 2) += c_02_reg;
    C(0, 3) += c_03_reg;

}
