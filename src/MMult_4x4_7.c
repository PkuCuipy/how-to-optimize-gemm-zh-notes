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
    // 这个版本中: 使用指针来索引 B 中的四个列

    register double c_00_reg = 0.0, c_01_reg = 0.0, c_02_reg = 0.0, c_03_reg = 0.0, 
                    c_10_reg = 0.0, c_11_reg = 0.0, c_12_reg = 0.0, c_13_reg = 0.0, 
                    c_20_reg = 0.0, c_21_reg = 0.0, c_22_reg = 0.0, c_23_reg = 0.0, 
                    c_30_reg = 0.0, c_31_reg = 0.0, c_32_reg = 0.0, c_33_reg = 0.0;

    register double a_0p_reg, 
                    a_1p_reg, 
                    a_2p_reg, 
                    a_3p_reg;

    double *b_p0_ptr, *b_p1_ptr, *b_p2_ptr, *b_p3_ptr;      // 分别指向 B 的四个列中元素
    
    int p;

    for (p = 0; p < k; p++) {
        
        a_0p_reg = A(0, p);
        a_1p_reg = A(1, p);
        a_2p_reg = A(2, p);
        a_3p_reg = A(3, p);

        b_p0_ptr = &B(p, 0);
        b_p1_ptr = &B(p, 1);
        b_p2_ptr = &B(p, 2);
        b_p3_ptr = &B(p, 3);

        // 第一行
        c_00_reg += a_0p_reg * *b_p0_ptr;
        c_01_reg += a_0p_reg * *b_p1_ptr;
        c_02_reg += a_0p_reg * *b_p2_ptr;
        c_03_reg += a_0p_reg * *b_p3_ptr;
        
        // 第二行
        c_10_reg += a_1p_reg * *b_p0_ptr;
        c_11_reg += a_1p_reg * *b_p1_ptr;
        c_12_reg += a_1p_reg * *b_p2_ptr;
        c_13_reg += a_1p_reg * *b_p3_ptr;

        // 第三行
        c_20_reg += a_2p_reg * *b_p0_ptr;
        c_21_reg += a_2p_reg * *b_p1_ptr;
        c_22_reg += a_2p_reg * *b_p2_ptr;
        c_23_reg += a_2p_reg * *b_p3_ptr;

        // 第四行
        c_30_reg += a_3p_reg * *b_p0_ptr++;     // 最后这里才执行指针移动!
        c_31_reg += a_3p_reg * *b_p1_ptr++;
        c_32_reg += a_3p_reg * *b_p2_ptr++;
        c_33_reg += a_3p_reg * *b_p3_ptr++;

    }

    C(0, 0) += c_00_reg;   C(0, 1) += c_01_reg;   C(0, 2) += c_02_reg;   C(0, 3) += c_03_reg;
    C(1, 0) += c_10_reg;   C(1, 1) += c_11_reg;   C(1, 2) += c_12_reg;   C(1, 3) += c_13_reg;
    C(2, 0) += c_20_reg;   C(2, 1) += c_21_reg;   C(2, 2) += c_22_reg;   C(2, 3) += c_23_reg;
    C(3, 0) += c_30_reg;   C(3, 1) += c_31_reg;   C(3, 2) += c_32_reg;   C(3, 3) += c_33_reg;

}
