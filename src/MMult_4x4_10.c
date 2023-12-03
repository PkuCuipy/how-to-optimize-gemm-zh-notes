/* Create macros so that the matrices are stored in column-major order */
// 所以列向量的元素间距为 1, 而行向量的元素间距为 ldx
#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

/* Routine for computing C = A * B + C */

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

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE 
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3

typedef union {
    __m128d v;
    double d[2];
} v2df_t;

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
    // 这个版本中: 使用向量寄存器和向量指令.
    //

    v2df_t c_00_c_10_vreg, c_01_c_11_vreg, c_02_c_12_vreg, c_03_c_13_vreg,  // C
           c_20_c_30_vreg, c_21_c_31_vreg, c_22_c_32_vreg, c_23_c_33_vreg,
           a_0p_a_1p_vreg, a_2p_a_3p_vreg,                                  // A
           b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;                      // B

    double *b_p0_ptr = &B(0, 0), 
           *b_p1_ptr = &B(0, 1), 
           *b_p2_ptr = &B(0, 2), 
           *b_p3_ptr = &B(0, 3);   // 分别指向 B 的四个列中的同一行元素

    c_00_c_10_vreg.v = _mm_setzero_pd();  
    c_01_c_11_vreg.v = _mm_setzero_pd(); 
    c_02_c_12_vreg.v = _mm_setzero_pd();
    c_03_c_13_vreg.v = _mm_setzero_pd();
    c_20_c_30_vreg.v = _mm_setzero_pd();    
    c_21_c_31_vreg.v = _mm_setzero_pd();
    c_22_c_32_vreg.v = _mm_setzero_pd();
    c_23_c_33_vreg.v = _mm_setzero_pd();
    
    int p;

    for (p = 0; p < k; p++) {
        
        // a_0p_reg = A(0, p);
        // a_1p_reg = A(1, p);
        // a_2p_reg = A(2, p);
        // a_3p_reg = A(3, p);
        a_0p_a_1p_vreg.v = _mm_load_pd( &A(0, p) );   // pd: packed double
        a_2p_a_3p_vreg.v = _mm_load_pd( &A(2, p) );

        // b_p0_reg = *b_p0_ptr++;
        // b_p1_reg = *b_p1_ptr++;
        // b_p2_reg = *b_p2_ptr++;
        // b_p3_reg = *b_p3_ptr++;
        b_p0_vreg.v = _mm_loaddup_pd( b_p0_ptr++ );   // loaddup: load & duplicate
        b_p1_vreg.v = _mm_loaddup_pd( b_p1_ptr++ );   
        b_p2_vreg.v = _mm_loaddup_pd( b_p2_ptr++ );   
        b_p3_vreg.v = _mm_loaddup_pd( b_p3_ptr++ );   

        // 第一行 & 第二行
        // c_00_reg += a_0p_reg * b_p0_reg;
        // c_10_reg += a_1p_reg * b_p0_reg;
        c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;

        // c_01_reg += a_0p_reg * b_p1_reg;
        // c_11_reg += a_1p_reg * b_p1_reg;
        c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;

        // c_02_reg += a_0p_reg * b_p2_reg;
        // c_12_reg += a_1p_reg * b_p2_reg;
        c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;

        // c_03_reg += a_0p_reg * b_p3_reg;
        // c_13_reg += a_1p_reg * b_p3_reg;
        c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

        // 第三行 & 第四行
        // c_20_reg += a_2p_reg * b_p0_reg;
        // c_30_reg += a_3p_reg * b_p0_reg; 
        c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;

        // c_21_reg += a_2p_reg * b_p1_reg;
        // c_31_reg += a_3p_reg * b_p1_reg;
        c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;

        // c_22_reg += a_2p_reg * b_p2_reg;
        // c_32_reg += a_3p_reg * b_p2_reg;
        c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;

        // c_23_reg += a_2p_reg * b_p3_reg;
        // c_33_reg += a_3p_reg * b_p3_reg;
        c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;

    }

    C(0, 0) += c_00_c_10_vreg.d[0];   C(0, 1) += c_01_c_11_vreg.d[0];   C(0, 2) += c_02_c_12_vreg.d[0];   C(0, 3) += c_03_c_13_vreg.d[0];
    C(1, 0) += c_00_c_10_vreg.d[1];   C(1, 1) += c_01_c_11_vreg.d[1];   C(1, 2) += c_02_c_12_vreg.d[1];   C(1, 3) += c_03_c_13_vreg.d[1];

    C(2, 0) += c_20_c_30_vreg.d[0];   C(2, 1) += c_21_c_31_vreg.d[0];   C(2, 2) += c_22_c_32_vreg.d[0];   C(2, 3) += c_23_c_33_vreg.d[0];
    C(3, 0) += c_20_c_30_vreg.d[1];   C(3, 1) += c_21_c_31_vreg.d[1];   C(3, 2) += c_22_c_32_vreg.d[1];   C(3, 3) += c_23_c_33_vreg.d[1];

}
