#include <immintrin.h>

/* Create macros so that the matrices are stored in column-major order */
// 所以列向量的元素间距为 1, 而行向量的元素间距为 ldx
#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

/* Block sizes. where: A(m x k) @ B(k x n) -> C(m x n) */
#define m_max 256
#define k_max 128
#define n_max 2000

#define min(i, j) ((i < j) ? (i) : (j))


/* Routine for computing C = A * B + C */

void AddDot16x4(int k, double* a, double* b, double* c, int ldc);
void InnerKernel(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc, int need);
void PackMatrixA(int, double*, int, double*);
void PackMatrixB(int, double*, int, double*);

void MY_MMult(int m, int n, int k, double *a, int lda, 
                                   double *b, int ldb, 
                                   double *c, int ldc) {
    int i, p, pb, ib;   
    // i 是对 m 个行的索引
    // p 是对 k 个列的索引
    // b 后缀表示 blocksize

    // 这里进行矩阵的 ｢分块乘法｣:
    // 每次计算中, 对 A 的 (ib x pb) 大小的块 和 B 的 (pb x n) 大小的块 进行矩阵乘, 
    // 得到的结果对应于 C 的 (ib x n) 大小的块 的部分结果.
    for (p = 0; p < k; p += k_max) {  
        pb = min(k - p, k_max);
        for (i = 0; i < m; i += m_max) {
            ib = min(m - i, m_max);
            InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, (i == 0));  // ｢A 的多个分块 ｣ 都与 ｢B 的同一个分块｣ 相乘, 
                                                                                            // 因此后者只需要在第一次被 pack, 故这里是 (i == 0)
        }
    }
}


void InnerKernel(int m, int n, int k, double *a, int lda, 
                                      double *b, int ldb, 
                                      double *c, int ldc, int need_pack_B) {
    int i, j;
    
    static double 
      packedA[m_max * k_max] __attribute__((aligned(8*sizeof(double)))),    // 64 字节对齐的 double packedA[m * k]; 也可以用 _mm_malloc 动态分配
      packedB[k_max * n_max];
    
    for (j = 0; j < n; j += 4) {
        if (need_pack_B)
            { PackMatrixB(k, &B(0, j), ldb, &packedB[j * k]); }
        for (i = 0; i < m; i += 16) {
            if (j == 0)  
                { PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]); }
            // 更新 C(i, j) 为左上角, C(i+15, j+3) 为右下角, 矩阵 C 的共 16x4 = 64 个元素
            AddDot16x4(k, &packedA[i * k], &packedB[k * j], &C(i, j), ldc);  
        }
    }
}


void PackMatrixA(int k, double* a, int lda, double* a_to) {
    // 把一个 16 x k 的子矩阵 打包到 a_to 的位置 (进行连续存储)
    int j;
    for (j = 0; j < k; j++) {
        double* a_ij_ptr = &A(0, j);
        *(a_to + 0x0) = *(a_ij_ptr + 0x0);
        *(a_to + 0x1) = *(a_ij_ptr + 0x1);
        *(a_to + 0x2) = *(a_ij_ptr + 0x2);
        *(a_to + 0x3) = *(a_ij_ptr + 0x3);
        *(a_to + 0x4) = *(a_ij_ptr + 0x4);
        *(a_to + 0x5) = *(a_ij_ptr + 0x5);
        *(a_to + 0x6) = *(a_ij_ptr + 0x6);
        *(a_to + 0x7) = *(a_ij_ptr + 0x7);
        *(a_to + 0x8) = *(a_ij_ptr + 0x8);
        *(a_to + 0x9) = *(a_ij_ptr + 0x9);
        *(a_to + 0xA) = *(a_ij_ptr + 0xA);
        *(a_to + 0xB) = *(a_ij_ptr + 0xB);
        *(a_to + 0xC) = *(a_ij_ptr + 0xC);
        *(a_to + 0xD) = *(a_ij_ptr + 0xD);
        *(a_to + 0xE) = *(a_ij_ptr + 0xE);
        *(a_to + 0xF) = *(a_ij_ptr + 0xF);
        a_to += 16;
    }
}


void PackMatrixB(int k, double* b, int ldb, double* b_to) {
    // 把一个 k x 4 的子矩阵 打包到 b_to 的位置 (进行连续存储)
    int i;
    double *b_i0_ptr = &B(0, 0), 
           *b_i1_ptr = &B(0, 1),
           *b_i2_ptr = &B(0, 2),
           *b_i3_ptr = &B(0, 3);

    register double b_i0, b_i1, b_i2, b_i3;

    for (i = 0; i < k; i++) {
        b_i0 = *b_i0_ptr++;
        b_i1 = *b_i1_ptr++;
        b_i2 = *b_i2_ptr++;
        b_i3 = *b_i3_ptr++;
        
        *b_to++ = b_i0;
        *b_to++ = b_i1;
        *b_to++ = b_i2;
        *b_to++ = b_i3;
    }
}


void AddDot16x4(int k, double* a, double* b, double* c, int ldc) {
    //
    // 计算矩阵 C 的如下共 16 x 4 = 64 个元素:
    //
    //   C(0,0)   C(0,1)   C(0,2)   C(0,3)
    //   C(1,0)   C(1,1)   C(1,2)   C(1,3)
    //   C(2,0)   C(2,1)   C(2,2)   C(2,3)
    //      :        :        :        :
    //   C(15,0)  C(15,1)  C(15,3)  C(15,3)
    //

    __m512d c00__c70_vreg,  c01__c71_vreg,  c02__c72_vreg,  c03__c73_vreg,  // C  
            c80__cF0_vreg,  c81__cF1_vreg,  c82__cF2_vreg,  c83__cF3_vreg,

            a0p__a7p_vreg,                                                  // A
            a8p__aFp_vreg,

            bp0_dup8_vreg,  bp1_dup8_vreg,  bp2_dup8_vreg,  bp3_dup8_vreg;  // B

    c00__c70_vreg = _mm512_setzero_pd();  
    c80__cF0_vreg = _mm512_setzero_pd();  
    c01__c71_vreg = _mm512_setzero_pd();  
    c81__cF1_vreg = _mm512_setzero_pd();  
    c02__c72_vreg = _mm512_setzero_pd();  
    c82__cF2_vreg = _mm512_setzero_pd();  
    c03__c73_vreg = _mm512_setzero_pd();   
    c83__cF3_vreg = _mm512_setzero_pd();   
    
    int p;

    for (p = 0; p < k; p++) {
        
        a0p__a7p_vreg = _mm512_load_pd( a );  
        a8p__aFp_vreg = _mm512_load_pd( a + 8 );  
        a += 16;

        bp0_dup8_vreg = _mm512_set1_pd(*(b + 0)); 
        bp1_dup8_vreg = _mm512_set1_pd(*(b + 1));
        bp2_dup8_vreg = _mm512_set1_pd(*(b + 2));
        bp3_dup8_vreg = _mm512_set1_pd(*(b + 3));
        b += 4;

        // 在寄存器中累加
        c00__c70_vreg += a0p__a7p_vreg * bp0_dup8_vreg;
        c80__cF0_vreg += a8p__aFp_vreg * bp0_dup8_vreg;
        c01__c71_vreg += a0p__a7p_vreg * bp1_dup8_vreg;
        c81__cF1_vreg += a8p__aFp_vreg * bp1_dup8_vreg;
        c02__c72_vreg += a0p__a7p_vreg * bp2_dup8_vreg;
        c82__cF2_vreg += a8p__aFp_vreg * bp2_dup8_vreg;
        c03__c73_vreg += a0p__a7p_vreg * bp3_dup8_vreg;
        c83__cF3_vreg += a8p__aFp_vreg * bp3_dup8_vreg;

    }

    // 写回内存
    double C_incr[64] __attribute__((aligned(64)));
    _mm512_store_pd(C_incr + 0x00, c00__c70_vreg);
    _mm512_store_pd(C_incr + 0x08, c80__cF0_vreg);
    _mm512_store_pd(C_incr + 0x10, c01__c71_vreg);
    _mm512_store_pd(C_incr + 0x18, c81__cF1_vreg);
    _mm512_store_pd(C_incr + 0x20, c02__c72_vreg);
    _mm512_store_pd(C_incr + 0x28, c82__cF2_vreg);
    _mm512_store_pd(C_incr + 0x30, c03__c73_vreg);
    _mm512_store_pd(C_incr + 0x38, c83__cF3_vreg);

    for (int j = 0; j < 4; j++) {
        C( 0, j) += C_incr[j * 16 +  0];
        C( 1, j) += C_incr[j * 16 +  1];
        C( 2, j) += C_incr[j * 16 +  2];
        C( 3, j) += C_incr[j * 16 +  3];
        C( 4, j) += C_incr[j * 16 +  4];
        C( 5, j) += C_incr[j * 16 +  5];
        C( 6, j) += C_incr[j * 16 +  6];
        C( 7, j) += C_incr[j * 16 +  7];
        C( 8, j) += C_incr[j * 16 +  8];
        C( 9, j) += C_incr[j * 16 +  9];
        C(10, j) += C_incr[j * 16 + 10];
        C(11, j) += C_incr[j * 16 + 11];
        C(12, j) += C_incr[j * 16 + 12];
        C(13, j) += C_incr[j * 16 + 13];
        C(14, j) += C_incr[j * 16 + 14];
        C(15, j) += C_incr[j * 16 + 15];
    }

}
