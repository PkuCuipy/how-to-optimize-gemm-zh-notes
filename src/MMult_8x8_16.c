#include <immintrin.h>

/* Create macros so that the matrices are stored in column-major order */
// 所以列向量的元素间距为 1, 而行向量的元素间距为 ldx
#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

/* Block sizes. where: A(m x k) @ B(k x n) -> C(m x n) */
#define m_max 256
#define k_max 256
#define n_max 2000

#define min(i, j) ((i < j) ? (i) : (j))


/* Routine for computing C = A * B + C */

void AddDot8x8(int k, double* a, double* b, double* c, int ldc);
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
    
    for (j = 0; j < n; j += 8) {
        if (need_pack_B)
            { PackMatrixB(k, &B(0, j), ldb, &packedB[j * k]); }
        for (i = 0; i < m; i += 8) {
            if (j == 0)  
                { PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]); }
            // 更新 C(i, j) 为左上角, C(i+7, j+7) 为右下角, 矩阵 C 的共 8x8 = 64 个元素
            AddDot8x8(k, &packedA[i * k], &packedB[k * j], &C(i, j), ldc);  
        }
    }
}


void PackMatrixA(int k, double* a, int lda, double* a_to) {
    // 把一个 8 x k 的子矩阵 打包到 a_to 的位置 (进行连续存储)
    int j;
    for (j = 0; j < k; j++) {
        double* a_ij_ptr = &A(0, j);
        *(a_to+0) = *(a_ij_ptr + 0);
        *(a_to+1) = *(a_ij_ptr + 1);
        *(a_to+2) = *(a_ij_ptr + 2);
        *(a_to+3) = *(a_ij_ptr + 3);
        *(a_to+4) = *(a_ij_ptr + 4);
        *(a_to+5) = *(a_ij_ptr + 5);
        *(a_to+6) = *(a_ij_ptr + 6);
        *(a_to+7) = *(a_ij_ptr + 7);
        a_to += 8;
    }
}


void PackMatrixB(int k, double* b, int ldb, double* b_to) {
    // 把一个 k x 8 的子矩阵 打包到 b_to 的位置 (进行连续存储)
    int i;
    double *b_i0_ptr = &B(0, 0), 
           *b_i1_ptr = &B(0, 1),
           *b_i2_ptr = &B(0, 2),
           *b_i3_ptr = &B(0, 3),
           *b_i4_ptr = &B(0, 4),
           *b_i5_ptr = &B(0, 5),
           *b_i6_ptr = &B(0, 6),
           *b_i7_ptr = &B(0, 7);

    register double b_i0, b_i1, b_i2, b_i3, b_i4, b_i5, b_i6, b_i7;

    for (i = 0; i < k; i++) {
        b_i0 = *b_i0_ptr++;
        b_i1 = *b_i1_ptr++;
        b_i2 = *b_i2_ptr++;
        b_i3 = *b_i3_ptr++;
        b_i4 = *b_i4_ptr++;
        b_i5 = *b_i5_ptr++;
        b_i6 = *b_i6_ptr++;
        b_i7 = *b_i7_ptr++;
        
        *b_to++ = b_i0;
        *b_to++ = b_i1;
        *b_to++ = b_i2;
        *b_to++ = b_i3;
        *b_to++ = b_i4;
        *b_to++ = b_i5;
        *b_to++ = b_i6;
        *b_to++ = b_i7;
    }
}


typedef union {
    __m512d v;
    double d[8];
} v8df_t;

void AddDot8x8(int k, double* a, double* b, double* c, int ldc) {
    //
    // 计算矩阵 C 的如下共 8 x 8 = 64 个元素:
    //
    //   C(0,0)  C(0,1)  ...  C(0,7)
    //   C(1,0)  C(1,1)  ...  C(1,7)
    //   C(2,0)  C(2,1)  ...  C(2,7)
    //      :       :            :
    //   C(7,0)  C(7,1)  ...  C(7,7)
    //

    v8df_t c00__c70_vreg,     // C
           c01__c71_vreg,      
           c02__c72_vreg,      
           c03__c73_vreg,      
           c04__c74_vreg,      
           c05__c75_vreg,      
           c06__c76_vreg,      
           c07__c77_vreg,     

           a0p__a7p_vreg,     // A

           bp0_dup8_vreg,     // B
           bp1_dup8_vreg,
           bp2_dup8_vreg,
           bp3_dup8_vreg,
           bp4_dup8_vreg,
           bp5_dup8_vreg,
           bp6_dup8_vreg,
           bp7_dup8_vreg;

    c00__c70_vreg.v = _mm512_setzero_pd();  
    c01__c71_vreg.v = _mm512_setzero_pd();  
    c02__c72_vreg.v = _mm512_setzero_pd();  
    c03__c73_vreg.v = _mm512_setzero_pd();  
    c04__c74_vreg.v = _mm512_setzero_pd();  
    c05__c75_vreg.v = _mm512_setzero_pd();  
    c06__c76_vreg.v = _mm512_setzero_pd();  
    c07__c77_vreg.v = _mm512_setzero_pd();  
    
    int p;

    for (p = 0; p < k; p++) {
        
        a0p__a7p_vreg.v = _mm512_load_pd( a );  
        a += 8;

        bp0_dup8_vreg.v = _mm512_set1_pd(*(b + 0)); 
        bp1_dup8_vreg.v = _mm512_set1_pd(*(b + 1));
        bp2_dup8_vreg.v = _mm512_set1_pd(*(b + 2));
        bp3_dup8_vreg.v = _mm512_set1_pd(*(b + 3));
        bp4_dup8_vreg.v = _mm512_set1_pd(*(b + 4));
        bp5_dup8_vreg.v = _mm512_set1_pd(*(b + 5));
        bp6_dup8_vreg.v = _mm512_set1_pd(*(b + 6));
        bp7_dup8_vreg.v = _mm512_set1_pd(*(b + 7));
        b += 8;

        // 在寄存器中累加
        c00__c70_vreg.v += a0p__a7p_vreg.v * bp0_dup8_vreg.v;
        c01__c71_vreg.v += a0p__a7p_vreg.v * bp1_dup8_vreg.v;
        c02__c72_vreg.v += a0p__a7p_vreg.v * bp2_dup8_vreg.v;
        c03__c73_vreg.v += a0p__a7p_vreg.v * bp3_dup8_vreg.v;
        c04__c74_vreg.v += a0p__a7p_vreg.v * bp4_dup8_vreg.v;
        c05__c75_vreg.v += a0p__a7p_vreg.v * bp5_dup8_vreg.v;
        c06__c76_vreg.v += a0p__a7p_vreg.v * bp6_dup8_vreg.v;
        c07__c77_vreg.v += a0p__a7p_vreg.v * bp7_dup8_vreg.v;

    }

    // 写回内存
    int i;
    
    for (i = 0; i < 8; i++) {
        C(i, 0) += c00__c70_vreg.d[i];
        C(i, 1) += c01__c71_vreg.d[i];
        C(i, 2) += c02__c72_vreg.d[i];
        C(i, 3) += c03__c73_vreg.d[i];
        C(i, 4) += c04__c74_vreg.d[i];
        C(i, 5) += c05__c75_vreg.d[i];
        C(i, 6) += c06__c76_vreg.d[i];
        C(i, 7) += c07__c77_vreg.d[i];
    }

}
