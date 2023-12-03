#include <immintrin.h>

/* Create macros so that the matrices are stored in column-major order */
// 所以列向量的元素间距为 1, 而行向量的元素间距为 ldx
#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

/* Block sizes. where: A(m x k) @ B(k x n) -> C(m x n) */
#define m_max 24*10
#define k_max 24*10
#define n_max 2000

#define min(i, j) ((i < j) ? (i) : (j))


/* Routine for computing C = A * B + C */

void AddDot24x8(int k, double* a, double* b, double* c, int ldc);
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
        for (i = 0; i < m; i += 24) {
            if (j == 0)  
                { PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]); }
            // 更新 C(i, j) 为左上角, C(i+15, j+7) 为右下角, 矩阵 C 的共 16x8 = 128 个元素
            AddDot24x8(k, &packedA[i * k], &packedB[k * j], &C(i, j), ldc);  
        }
    }
}


void PackMatrixA(int k, double* a, int lda, double* a_to) {
    // 把一个 24 x k 的子矩阵 打包到 a_to 的位置 (进行连续存储)
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
        *(a_to+8) = *(a_ij_ptr + 8);
        *(a_to+9) = *(a_ij_ptr + 9);
        *(a_to+10) = *(a_ij_ptr + 10);
        *(a_to+11) = *(a_ij_ptr + 11);
        *(a_to+12) = *(a_ij_ptr + 12);
        *(a_to+13) = *(a_ij_ptr + 13);
        *(a_to+14) = *(a_ij_ptr + 14);
        *(a_to+15) = *(a_ij_ptr + 15);
        *(a_to+16) = *(a_ij_ptr + 16);
        *(a_to+17) = *(a_ij_ptr + 17);
        *(a_to+18) = *(a_ij_ptr + 18);
        *(a_to+19) = *(a_ij_ptr + 19);
        *(a_to+20) = *(a_ij_ptr + 20);
        *(a_to+21) = *(a_ij_ptr + 21);
        *(a_to+22) = *(a_ij_ptr + 22);
        *(a_to+23) = *(a_ij_ptr + 23);
        a_to += 24;
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


void AddDot24x8(int k, double* a, double* b, double* c, int ldc) {
    //
    // 计算矩阵 C 的如下共 24 x 8 = 64 个元素:
    //
    //   C(0,0)    C(0,1)  ...  C(0,7)
    //   C(1,0)    C(1,1)  ...  C(1,7)
    //   C(2,0)    C(2,1)  ...  C(2,7)
    //      :        :            :
    //   C(23,0)  C(23,1)  ...  C(23,7)
    //

    __m512d c00__c70_vreg,     // C
            c01__c71_vreg,      
            c02__c72_vreg,      
            c03__c73_vreg,      
            c04__c74_vreg,      
            c05__c75_vreg,      
            c06__c76_vreg,      
            c07__c77_vreg, 

            c80__cf0_vreg, 
            c81__cf1_vreg,      
            c82__cf2_vreg,      
            c83__cf3_vreg,      
            c84__cf4_vreg,      
            c85__cf5_vreg,      
            c86__cf6_vreg,      
            c87__cf7_vreg,   

            cg0__cn0_vreg, 
            cg1__cn1_vreg,      
            cg2__cn2_vreg,      
            cg3__cn3_vreg,      
            cg4__cn4_vreg,      
            cg5__cn5_vreg,      
            cg6__cn6_vreg,      
            cg7__cn7_vreg,   


            a0p__a7p_vreg,     // A
            a8p__afp_vreg,
            agp__anp_vreg,


            bp0_dup8_vreg,     // B
            bp1_dup8_vreg,
            bp2_dup8_vreg,
            bp3_dup8_vreg,
            bp4_dup8_vreg,
            bp5_dup8_vreg,
            bp6_dup8_vreg,
            bp7_dup8_vreg;

    c00__c70_vreg = _mm512_setzero_pd();  
    c01__c71_vreg = _mm512_setzero_pd();  
    c02__c72_vreg = _mm512_setzero_pd();  
    c03__c73_vreg = _mm512_setzero_pd();  
    c04__c74_vreg = _mm512_setzero_pd();  
    c05__c75_vreg = _mm512_setzero_pd();  
    c06__c76_vreg = _mm512_setzero_pd();  
    c07__c77_vreg = _mm512_setzero_pd();  
    c80__cf0_vreg = _mm512_setzero_pd();  
    c81__cf1_vreg = _mm512_setzero_pd();  
    c82__cf2_vreg = _mm512_setzero_pd();  
    c83__cf3_vreg = _mm512_setzero_pd();  
    c84__cf4_vreg = _mm512_setzero_pd();  
    c85__cf5_vreg = _mm512_setzero_pd();  
    c86__cf6_vreg = _mm512_setzero_pd();  
    c87__cf7_vreg = _mm512_setzero_pd(); 
    cg0__cn0_vreg = _mm512_setzero_pd();  
    cg1__cn1_vreg = _mm512_setzero_pd();  
    cg2__cn2_vreg = _mm512_setzero_pd();  
    cg3__cn3_vreg = _mm512_setzero_pd();  
    cg4__cn4_vreg = _mm512_setzero_pd();  
    cg5__cn5_vreg = _mm512_setzero_pd();  
    cg6__cn6_vreg = _mm512_setzero_pd();  
    cg7__cn7_vreg = _mm512_setzero_pd(); 

    int p;

    for (p = 0; p < k; p++) {
        
        a0p__a7p_vreg = _mm512_load_pd( a );  
        a8p__afp_vreg = _mm512_load_pd( a + 8 );  
        agp__anp_vreg = _mm512_load_pd( a + 16 );  
        a += 24;

        bp0_dup8_vreg = _mm512_set1_pd(*(b + 0)); 
        bp1_dup8_vreg = _mm512_set1_pd(*(b + 1));
        bp2_dup8_vreg = _mm512_set1_pd(*(b + 2));
        bp3_dup8_vreg = _mm512_set1_pd(*(b + 3));
        bp4_dup8_vreg = _mm512_set1_pd(*(b + 4));
        bp5_dup8_vreg = _mm512_set1_pd(*(b + 5));
        bp6_dup8_vreg = _mm512_set1_pd(*(b + 6));
        bp7_dup8_vreg = _mm512_set1_pd(*(b + 7));
        b += 8;

        // 在寄存器中累加 (这里命名: 01...89ab...efgh...mn)
        c00__c70_vreg += a0p__a7p_vreg * bp0_dup8_vreg;
        c80__cf0_vreg += a8p__afp_vreg * bp0_dup8_vreg;
        cg0__cn0_vreg += agp__anp_vreg * bp0_dup8_vreg;

        c01__c71_vreg += a0p__a7p_vreg * bp1_dup8_vreg;
        c81__cf1_vreg += a8p__afp_vreg * bp1_dup8_vreg;
        cg1__cn1_vreg += agp__anp_vreg * bp1_dup8_vreg;

        c02__c72_vreg += a0p__a7p_vreg * bp2_dup8_vreg;
        c82__cf2_vreg += a8p__afp_vreg * bp2_dup8_vreg;
        cg2__cn2_vreg += agp__anp_vreg * bp2_dup8_vreg;

        c03__c73_vreg += a0p__a7p_vreg * bp3_dup8_vreg;
        c83__cf3_vreg += a8p__afp_vreg * bp3_dup8_vreg;
        cg3__cn3_vreg += agp__anp_vreg * bp3_dup8_vreg;
        
        c04__c74_vreg += a0p__a7p_vreg * bp4_dup8_vreg;
        c84__cf4_vreg += a8p__afp_vreg * bp4_dup8_vreg;
        cg4__cn4_vreg += agp__anp_vreg * bp4_dup8_vreg;
        
        c05__c75_vreg += a0p__a7p_vreg * bp5_dup8_vreg;
        c85__cf5_vreg += a8p__afp_vreg * bp5_dup8_vreg;
        cg5__cn5_vreg += agp__anp_vreg * bp5_dup8_vreg;
        
        c06__c76_vreg += a0p__a7p_vreg * bp6_dup8_vreg;
        c86__cf6_vreg += a8p__afp_vreg * bp6_dup8_vreg;
        cg6__cn6_vreg += agp__anp_vreg * bp6_dup8_vreg;
        
        c07__c77_vreg += a0p__a7p_vreg * bp7_dup8_vreg;
        c87__cf7_vreg += a8p__afp_vreg * bp7_dup8_vreg;
        cg7__cn7_vreg += agp__anp_vreg * bp7_dup8_vreg;

    }

    // 写回内存
    double delta_C[192] __attribute__((aligned(64)));
    _mm512_store_pd(delta_C + 0, c00__c70_vreg);
    _mm512_store_pd(delta_C + 8, c80__cf0_vreg);
    _mm512_store_pd(delta_C + 16, cg0__cn0_vreg);

    _mm512_store_pd(delta_C + 24, c01__c71_vreg);
    _mm512_store_pd(delta_C + 32, c81__cf1_vreg);
    _mm512_store_pd(delta_C + 40, cg1__cn1_vreg);
    
    _mm512_store_pd(delta_C + 48, c02__c72_vreg);
    _mm512_store_pd(delta_C + 56, c82__cf2_vreg);
    _mm512_store_pd(delta_C + 64, cg2__cn2_vreg);
    
    _mm512_store_pd(delta_C + 72, c03__c73_vreg);
    _mm512_store_pd(delta_C + 80, c83__cf3_vreg);
    _mm512_store_pd(delta_C + 88, cg3__cn3_vreg);
    
    _mm512_store_pd(delta_C + 96, c04__c74_vreg);
    _mm512_store_pd(delta_C + 104, c84__cf4_vreg);
    _mm512_store_pd(delta_C + 112, cg4__cn4_vreg);
    
    _mm512_store_pd(delta_C + 120, c05__c75_vreg);
    _mm512_store_pd(delta_C + 128, c85__cf5_vreg);
    _mm512_store_pd(delta_C + 136, cg5__cn5_vreg);
    
    _mm512_store_pd(delta_C + 144, c06__c76_vreg);
    _mm512_store_pd(delta_C + 152, c86__cf6_vreg);
    _mm512_store_pd(delta_C + 160, cg6__cn6_vreg);
    
    _mm512_store_pd(delta_C + 168, c07__c77_vreg);
    _mm512_store_pd(delta_C + 176, c87__cf7_vreg);
    _mm512_store_pd(delta_C + 184, cg7__cn7_vreg);

    for (int j = 0; j < 8; j++) {
        C(0, j) += delta_C[j * 24 + 0];
        C(1, j) += delta_C[j * 24 + 1];
        C(2, j) += delta_C[j * 24 + 2];
        C(3, j) += delta_C[j * 24 + 3];
        C(4, j) += delta_C[j * 24 + 4];
        C(5, j) += delta_C[j * 24 + 5];
        C(6, j) += delta_C[j * 24 + 6];
        C(7, j) += delta_C[j * 24 + 7];
        C(8, j) += delta_C[j * 24 + 8];
        C(9, j) += delta_C[j * 24 + 9];
        C(10, j) += delta_C[j * 24 + 10];
        C(11, j) += delta_C[j * 24 + 11];
        C(12, j) += delta_C[j * 24 + 12];
        C(13, j) += delta_C[j * 24 + 13];
        C(14, j) += delta_C[j * 24 + 14];
        C(15, j) += delta_C[j * 24 + 15];
        C(16, j) += delta_C[j * 24 + 16];
        C(17, j) += delta_C[j * 24 + 17];
        C(18, j) += delta_C[j * 24 + 18];
        C(19, j) += delta_C[j * 24 + 19];
        C(20, j) += delta_C[j * 24 + 20];
        C(21, j) += delta_C[j * 24 + 21];
        C(22, j) += delta_C[j * 24 + 22];
        C(23, j) += delta_C[j * 24 + 23];
    }

}
