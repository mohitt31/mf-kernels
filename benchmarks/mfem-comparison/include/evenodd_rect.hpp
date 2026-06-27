// evenodd_rect.hpp - even-odd (symmetry) sum-factorization for GENERAL
// rectangular B (n_q x n_d), n_q != n_d, either dimension odd. Required for the
// CEED BP convention (n_q = p+2, n_d = p+1). Halves the contraction FLOPs.
//
// Values B  : B[n_q-1-q][n_d-1-i] =  B[q][i]   (symmetric point sets)
// Derivs dB : dB[n_q-1-q][n_d-1-i] = -dB[q][i] (antisymmetric)
#pragma once
#include "apply_1d.hpp"
namespace mfk {
#if defined(__AVX2__)
// VALUES. Scalar S (n_q x n_d). Batched over `lanes` cells.
MFK_ALWAYS_INLINE
void apply_1d_evenodd_rect(const double* MFK_RESTRICT S,
                           const double* MFK_RESTRICT u,
                           double*       MFK_RESTRICT v,
                           int n_q, int n_d, int n_spec){
  const int hq=n_q/2, hd=n_d/2; const bool oddd=(n_d&1), oddq=(n_q&1); const int md=hd;
  for(int k=0;k<n_spec;++k){
    const double* uk=u+(size_t)k*n_d*lanes; double* vk=v+(size_t)k*n_q*lanes;
    __m256d E[32],O[32]; // hd <= 32 for p<=8 (n_d<=9)
    for(int i=0;i<hd;++i){
      __m256d a=_mm256_load_pd(uk+i*lanes), b=_mm256_load_pd(uk+(n_d-1-i)*lanes);
      E[i]=_mm256_add_pd(a,b); O[i]=_mm256_sub_pd(a,b);
    }
    __m256d umid = oddd? _mm256_load_pd(uk+md*lanes) : _mm256_setzero_pd();
    for(int q=0;q<hq;++q){
      __m256d P=_mm256_setzero_pd(), M=_mm256_setzero_pd();
      const double* Sq=S+q*n_d;
      for(int i=0;i<hd;++i){
        double sp=Sq[i]+Sq[n_d-1-i], sm=Sq[i]-Sq[n_d-1-i];
        P=_mm256_fmadd_pd(_mm256_set1_pd(sp),E[i],P);
        M=_mm256_fmadd_pd(_mm256_set1_pd(sm),O[i],M);
      }
      if(oddd) P=_mm256_fmadd_pd(_mm256_set1_pd(2.0*Sq[md]),umid,P); // mid node into P
      __m256d h=_mm256_set1_pd(0.5);
      _mm256_store_pd(vk+q*lanes,            _mm256_mul_pd(h,_mm256_add_pd(P,M)));
      _mm256_store_pd(vk+(n_q-1-q)*lanes,    _mm256_mul_pd(h,_mm256_sub_pd(P,M)));
    }
    if(oddq){ // middle quad row = sum S[qmid][i] E[i] + S[qmid][md] u[md]
      const double* Sq=S+hq*n_d; __m256d acc=_mm256_setzero_pd();
      for(int i=0;i<hd;++i) acc=_mm256_fmadd_pd(_mm256_set1_pd(Sq[i]),E[i],acc);
      if(oddd) acc=_mm256_fmadd_pd(_mm256_set1_pd(Sq[md]),umid,acc);
      _mm256_store_pd(vk+hq*lanes,acc);
    }
  }
}
// DERIVATIVES. Antisymmetric: channels swap (P uses O, M uses E).
MFK_ALWAYS_INLINE
void apply_1d_evenodd_rect_deriv(const double* MFK_RESTRICT S,
                                 const double* MFK_RESTRICT u,
                                 double*       MFK_RESTRICT v,
                                 int n_q, int n_d, int n_spec){
  const int hq=n_q/2, hd=n_d/2; const bool oddd=(n_d&1), oddq=(n_q&1); const int md=hd;
  for(int k=0;k<n_spec;++k){
    const double* uk=u+(size_t)k*n_d*lanes; double* vk=v+(size_t)k*n_q*lanes;
    __m256d E[32],O[32];
    for(int i=0;i<hd;++i){
      __m256d a=_mm256_load_pd(uk+i*lanes), b=_mm256_load_pd(uk+(n_d-1-i)*lanes);
      E[i]=_mm256_add_pd(a,b); O[i]=_mm256_sub_pd(a,b);
    }
    __m256d umid = oddd? _mm256_load_pd(uk+md*lanes) : _mm256_setzero_pd();
    for(int q=0;q<hq;++q){
      __m256d P=_mm256_setzero_pd(), M=_mm256_setzero_pd();
      const double* Sq=S+q*n_d;
      for(int i=0;i<hd;++i){
        double sm=Sq[i]-Sq[n_d-1-i]; // P-channel coeff (antisym)
        double sp=Sq[i]+Sq[n_d-1-i]; // M-channel coeff
        P=_mm256_fmadd_pd(_mm256_set1_pd(sm),O[i],P);
        M=_mm256_fmadd_pd(_mm256_set1_pd(sp),E[i],M);
      }
      // mid node: dB[q][md]; antisym => dB[qmid_mirror]... contributes (dB[q][md]) to v[q]
      if(oddd) { M=_mm256_fmadd_pd(_mm256_set1_pd(2.0*Sq[md]),umid,M); } // antisym: mid node -> M channel
      __m256d h=_mm256_set1_pd(0.5);
      _mm256_store_pd(vk+q*lanes,            _mm256_mul_pd(h,_mm256_add_pd(P,M)));
      _mm256_store_pd(vk+(n_q-1-q)*lanes,    _mm256_mul_pd(h,_mm256_sub_pd(P,M)));
    }
    if(oddq){ // middle row of derivative is antisymmetric => uses O
      const double* Sq=S+hq*n_d; __m256d acc=_mm256_setzero_pd();
      for(int i=0;i<hd;++i) acc=_mm256_fmadd_pd(_mm256_set1_pd(Sq[i]),O[i],acc);
      // mid node deriv at mid row is 0 by antisymmetry; skip
      _mm256_store_pd(vk+hq*lanes,acc);
    }
  }
}
#endif
} // namespace mfk
