// validate_bp.cpp - MFEM-independent correctness check.
//
// Builds the GLL/GL basis from numerics.hpp, then verifies the sum-factorized
// BP1 (mass) and BP3 (Poisson) element operators in bp_operator.hpp against a
// dense O(p^6) reference, for every kernel variant and p = 1..8. This proves the
// operator is mathematically correct independent of MFEM; the MFEM round-off
// check (1e-13 on the global vector) is a separate step in diff_bp.cpp that runs
// only on the benchmark machine where MFEM is available.
//
// Variants checked: naive, avx2, avx2_blocked (standard generic kernels, Kv=Kd),
// and even-odd (Kv = apply_1d_evenodd_rect, Kd = apply_1d_evenodd_rect_deriv).
#include "numerics.hpp"
#include "apply_1d.hpp"
#include "evenodd_rect.hpp"
#include "bp_operator.hpp"
#include "aligned_buffer.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>
#include <string>
using namespace mfk;

static void K_naive  (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_naive(S,u,v,q,d,s);}
static void K_avx2   (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_avx2(S,u,v,q,d,s);}
static void K_blocked(const double*S,const double*u,double*v,int q,int d,int s){apply_1d_avx2_blocked(S,u,v,q,d,s);}
static void K_eo_val (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_evenodd_rect(S,u,v,q,d,s);}
static void K_eo_der (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_evenodd_rect_deriv(S,u,v,q,d,s);}

// dense BP1 single-lane reference. node J=i1+nd*i2+nd^2*i3, quad Q=q3+nq*q2+nq^2*q1.
static std::vector<double> dense_bp1(const std::vector<double>&u,const std::vector<double>&B,
    const std::vector<double>&qw,double detJ,int nq,int nd){
  int nd3=nd*nd*nd,nq3=nq*nq*nq; std::vector<double> q(nq3,0.0),v(nd3,0.0);
  auto B3=[&](int q1,int q2,int q3,int i1,int i2,int i3){return B[q1*nd+i1]*B[q2*nd+i2]*B[q3*nd+i3];};
  for(int q1=0;q1<nq;++q1)for(int q2=0;q2<nq;++q2)for(int q3=0;q3<nq;++q3){
    int Q=(q1*nq+q2)*nq+q3; double s=0;
    for(int i3=0;i3<nd;++i3)for(int i2=0;i2<nd;++i2)for(int i1=0;i1<nd;++i1)
      s+=B3(q1,q2,q3,i1,i2,i3)*u[(i3*nd+i2)*nd+i1];
    q[Q]=s*qw[q1]*qw[q2]*qw[q3]*detJ;
  }
  for(int q1=0;q1<nq;++q1)for(int q2=0;q2<nq;++q2)for(int q3=0;q3<nq;++q3){
    int Q=(q1*nq+q2)*nq+q3;
    for(int i3=0;i3<nd;++i3)for(int i2=0;i2<nd;++i2)for(int i1=0;i1<nd;++i1)
      v[(i3*nd+i2)*nd+i1]+=B3(q1,q2,q3,i1,i2,i3)*q[Q];
  }
  return v;
}
// dense BP3 single-lane reference, Cartesian element side h (diagonal metric w*(h/2)).
static std::vector<double> dense_bp3(const std::vector<double>&u,const std::vector<double>&B,
    const std::vector<double>&dB,const std::vector<double>&qw,double h,int nq,int nd){
  int nd3=nd*nd*nd,nq3=nq*nq*nq; std::vector<double> gx(nq3),gy(nq3),gz(nq3),v(nd3,0.0);
  auto val=[&](const std::vector<double>&M,int q,int i){return M[q*nd+i];};
  for(int q1=0;q1<nq;++q1)for(int q2=0;q2<nq;++q2)for(int q3=0;q3<nq;++q3){
    int Q=(q1*nq+q2)*nq+q3; double ax=0,ay=0,az=0;
    for(int i3=0;i3<nd;++i3)for(int i2=0;i2<nd;++i2)for(int i1=0;i1<nd;++i1){
      double uu=u[(i3*nd+i2)*nd+i1];
      ax+=val(dB,q1,i1)*val(B,q2,i2)*val(B,q3,i3)*uu;
      ay+=val(B,q1,i1)*val(dB,q2,i2)*val(B,q3,i3)*uu;
      az+=val(B,q1,i1)*val(B,q2,i2)*val(dB,q3,i3)*uu;
    }
    double m=qw[q1]*qw[q2]*qw[q3]*(h*0.5);
    gx[Q]=ax*m; gy[Q]=ay*m; gz[Q]=az*m;
  }
  for(int q1=0;q1<nq;++q1)for(int q2=0;q2<nq;++q2)for(int q3=0;q3<nq;++q3){
    int Q=(q1*nq+q2)*nq+q3;
    for(int i3=0;i3<nd;++i3)for(int i2=0;i2<nd;++i2)for(int i1=0;i1<nd;++i1)
      v[(i3*nd+i2)*nd+i1]+= val(dB,q1,i1)*val(B,q2,i2)*val(B,q3,i3)*gx[Q]
                          + val(B,q1,i1)*val(dB,q2,i2)*val(B,q3,i3)*gy[Q]
                          + val(B,q1,i1)*val(B,q2,i2)*val(dB,q3,i3)*gz[Q];
  }
  return v;
}

int main(){
  std::mt19937_64 rng(12345); std::uniform_real_distribution<double> dist(-1,1);
  struct V{const char*name; Apply1DFn Kv,Kd;};
  std::vector<V> variants={
    {"naive",   K_naive,  K_naive},
    {"avx2",    K_avx2,   K_avx2},
    {"blocked", K_blocked,K_blocked},
    {"evenodd", K_eo_val, K_eo_der},
  };
  double worst_overall=0; bool ok=true;
  printf("%-3s %-3s %-3s | %-9s %-9s %-9s %-9s\n","p","nd","nq",
         "naiveBP1","avx2BP1","blkBP1","eoBP1");
  for(int p=1;p<=8;++p){
    int nd=p+1,nq=p+2,nd3=nd*nd*nd;
    std::vector<double> B,dB,qx,qw,nodes; num::build_B(p,B,dB,qx,qw,nodes);
    std::vector<double> Bt(nd*nq),dBt(nd*nq);
    for(int i=0;i<nd;++i)for(int q=0;q<nq;++q){Bt[i*nq+q]=B[q*nd+i];dBt[i*nq+q]=dB[q*nd+i];}
    double h[4]; for(int l=0;l<4;++l) h[l]=0.5+0.3*l;
    auto ue_=make_aligned<double>(nd3*4),ve_=make_aligned<double>(nd3*4);
    double *ue=ue_.get(),*ve=ve_.get();
    std::vector<std::vector<double>> uL(4,std::vector<double>(nd3));
    for(int l=0;l<4;++l)for(int j=0;j<nd3;++j){double x=dist(rng);uL[l][j]=x;ue[j*4+l]=x;}
    int nq3=nq*nq*nq;
    std::vector<double> qd1(nq3*4),qd6(nq3*6*4);
    for(int q1=0;q1<nq;++q1)for(int q2=0;q2<nq;++q2)for(int q3=0;q3<nq;++q3){
      int Q=(q1*nq+q2)*nq+q3; double w=qw[q1]*qw[q2]*qw[q3];
      for(int l=0;l<4;++l){ double detJ=std::pow(h[l]*0.5,3);
        qd1[(size_t)Q*4+l]=w*detJ;
        double m=w*(h[l]*0.5);
        double* d=&qd6[((size_t)Q*6)*4];
        d[0*4+l]=m;d[1*4+l]=0;d[2*4+l]=0;d[3*4+l]=m;d[4*4+l]=0;d[5*4+l]=m;
      }
    }
    auto t1_=make_aligned<double>(nq3*4),t2_=make_aligned<double>(nq3*4),Q_=make_aligned<double>(nq3*4);
    auto Gx_=make_aligned<double>(nq3*4),Gy_=make_aligned<double>(nq3*4),Gz_=make_aligned<double>(nq3*4),acc_=make_aligned<double>(nd3*4);
    double *t1=t1_.get(),*t2=t2_.get(),*Q=Q_.get(),*Gx=Gx_.get(),*Gy=Gy_.get(),*Gz=Gz_.get(),*acc=acc_.get();
    double d1[4]={0,0,0,0}, d3[4]={0,0,0,0};
    for(size_t vi=0; vi<variants.size(); ++vi){
      // BP1
      bp1_elem(variants[vi].Kv,B.data(),Bt.data(),ue,ve,qd1.data(),t1,t2,Q,nq,nd);
      for(int l=0;l<4;++l){auto r=dense_bp1(uL[l],B,qw,std::pow(h[l]*0.5,3),nq,nd);
        for(int j=0;j<nd3;++j) d1[vi]=std::max(d1[vi],std::abs(ve[j*4+l]-r[j]));}
      // BP3
      bp3_elem(variants[vi].Kv,variants[vi].Kd,B.data(),dB.data(),Bt.data(),dBt.data(),
               ue,ve,qd6.data(),t1,t2,Gx,Gy,Gz,acc,nq,nd);
      for(int l=0;l<4;++l){auto r=dense_bp3(uL[l],B,dB,qw,h[l],nq,nd);
        for(int j=0;j<nd3;++j) d3[vi]=std::max(d3[vi],std::abs(ve[j*4+l]-r[j]));}
      worst_overall=std::max(worst_overall,std::max(d1[vi],d3[vi]));
      if(d1[vi]>1e-12||d3[vi]>1e-12) ok=false;
    }
    printf("%-3d %-3d %-3d | %-9.1e %-9.1e %-9.1e %-9.1e\n",p,nd,nq,d1[0],d1[1],d1[2],d1[3]);
    printf("%-3s %-3s %-3s | %-9.1e %-9.1e %-9.1e %-9.1e   (BP3)\n","","","",d3[0],d3[1],d3[2],d3[3]);
  }
  printf("\nworst max-abs-diff over all p / variants / BP1+BP3 = %.2e\n",worst_overall);
  printf("%s\n", ok ? "PASS: sum-fact operator == dense reference to ~1e-13"
                    : "FAIL: a variant exceeded 1e-12");
  return ok?0:1;
}
