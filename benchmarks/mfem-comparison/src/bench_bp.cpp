// bench_bp.cpp - mf-kernels side of the MFEM comparison.
//
// Benchmarks the FULL CEED operator apply  v = G^T B^T D B G u  for BP1 (mass)
// and BP3 (Poisson), p = 1..8, over a structured n x n x n hexahedral mesh of
// order-p continuous Q_p elements, for each kernel variant
// (naive / avx2 / avx2_blocked / even-odd).
//
// Why an internal mesh: it makes G (element gather/scatter) well-defined and the
// operator a true global apply that ingests/produces an L-vector, so the
// standalone path here is a faithful dry-run of exactly what mfem_bp times on the
// benchmark machine, and the same harness consumes MFEM-exported data with
// --mfem-data. The Cartesian geometry gives constant detJ / diagonal metric,
// matching validate_bp.cpp's reference.
//
// Reports, per (operator, p, variant): unique global DOFs, elements, median
// time/apply, IQR, unique-DOFs/sec, and GFLOP/s using the shared standard FLOP
// count. CSV columns are stable for plots.py.
//
// Build: see scripts/build.sh. Run pinned: taskset -c 0 ./bench_bp ... (run.sh).
#include "numerics.hpp"
#include "apply_1d.hpp"
#include "evenodd_rect.hpp"
#include "bp_operator.hpp"
#include "aligned_buffer.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
using namespace mfk;

static void K_naive  (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_naive(S,u,v,q,d,s);}
static void K_avx2   (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_avx2(S,u,v,q,d,s);}
static void K_blocked(const double*S,const double*u,double*v,int q,int d,int s){apply_1d_avx2_blocked(S,u,v,q,d,s);}
static void K_eo_val (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_evenodd_rect(S,u,v,q,d,s);}
static void K_eo_der (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_evenodd_rect_deriv(S,u,v,q,d,s);}

static inline double now_s(){
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Variant { const char* name; Apply1DFn Kv,Kd; };

// Structured Q_p mesh on [0,1]^3: n elements per dim, side h = 1/n. Global node
// grid is (n*p+1)^3, lexicographic with x fastest. Element-local node order is
// i1 fastest (matches bp_operator.hpp).
struct Mesh {
  int n, p, nd, nq, np1;
  long nelem, ndofs;        // elements, unique global dofs
  double h;
  std::vector<int> emap;    // [elem][nd^3] global dof index (i1 fastest)
  void build(int n_, int p_){
    n=n_; p=p_; nd=p+1; nq=p+2; np1=n*p+1; h=1.0/n;
    nelem=(long)n*n*n; ndofs=(long)np1*np1*np1;
    emap.resize((size_t)nelem*nd*nd*nd);
    long e=0;
    for(int ez=0; ez<n; ++ez)for(int ey=0; ey<n; ++ey)for(int ex=0; ex<n; ++ex,++e){
      int* m=&emap[(size_t)e*nd*nd*nd];
      for(int i3=0;i3<nd;++i3)for(int i2=0;i2<nd;++i2)for(int i1=0;i1<nd;++i1){
        int gx=ex*p+i1, gy=ey*p+i2, gz=ez*p+i3;
        m[(i3*nd+i2)*nd+i1]=(gz*np1+gy)*np1+gx;
      }
    }
  }
};

// choose n so unique dofs ~ target, at least 2 elements/dim.
static int pick_n(int p, long target){
  int n=(int)std::lround((std::cbrt((double)target)-1.0)/p);
  return std::max(2,n);
}

int main(int argc, char** argv){
  long target=2000000;       // ~2M unique dofs per case
  int reps=21;               // odd -> clean median
  const char* op_sel="both"; // bp1 | bp3 | both
  const char* csv=nullptr;
  for(int a=1;a<argc;++a){
    if(!strcmp(argv[a],"--target")&&a+1<argc) target=atol(argv[++a]);
    else if(!strcmp(argv[a],"--reps")&&a+1<argc) reps=atoi(argv[++a]);
    else if(!strcmp(argv[a],"--op")&&a+1<argc) op_sel=argv[++a];
    else if(!strcmp(argv[a],"--csv")&&a+1<argc) csv=argv[++a];
  }
  std::vector<Variant> variants={
    {"naive",   K_naive,  K_naive},
    {"avx2",    K_avx2,   K_avx2},
    {"blocked", K_blocked,K_blocked},
    {"evenodd", K_eo_val, K_eo_der},
  };
  FILE* f = csv ? fopen(csv,"w") : nullptr;
  if(f) fprintf(f,"operator,p,variant,elements,unique_dofs,evec_dofs,"
                  "median_s,iqr_s,dofs_per_s,gflops\n");
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> dist(-1,1);

  printf("%-4s %-3s %-8s %-9s %-9s %-11s %-10s %-9s\n",
         "op","p","variant","elems","uniqueDOF","median_s","GDOF/s","GFLOP/s");
  for(int which=0; which<2; ++which){
    bool do_bp1=(which==0), is_bp1=do_bp1;
    const char* opname=is_bp1?"BP1":"BP3";
    if(is_bp1 && !strcmp(op_sel,"bp3")) continue;
    if(!is_bp1 && !strcmp(op_sel,"bp1")) continue;
    for(int p=1;p<=8;++p){
      int nd=p+1,nq=p+2,nd3=nd*nd*nd,nq3=nq*nq*nq;
      Mesh mesh; mesh.build(pick_n(p,target),p);
      long nelem=mesh.nelem;
      int nbatch=(int)((nelem+lanes-1)/lanes);
      // basis
      std::vector<double> B,dB,qx,qw,nodes; num::build_B(p,B,dB,qx,qw,nodes);
      std::vector<double> Bt(nd*nq),dBt(nd*nq);
      for(int i=0;i<nd;++i)for(int q=0;q<nq;++q){Bt[i*nq+q]=B[q*nd+i];dBt[i*nq+q]=dB[q*nd+i];}
      // quad data (Cartesian, constant per element). detJ=(h/2)^3, diag metric w*(h/2).
      std::vector<double> qd1(nq3*lanes), qd6(nq3*6*lanes);
      double detJ=std::pow(mesh.h*0.5,3), mscale=mesh.h*0.5;
      for(int q1=0;q1<nq;++q1)for(int q2=0;q2<nq;++q2)for(int q3=0;q3<nq;++q3){
        int Q=(q1*nq+q2)*nq+q3; double w=qw[q1]*qw[q2]*qw[q3];
        for(int l=0;l<lanes;++l){
          qd1[(size_t)Q*lanes+l]=w*detJ;
          double m=w*mscale; double* d=&qd6[((size_t)Q*6)*lanes];
          d[0*lanes+l]=m;d[1*lanes+l]=0;d[2*lanes+l]=0;d[3*lanes+l]=m;d[4*lanes+l]=0;d[5*lanes+l]=m;
        }
      }
      // global vectors + element scratch
      std::vector<double> ug(mesh.ndofs), vg(mesh.ndofs);
      for(auto& x:ug) x=dist(rng);
      auto ue=make_aligned<double>((size_t)nd3*lanes);
      auto ve=make_aligned<double>((size_t)nd3*lanes);
      auto t1=make_aligned<double>((size_t)nq3*lanes),t2=make_aligned<double>((size_t)nq3*lanes);
      auto Q =make_aligned<double>((size_t)nq3*lanes);
      auto Gx=make_aligned<double>((size_t)nq3*lanes),Gy=make_aligned<double>((size_t)nq3*lanes),Gz=make_aligned<double>((size_t)nq3*lanes);
      auto acc=make_aligned<double>((size_t)nd3*lanes);

      // full operator apply: G (gather) -> element op -> G^T (scatter-add)
      auto apply=[&](const Variant& V){
        std::fill(vg.begin(),vg.end(),0.0);
        for(int b=0;b<nbatch;++b){
          long e0=(long)b*lanes;
          int act=(int)std::min((long)lanes,nelem-e0);
          // gather
          for(int j=0;j<nd3;++j)for(int l=0;l<lanes;++l){
            long e=e0+l; double val=0.0;
            if(l<act) val=ug[mesh.emap[(size_t)e*nd3+j]];
            ue[(size_t)j*lanes+l]=val;
          }
          // element apply
          if(is_bp1)
            bp1_elem(V.Kv,B.data(),Bt.data(),ue.get(),ve.get(),qd1.data(),
                     t1.get(),t2.get(),Q.get(),nq,nd);
          else
            bp3_elem(V.Kv,V.Kd,B.data(),dB.data(),Bt.data(),dBt.data(),
                     ue.get(),ve.get(),qd6.data(),t1.get(),t2.get(),
                     Gx.get(),Gy.get(),Gz.get(),acc.get(),nq,nd);
          // scatter-add
          for(int l=0;l<act;++l){ long e=e0+l;
            for(int j=0;j<nd3;++j) vg[mesh.emap[(size_t)e*nd3+j]]+=ve[(size_t)j*lanes+l];
          }
        }
      };
      double fl_cell = is_bp1? bp1_flops_per_cell(nq,nd) : bp3_flops_per_cell(nq,nd);
      double flops = fl_cell*(double)nelem;
      for(size_t vi=0; vi<variants.size(); ++vi){
        apply(variants[vi]); // warmup
        std::vector<double> ts; ts.reserve(reps);
        for(int r=0;r<reps;++r){ double a=now_s(); apply(variants[vi]); ts.push_back(now_s()-a); }
        std::sort(ts.begin(),ts.end());
        double med=ts[ts.size()/2];
        double q1=ts[ts.size()/4], q3=ts[(3*ts.size())/4]; double iqr=q3-q1;
        double dps=mesh.ndofs/med, gflops=flops/med*1e-9;
        printf("%-4s %-3d %-8s %-9ld %-9ld %-11.3e %-10.3f %-9.2f\n",
               opname,p,variants[vi].name,nelem,mesh.ndofs,med,dps*1e-9,gflops);
        if(f) fprintf(f,"%s,%d,%s,%ld,%ld,%ld,%.6e,%.6e,%.6e,%.4f\n",
               opname,p,variants[vi].name,nelem,mesh.ndofs,(long)nelem*nd3,
               med,iqr,dps,gflops);
      }
    }
  }
  if(f) fclose(f);
  printf("\n(mf-kernels full operator G^T B^T D B G, single core. "
         "FLOP/s uses shared standard sum-fact count.)\n");
  return 0;
}
