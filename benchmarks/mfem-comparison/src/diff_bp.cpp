// diff_bp.cpp - the round-off correctness check against MFEM.
//
// Design (revised): the basis B, derivative dB, and per-quad data D are built HERE
// from the validated numerics.hpp using the SAME discretization MFEM uses (order-p
// Gauss-Lobatto nodes, n_q = p+2 Gauss-Legendre quadrature, structured Cartesian
// mesh of side h = 1/n). The physical element operator is independent of the
// reference-frame parametrization, so this reproduces MFEM's element operator to
// round-off. From MFEM we import ONLY the element restriction map (emap), the mesh
// size n (meta.txt), and the reference pair x_in / y_out = A_mfem x. We then apply
// v = G^T B^T D B G x using MFEM's own emap and compare v to y_out.
//
// This deliberately does NOT read MFEM's exported B/dB/qdata: their internal memory
// layout is an easy place to mismatch, and importing them is unnecessary because
// numerics.hpp already agrees with the dense reference to ~1e-14 (see validate_bp).
//
// Usage: diff_bp <export_dir> <bp1|bp3> [variant]   variant in
//   {naive,avx2,blocked,evenodd} (default avx2). All should pass at <= 1e-13.
#include "numerics.hpp"
#include "apply_1d.hpp"
#include "evenodd_rect.hpp"
#include "bp_operator.hpp"
#include "aligned_buffer.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
using namespace mfk;

static void K_naive  (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_naive(S,u,v,q,d,s);}
static void K_avx2   (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_avx2(S,u,v,q,d,s);}
static void K_blocked(const double*S,const double*u,double*v,int q,int d,int s){apply_1d_avx2_blocked(S,u,v,q,d,s);}
static void K_eo_val (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_evenodd_rect(S,u,v,q,d,s);}
static void K_eo_der (const double*S,const double*u,double*v,int q,int d,int s){apply_1d_evenodd_rect_deriv(S,u,v,q,d,s);}

static std::vector<double> read_vec(const std::string& path, std::vector<long>* hdr=nullptr){
  std::ifstream f(path); if(!f){ fprintf(stderr,"missing %s\n",path.c_str()); exit(2);}
  std::string line; std::getline(f,line); std::istringstream h(line);
  if(hdr){ hdr->clear(); long t; while(h>>t) hdr->push_back(t); }
  std::vector<double> v; double x; while(f>>x) v.push_back(x); return v;
}

int main(int argc,char**argv){
  if(argc<3){ printf("usage: diff_bp <dir> <bp1|bp3> [variant]\n"); return 1; }
  std::string dir=argv[1]; bool bp1=!strcmp(argv[2],"bp1");
  std::string vsel=argc>3?argv[3]:"avx2";
  Apply1DFn Kv=K_avx2, Kd=K_avx2;
  if(vsel=="naive"){Kv=Kd=K_naive;}
  else if(vsel=="blocked"){Kv=Kd=K_blocked;}
  else if(vsel=="evenodd"){Kv=K_eo_val;Kd=K_eo_der;}

  std::vector<long> mh; auto mv=read_vec(dir+"/meta.txt",&mh);
  int n = mh.empty()? (int)(mv.empty()?3:mv[0]) : (int)mh[0];
  double h=1.0/n;

  double worst_abs=0, worst_rel=0; bool ok=true;
  for(int p=1;p<=8;++p){
    int nd=p+1,nq=p+2,nd3=nd*nd*nd,nq3=nq*nq*nq;
    std::vector<double> B,dB,qx,qw,nodes; num::build_B(p,B,dB,qx,qw,nodes);
    std::vector<double> Bt(nd*nq),dBt(nd*nq);
    for(int i=0;i<nd;++i)for(int q=0;q<nq;++q){Bt[i*nq+q]=B[q*nd+i];dBt[i*nq+q]=dB[q*nd+i];}
    auto qd1l=make_aligned<double>((size_t)nq3*lanes), qd6l=make_aligned<double>((size_t)nq3*6*lanes);
    double detJ=std::pow(h*0.5,3), mscale=h*0.5;
    for(int q1=0;q1<nq;++q1)for(int q2=0;q2<nq;++q2)for(int q3=0;q3<nq;++q3){
      int Q=(q1*nq+q2)*nq+q3; double w=qw[q1]*qw[q2]*qw[q3];
      for(int l=0;l<lanes;++l){
        qd1l[(size_t)Q*lanes+l]=w*detJ;
        double m=w*mscale; double* d=&qd6l[((size_t)Q*6)*lanes];
        d[0*lanes+l]=m;d[1*lanes+l]=0;d[2*lanes+l]=0;d[3*lanes+l]=m;d[4*lanes+l]=0;d[5*lanes+l]=m;
      }
    }
    std::vector<long> eh; auto emapd=read_vec(dir+"/emap_"+std::to_string(p)+".txt",&eh);
    int NE=(int)eh[0], vsize=(int)eh[2];
    std::vector<long> emap(emapd.size()); for(size_t i=0;i<emapd.size();++i) emap[i]=(long)std::llround(emapd[i]);
    auto x=read_vec(dir+"/x_"+std::to_string(p)+".txt");
    auto yref=read_vec(dir+"/y_"+std::to_string(p)+".txt");

    std::vector<double> v(vsize,0.0);
    auto ue=make_aligned<double>((size_t)nd3*lanes), ve=make_aligned<double>((size_t)nd3*lanes);
    auto t1=make_aligned<double>((size_t)nq3*lanes),t2=make_aligned<double>((size_t)nq3*lanes),Q=make_aligned<double>((size_t)nq3*lanes);
    auto Gx=make_aligned<double>((size_t)nq3*lanes),Gy=make_aligned<double>((size_t)nq3*lanes),Gz=make_aligned<double>((size_t)nq3*lanes),acc=make_aligned<double>((size_t)nd3*lanes);

    int nbatch=(NE+lanes-1)/lanes;
    for(int b=0;b<nbatch;++b){
      int e0=b*lanes, act=std::min(lanes,NE-e0);
      for(int j=0;j<nd3;++j)for(int l=0;l<lanes;++l){
        double val=0.0; if(l<act) val=x[emap[(size_t)(e0+l)*nd3+j]];
        ue[(size_t)j*lanes+l]=val;
      }
      if(bp1)
        bp1_elem(Kv,B.data(),Bt.data(),ue.get(),ve.get(),qd1l.get(),t1.get(),t2.get(),Q.get(),nq,nd);
      else
        bp3_elem(Kv,Kd,B.data(),dB.data(),Bt.data(),dBt.data(),ue.get(),ve.get(),
                 qd6l.get(),t1.get(),t2.get(),Gx.get(),Gy.get(),Gz.get(),acc.get(),nq,nd);
      for(int l=0;l<act;++l)for(int j=0;j<nd3;++j)
        v[emap[(size_t)(e0+l)*nd3+j]]+=ve[(size_t)j*lanes+l];
    }
    double mabs=0,nrm=0;
    for(int i=0;i<vsize;++i){ mabs=std::max(mabs,std::abs(v[i]-yref[i])); nrm=std::max(nrm,std::abs(yref[i])); }
    double mrel=mabs/(nrm>0?nrm:1.0);
    worst_abs=std::max(worst_abs,mabs); worst_rel=std::max(worst_rel,mrel);
    bool pass=mabs<1e-13 || mrel<1e-13;
    if(!pass) ok=false;
    printf("p=%d NE=%d vsize=%d  maxabs=%.3e  maxrel=%.3e  %s\n",
           p,NE,vsize,mabs,mrel,pass?"ok":"FAIL");
    if(p==1 && !pass){
      printf("   [debug p=1] first entries (v, yref, diff):\n");
      for(int i=0;i<std::min(vsize,8);++i)
        printf("     [%d] %.6e  %.6e  %.2e\n",i,v[i],yref[i],v[i]-yref[i]);
    }
  }
  printf("\nvariant=%s  worst abs=%.3e  worst rel=%.3e  -> %s (threshold 1e-13)\n",
         vsel.c_str(),worst_abs,worst_rel, ok?"PASS":"FAIL");
  return ok?0:1;
}
