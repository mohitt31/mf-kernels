// diff_bp.cpp - the round-off correctness check against MFEM.
//
// Reads the data mfem_bp --export dumped (B, dB, element restriction map, per-quad
// D, and the reference pair x_in / y_out = A_mfem x), then applies the mf-kernels
// full operator v = G^T B^T D B G x using that SAME data, and reports the maximum
// absolute and relative difference ||v - y_out|| per p. Because the basis, the
// quadrature data, and the gather/scatter map are all MFEM's own, any difference
// is pure floating-point reassociation and must sit at ~1e-13 or below.
//
// Run after mfem_bp --export. Usage: diff_bp <export_dir> <bp1|bp3> [variant]
//   variant in {naive,avx2,blocked,evenodd} (default avx2). All should pass.
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

static std::vector<double> read_vec(const std::string& path, int& a, int& b, int& c){
  std::ifstream f(path); if(!f){ fprintf(stderr,"missing %s\n",path.c_str()); exit(2);}
  std::string line; std::getline(f,line); std::istringstream h(line);
  a=b=c=-1; h>>a; if(!(h>>b)) b=-1; if(!(h>>c)) c=-1;
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

  double worst_abs=0, worst_rel=0; bool ok=true;
  for(int p=1;p<=8;++p){
    int nd=p+1,nq=p+2,nd3=nd*nd*nd,nq3=nq*nq*nq;
    int ra,rb,rc;
    auto B = read_vec(dir+"/B_"+std::to_string(p)+".txt",ra,rb,rc);  // nq x nd
    auto dB= read_vec(dir+"/G_"+std::to_string(p)+".txt",ra,rb,rc);
    // emap header: NE nd3 vsize
    int NE,c2,vsize; auto emapd=read_vec(dir+"/emap_"+std::to_string(p)+".txt",NE,c2,vsize);
    std::vector<long> emap(emapd.size()); for(size_t i=0;i<emapd.size();++i) emap[i]=(long)std::llround(emapd[i]);
    int ne2,nq2,ncomp; auto qd=read_vec(dir+"/qd_"+std::to_string(p)+".txt",ne2,nq2,ncomp);
    int xs; auto x=read_vec(dir+"/x_"+std::to_string(p)+".txt",xs,rb,rc);
    int ys; auto yref=read_vec(dir+"/y_"+std::to_string(p)+".txt",ys,rb,rc);

    std::vector<double> Bt(nd*nq),dBt(nd*nq);
    for(int i=0;i<nd;++i)for(int q=0;q<nq;++q){Bt[i*nq+q]=B[q*nd+i];dBt[i*nq+q]=dB[q*nd+i];}

    std::vector<double> v(vsize,0.0);
    auto ue=make_aligned<double>((size_t)nd3*lanes), ve=make_aligned<double>((size_t)nd3*lanes);
    auto t1=make_aligned<double>((size_t)nq3*lanes),t2=make_aligned<double>((size_t)nq3*lanes),Q=make_aligned<double>((size_t)nq3*lanes);
    auto Gx=make_aligned<double>((size_t)nq3*lanes),Gy=make_aligned<double>((size_t)nq3*lanes),Gz=make_aligned<double>((size_t)nq3*lanes),acc=make_aligned<double>((size_t)nd3*lanes);
    auto qd1l=make_aligned<double>((size_t)nq3*lanes), qd6l=make_aligned<double>((size_t)nq3*6*lanes);

    int nbatch=(NE+lanes-1)/lanes;
    for(int b=0;b<nbatch;++b){
      int e0=b*lanes, act=std::min(lanes,NE-e0);
      // gather
      for(int j=0;j<nd3;++j)for(int l=0;l<lanes;++l){
        double val=0.0; if(l<act) val=x[emap[(size_t)(e0+l)*nd3+j]];
        ue[(size_t)j*lanes+l]=val;
      }
      // per-lane quad data assembled from this batch's elements
      if(bp1){
        for(int q=0;q<nq3;++q)for(int l=0;l<lanes;++l){
          double d=0.0; if(l<act) d=qd[((size_t)(e0+l)*nq3)+q];
          qd1l[(size_t)q*lanes+l]=d;
        }
        bp1_elem(Kv,B.data(),Bt.data(),ue.get(),ve.get(),qd1l.get(),t1.get(),t2.get(),Q.get(),nq,nd);
      } else {
        for(int q=0;q<nq3;++q)for(int comp=0;comp<6;++comp)for(int l=0;l<lanes;++l){
          double d=0.0; if(l<act) d=qd[(((size_t)(e0+l)*nq3)+q)*6+comp];
          qd6l[((size_t)q*6+comp)*lanes+l]=d;
        }
        bp3_elem(Kv,Kd,B.data(),dB.data(),Bt.data(),dBt.data(),ue.get(),ve.get(),
                 qd6l.get(),t1.get(),t2.get(),Gx.get(),Gy.get(),Gz.get(),acc.get(),nq,nd);
      }
      // scatter-add
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
  }
  printf("\nvariant=%s  worst abs=%.3e  worst rel=%.3e  -> %s (threshold 1e-13)\n",
         vsel.c_str(),worst_abs,worst_rel, ok?"PASS":"FAIL");
  return ok?0:1;
}
