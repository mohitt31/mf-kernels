// mfem_bp.cpp - MFEM side of the comparison: native partial assembly (PA) and,
// when MFEM is built with -DMFEM_USE_CEED=YES, the libCEED CPU backends.
//
// Two modes:
//   --bench   : sweep p = 1..8 on a structured hex mesh sized to ~--target unique
//               dofs, time the operator apply for the selected --device, and
//               append rows to --csv. Run once per device string.
//   --export  : small mesh (n = --n, default 3), native PA. Dump, per p, the
//               exact element basis (B, dB), the per-quad data D (mass weight for
//               BP1; symmetric 3x3 metric for BP3), the lexicographic element
//               restriction map, and a reference (x_in, y_out = A x). diff_bp.cpp
//               then re-applies mf-kernels with this identical data and checks the
//               global vectors agree to <= 1e-13. Run once (native).
//
// Device strings (pass with --device):
//   cpu                              native MFEM PA
//   ceed-cpu:/cpu/self/avx/blocked   libCEED, AVX blocked backend
//   ceed-cpu:/cpu/self/xsmm/blocked  libCEED, LIBXSMM blocked backend
//
// Quadrature follows the CEED BP convention: n_q = p+2 Gauss-Legendre points per
// dimension, i.e. integration rule order 2*(p+2)-1 = 2*p+3 on the cube. Basis is
// the order-p Gauss-Lobatto nodal H1 space (n_d = p+1 nodes per dimension).
//
// ---- BUILD ----  needs a real MFEM install; see scripts/build.sh.
//   g++ -std=c++17 -O3 -march=native mfem_bp.cpp -o mfem_bp \
//       -I$MFEM_DIR/include -L$MFEM_DIR/lib -lmfem [ceed/xsmm libs pulled by mfem]
//
// ---- TWEAK RISKS (see README + methodology_flags) ----
//  (1) Backend device string spelling: verify with `mfem_bp --device ... ` and
//      Device::Print(); the exact libCEED resource path must exist in your build.
//  (2) GeometricFactors metric assembly for DiffusionIntegrator: the 6-component
//      symmetric layout exported here is (xx,xy,xz,yy,yz,zz); confirm sign/scale
//      against a 1-element hand check before trusting the p-sweep.
//  (3) ElementRestriction with LEXICOGRAPHIC ordering is assumed conforming with
//      no essential BCs; that holds for the BP setup (no Dirichlet) used here.
#include "mfem.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
using namespace mfem;

static double now_s(){
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int pick_n(int p, long target){
  int n=(int)std::lround((std::cbrt((double)target)-1.0)/p);
  return n<2?2:n;
}

// Assemble per-quadrature-point data in mf-kernels' ordering.
// Quad index Q = q3 + nq*q2 + nq*nq*q1  (q3 fastest), matching bp_operator.hpp.
// BP1: qd1[elem][Q] = w_Q * detJ_Q.
// BP3: qd6[elem][Q][0..5] = w_Q * detJ_Q * (J^{-1} J^{-T})  (xx,xy,xz,yy,yz,zz).
static void build_qdata(Mesh& mesh, const IntegrationRule& ir, int nq, bool bp1,
                        std::vector<double>& qd){
  const int dim=3;
  const GeometricFactors* g = mesh.GetGeometricFactors(
      ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
  const int NE = mesh.GetNE();
  const int nqp = ir.GetNPoints();           // = nq^3
  // GeometricFactors store J as (NQ, dim, dim, NE) and detJ as (NQ, NE) with the
  // quadrature point index in the rule's native (lexicographic, x-fastest) order.
  // We remap that native q-order to our Q (q3-fastest) below.
  // Map: rule point index pr corresponds to (qa,qb,qc) with qa x-fastest.
  qd.assign((size_t)NE*nqp*(bp1?1:6), 0.0);
  for (int e=0;e<NE;++e){
    for (int q1=0;q1<nq;++q1)for(int q2=0;q2<nq;++q2)for(int q3=0;q3<nq;++q3){
      int Qmine=(q1*nq+q2)*nq+q3;             // q3 fastest
      int pr   =(q3*nq+q2)*nq+q1;             // rule order: x(=q1) fastest -> index q1 fastest
      const IntegrationPoint& ip = ir.IntPoint(pr);
      double w = ip.weight;
      double detJ = g->detJ((size_t)pr + (size_t)nqp*e);
      if (bp1){
        qd[((size_t)e*nqp+Qmine)] = w*detJ;
      } else {
        // J is column-major dim x dim at (pr,e)
        double J[9];
        for(int a=0;a<dim;++a)for(int b=0;b<dim;++b)
          J[a+3*b]=g->J(((size_t)pr + (size_t)nqp*e)*dim*dim + a + dim*b);
        // invert J
        double d=detJ;
        double inv[9];
        inv[0]=(J[4]*J[8]-J[7]*J[5])/d; inv[3]=-(J[3]*J[8]-J[6]*J[5])/d; inv[6]=(J[3]*J[7]-J[6]*J[4])/d;
        inv[1]=-(J[1]*J[8]-J[7]*J[2])/d; inv[4]=(J[0]*J[8]-J[6]*J[2])/d; inv[7]=-(J[0]*J[7]-J[6]*J[1])/d;
        inv[2]=(J[1]*J[5]-J[4]*J[2])/d; inv[5]=-(J[0]*J[5]-J[3]*J[2])/d; inv[8]=(J[0]*J[4]-J[3]*J[1])/d;
        // M = w*detJ * Jinv * Jinv^T  (symmetric)
        double s=w*d;
        auto row=[&](int r,int c){return inv[r+3*0]*inv[c+3*0]+inv[r+3*1]*inv[c+3*1]+inv[r+3*2]*inv[c+3*2];};
        double* o=&qd[((size_t)e*nqp+Qmine)*6];
        o[0]=s*row(0,0); o[1]=s*row(0,1); o[2]=s*row(0,2);
        o[3]=s*row(1,1); o[4]=s*row(1,2); o[5]=s*row(2,2);
      }
    }
  }
}

static void export_case(int p, Mesh& mesh, FiniteElementSpace& fes,
                        const IntegrationRule& ir, bool bp1, const std::string& dir){
  const int nd=p+1, nq=p+2;
  const FiniteElement* fe = fes.GetFE(0);
  const DofToQuad& maps = fe->GetDofToQuad(ir, DofToQuad::TENSOR); // 1D B,G
  // maps.B : (nq, nd) row-major with q fastest? MFEM stores B as (nqpt1d, ndof1d).
  // maps.G : (nq, nd) 1D derivative. Write in our layout B[q*nd+i].
  std::ofstream fb(dir+"/B_"+std::to_string(p)+".txt");
  fb<<nq<<" "<<nd<<"\n";
  for(int q=0;q<nq;++q){ for(int i=0;i<nd;++i) fb<<maps.B[q+nq*i]<<" "; fb<<"\n"; }
  std::ofstream fg(dir+"/G_"+std::to_string(p)+".txt");
  fg<<nq<<" "<<nd<<"\n";
  for(int q=0;q<nq;++q){ for(int i=0;i<nd;++i) fg<<maps.G[q+nq*i]<<" "; fg<<"\n"; }

  // element restriction (lexicographic) -> per-element global L-vector indices
  const Operator* Rop = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
  const ElementRestriction* R = dynamic_cast<const ElementRestriction*>(Rop);
  const int NE=fes.GetNE(), nd3=nd*nd*nd;
  std::ofstream fm(dir+"/emap_"+std::to_string(p)+".txt");
  fm<<NE<<" "<<nd3<<" "<<fes.GetVSize()<<"\n";
  // Use a gather of an index vector to recover the map: apply R to [0,1,2,...].
  Vector idxL(fes.GetVSize()); for(int i=0;i<idxL.Size();++i) idxL[i]=i;
  Vector idxE(R->Height()); R->Mult(idxL, idxE);
  for(int e=0;e<NE;++e){ for(int j=0;j<nd3;++j) fm<<(long)std::llround(idxE[(size_t)e*nd3+j])<<" "; fm<<"\n"; }

  // qdata
  std::vector<double> qd; build_qdata(mesh, ir, nq, bp1, qd);
  std::ofstream fq(dir+"/qd_"+std::to_string(p)+".txt");
  fq<<NE<<" "<<nq<<" "<<(bp1?1:6)<<"\n";
  for(double v:qd) fq<<v<<" ";
  fq<<"\n";

  // reference vectors: x random, y = A x via native PA
  BilinearForm a(&fes);
  a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
  ConstantCoefficient one(1.0);
  if(bp1) a.AddDomainIntegrator(new MassIntegrator(one,&ir));
  else    a.AddDomainIntegrator(new DiffusionIntegrator(one,&ir));
  a.Assemble();
  Vector x(fes.GetVSize()), y(fes.GetVSize());
  for(int i=0;i<x.Size();++i) x[i]=std::sin(0.3*i+1.0)+0.1*((i*2654435761u)%1000)/1000.0;
  a.Mult(x,y);
  std::ofstream fx(dir+"/x_"+std::to_string(p)+".txt"); fx<<x.Size()<<"\n"; for(int i=0;i<x.Size();++i) fx<<x[i]<<" ";
  std::ofstream fy(dir+"/y_"+std::to_string(p)+".txt"); fy<<y.Size()<<"\n"; for(int i=0;i<y.Size();++i) fy<<y[i]<<" ";
}

int main(int argc, char** argv){
  const char* device_config="cpu";
  const char* op="bp1";
  const char* csv=nullptr;
  const char* expdir=nullptr;
  long target=2000000;
  int nexp=3, reps=21;
  for(int i=1;i<argc;++i){
    if(!strcmp(argv[i],"--device")&&i+1<argc) device_config=argv[++i];
    else if(!strcmp(argv[i],"--op")&&i+1<argc) op=argv[++i];
    else if(!strcmp(argv[i],"--csv")&&i+1<argc) csv=argv[++i];
    else if(!strcmp(argv[i],"--export")&&i+1<argc) expdir=argv[++i];
    else if(!strcmp(argv[i],"--target")&&i+1<argc) target=atol(argv[++i]);
    else if(!strcmp(argv[i],"--n")&&i+1<argc) nexp=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--reps")&&i+1<argc) reps=atoi(argv[++i]);
  }
  bool bp1=!strcmp(op,"bp1");
  Device device(device_config);
  device.Print();

  const int dim=3;
  FILE* f = csv ? fopen(csv,"a") : nullptr;

  for(int p=1;p<=8;++p){
    int n = expdir? nexp : pick_n(p,target);
    Mesh mesh = Mesh::MakeCartesian3D(n,n,n,Element::HEXAHEDRON);
    H1_FECollection fec(p, dim, BasisType::GaussLobatto);
    FiniteElementSpace fes(&mesh, &fec);
    const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 2*p+3); // q=p+2

    if(expdir){
      export_case(p, mesh, fes, ir, bp1, expdir);
      printf("[export] p=%d  NE=%d  vsize=%d  -> %s\n",p,fes.GetNE(),fes.GetVSize(),expdir);
      continue;
    }

    BilinearForm a(&fes);
    a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    ConstantCoefficient one(1.0);
    if(bp1) a.AddDomainIntegrator(new MassIntegrator(one,&ir));
    else    a.AddDomainIntegrator(new DiffusionIntegrator(one,&ir));
    a.Assemble();

    Vector x(fes.GetVSize()), y(fes.GetVSize());
    x.Randomize(1);
    a.Mult(x,y);                       // warmup
    std::vector<double> ts; ts.reserve(reps);
    for(int r=0;r<reps;++r){ double t0=now_s(); a.Mult(x,y); ts.push_back(now_s()-t0); }
    std::sort(ts.begin(),ts.end());
    double med=ts[ts.size()/2];
    double iqr=ts[(3*ts.size())/4]-ts[ts.size()/4];
    long udof=fes.GetTrueVSize();
    printf("%-4s p=%-2d dev=%-30s NE=%-8d uDOF=%-9ld med=%.3es DOF/s=%.3eG\n",
           op,p,device_config,fes.GetNE(),udof,med,udof/med*1e-9);
    if(f) fprintf(f,"%s,%d,%s,%d,%ld,%.6e,%.6e,%.6e\n",
                  op,p,device_config,fes.GetNE(),udof,med,iqr,udof/med);
  }
  if(f) fclose(f);
  return 0;
}
