// mfem_bp.cpp - MFEM side of the comparison: native partial assembly (PA) and,
// when MFEM is built with -DMFEM_USE_CEED=YES, the libCEED CPU backends.
//
// Two modes:
//   --bench   : sweep p = 1..8 on a structured hex mesh sized to ~--target unique
//               dofs, time the operator apply for the selected --device, and
//               append rows to --csv. Run once per device string.
//   --export  : small mesh (n = --n, default 3), native PA, into <dir>/<op>/. Dump,
//               per p: the lexicographic element restriction map, a reference pair
//               (x_in, y_out = A x), the mesh size, and a convention-independent
//               invariant (1^T M 1 = domain volume for BP1; ||A 1|| ~ 0 for BP3).
//               diff_bp.cpp re-applies mf-kernels (building the basis itself) and
//               checks the global vectors agree to <= 1e-13. Run once per op.
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
//  (2) ElementRestriction with LEXICOGRAPHIC ordering is assumed conforming with
//      no essential BCs; that holds for the BP setup (no Dirichlet) used here.
//  (3) The exported invariant is the quickest convention check: if BP1's 1^T M 1
//      is not ~1.0 (unit-cube volume) or BP3's ||A 1|| is not ~0, stop before the
//      p-sweep -- the discretization or geometry is off, not the kernels.
#include "mfem.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
using namespace mfem;

static double now_s(){
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int pick_n(int p, long target){
  int n=(int)std::lround((std::cbrt((double)target)-1.0)/p);
  return n<2?2:n;
}


static void export_case(int p, Mesh& mesh, FiniteElementSpace& fes,
                        const IntegrationRule& ir, bool bp1, const std::string& dir){
  const int nd=p+1;
  const int P=17; // full double precision for the 1e-13 check

  // element restriction (lexicographic) -> per-element global L-vector indices
  const Operator* Rop = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
  const ElementRestriction* R = dynamic_cast<const ElementRestriction*>(Rop);
  const int NE=fes.GetNE(), nd3=nd*nd*nd;
  std::ofstream fm(dir+"/emap_"+std::to_string(p)+".txt");
  fm<<NE<<" "<<nd3<<" "<<fes.GetVSize()<<"\n";
  Vector idxL(fes.GetVSize()); for(int i=0;i<idxL.Size();++i) idxL[i]=i;
  Vector idxE(R->Height()); R->Mult(idxL, idxE);
  for(int e=0;e<NE;++e){ for(int j=0;j<nd3;++j) fm<<(long)std::llround(idxE[(size_t)e*nd3+j])<<" "; fm<<"\n"; }

  // operator A (native PA)
  BilinearForm a(&fes);
  a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
  ConstantCoefficient one(1.0);
  if(bp1) a.AddDomainIntegrator(new MassIntegrator(one,&ir));
  else    a.AddDomainIntegrator(new DiffusionIntegrator(one,&ir));
  a.Assemble();

  // reference vectors: deterministic x, y = A x
  Vector x(fes.GetVSize()), y(fes.GetVSize());
  for(int i=0;i<x.Size();++i) x[i]=std::sin(0.3*i+1.0)+0.1*((i*2654435761u)%1000)/1000.0;
  a.Mult(x,y);
  { std::ofstream fx(dir+"/x_"+std::to_string(p)+".txt"); fx<<std::setprecision(P);
    fx<<x.Size()<<"\n"; for(int i=0;i<x.Size();++i) fx<<x[i]<<" "; }
  { std::ofstream fy(dir+"/y_"+std::to_string(p)+".txt"); fy<<std::setprecision(P);
    fy<<y.Size()<<"\n"; for(int i=0;i<y.Size();++i) fy<<y[i]<<" "; }

  // convention-independent invariant: bp1 -> 1^T M 1 = domain volume (=1 for unit
  // cube); bp3 -> ||A 1|| ~ 0 (gradient of a constant is zero).
  Vector ones(fes.GetVSize()); ones=1.0; Vector Aones(fes.GetVSize()); a.Mult(ones,Aones);
  double inv = bp1 ? (ones*Aones) : Aones.Norml2();
  { std::ofstream fi(dir+"/invariant_"+std::to_string(p)+".txt"); fi<<std::setprecision(P)<<inv<<"\n"; }
  printf("   [invariant] p=%d  %s = %.6e  (bp1 expect ~1.0 = unit-cube volume; bp3 expect ~0)\n",
         p, bp1?"1^T M 1":"||A 1||", inv);
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
      { std::string ed=std::string(expdir)+"/"+op; mkdir(expdir,0777); mkdir(ed.c_str(),0777);
        export_case(p, mesh, fes, ir, bp1, ed);
        std::ofstream fmeta(ed+"/meta.txt"); fmeta<<n<<"\n"; }
      printf("[export] p=%d  NE=%d  vsize=%d  -> %s/%s\n",p,fes.GetNE(),fes.GetVSize(),expdir,op);
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
