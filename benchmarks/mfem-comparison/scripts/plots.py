#!/usr/bin/env python3
"""plots.py - merge mfk.csv + mfem.csv and emit comparison figures.

Outputs (into <results_dir>):
  dofs_per_sec_bp1.png, dofs_per_sec_bp3.png   throughput vs polynomial order
  gflops_bp1.png,       gflops_bp3.png         GFLOP/s vs polynomial order

Reads:
  mfk.csv  : operator,p,variant,elements,unique_dofs,evec_dofs,median_s,iqr_s,dofs_per_s,gflops
  mfem.csv : op,p,device,NE,uDOF,median_s,iqr_s,dofs_per_s

All four implementations are plotted on one axis per operator so the like-for-like
comparison is immediate. No claim is annotated on the plots; interpretation lives
in the write-up.
"""
import csv, sys, os
from collections import defaultdict

def load_mfk(path):
    d = defaultdict(dict)  # (op)->variant->list[(p,dps,gflops)]
    if not os.path.exists(path): return d
    with open(path) as f:
        for r in csv.DictReader(f):
            op = r["operator"].upper()
            d[op].setdefault(r["variant"], []).append(
                (int(r["p"]), float(r["dofs_per_s"]), float(r["gflops"])))
    for op in d:
        for v in d[op]: d[op][v].sort()
    return d

def load_mfem(path):
    d = defaultdict(dict)  # op->device->list[(p,dps)]
    if not os.path.exists(path): return d
    with open(path) as f:
        for r in csv.DictReader(f):
            op = r["op"].upper()
            dev = r["device"]
            label = {"cpu":"MFEM PA (native)"}.get(dev, dev.split("/")[-2] if "/" in dev else dev)
            if "avx" in dev: label = "libCEED avx/blocked"
            if "xsmm" in dev: label = "libCEED xsmm/blocked"
            d[op].setdefault(label, []).append((int(r["p"]), float(r["dofs_per_s"])))
    for op in d:
        for v in d[op]: d[op][v].sort()
    return d

def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots"); return
    mfk = load_mfk(os.path.join(res, "mfk.csv"))
    mfem = load_mfem(os.path.join(res, "mfem.csv"))
    ops = sorted(set(list(mfk.keys()) + list(mfem.keys())))
    for op in ops:
        # throughput
        plt.figure(figsize=(7,5))
        for v, pts in sorted(mfk.get(op, {}).items()):
            xs=[p for p,_,_ in pts]; ys=[d/1e9 for _,d,_ in pts]
            plt.plot(xs, ys, marker="o", label=f"mf-kernels {v}")
        for v, pts in sorted(mfem.get(op, {}).items()):
            xs=[p for p,_ in pts]; ys=[d/1e9 for _,d in pts]
            plt.plot(xs, ys, marker="s", linestyle="--", label=v)
        plt.xlabel("polynomial order p"); plt.ylabel("unique DOFs/sec (G)")
        plt.title(f"{op}: throughput vs order (single core)")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(res, f"dofs_per_sec_{op.lower()}.png"), dpi=130)
        plt.close()
        # gflops (mf-kernels only; MFEM gflops not directly emitted)
        plt.figure(figsize=(7,5))
        for v, pts in sorted(mfk.get(op, {}).items()):
            xs=[p for p,_,_ in pts]; ys=[g for _,_,g in pts]
            plt.plot(xs, ys, marker="o", label=f"mf-kernels {v}")
        plt.xlabel("polynomial order p"); plt.ylabel("GFLOP/s (shared standard count)")
        plt.title(f"{op}: GFLOP/s vs order (single core)")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(res, f"gflops_{op.lower()}.png"), dpi=130)
        plt.close()
    print(f"wrote plots into {res}/")

if __name__ == "__main__":
    main()
