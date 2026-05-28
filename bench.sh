#!/usr/bin/env bash
set -e

echo "Building mf-kernels..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo -e "\nRunning benchmarks..."
./build/bench_sumfact --csv results.csv > /dev/null

echo -e "\n================================================================="
echo "Benchmark Results (Markdown Table format for Q5, Q7, Q9)"
echo "================================================================="
echo ""
echo "| p | naive | pitfall | avx2 | blocked | evenodd | best vs naive |"
echo "|---|------:|--------:|-----:|--------:|--------:|--------------:|"

python3 -c "
import csv
data = {}
with open('results.csv', 'r') as f:
    for row in csv.DictReader(f):
        p = int(row['p'])
        if p in [5, 7, 9]:
            data.setdefault(p, {})[row['variant']] = {
                'gflops': float(row['gflops']),
                'speedup': float(row['speedup_vs_naive'])
            }

for p in [5, 7, 9]:
    if p not in data: continue
    d = data[p]
    def g(v): return f\"{d[v]['gflops']:.1f}\" if v in d else '—'
    speedups = [d[v]['speedup'] for v in d]
    best_str = f\"**{max(speedups):.2f}×**\" if speedups else '—'
    print(f\"| {p} | {g('naive'):>4s} | {g('pitfall'):>5s} | {g('avx2'):>4s} | {g('avx2_blocked'):>7s} | {g('evenodd_avx2'):>7s} | {best_str:>13s} |\")
"
