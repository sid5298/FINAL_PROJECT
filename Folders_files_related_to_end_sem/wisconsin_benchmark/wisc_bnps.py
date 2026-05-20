"""
wisc_bnps.py  —  BNPS Serial SLP + BNPS CUDA SLP benchmark.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paste as Cell 2 in Colab AFTER running Cell 1 (wisc_config).
Also upload to Colab: bnps3.py, bnps_fast.cu (or bnps.cu)

Outputs: wisc_bnps_results.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTE: No imports needed — Cell 1 (wisc_config) defines everything.
"""

import os, sys, re, time, json, subprocess, shutil
import numpy as np
# All other names (X_train, X_test, y_train, y_test, NF, MEMBRANE_COUNTS,
# STEPS_SWEEP, RUNS, stratified_sample, metrics, piecewise_sigmoid,
# MembraneSLP, make_pep, extract_weights, BNPS_LR) come from Cell 1.

CWD = os.getcwd()

# ── COMPILE CUDA ──────────────────────────────────────────────────────
print("Compiling BNPS CUDA kernel...")
_nvcc = shutil.which('nvcc')
if _nvcc is None:
    for p in ['/usr/local/cuda/bin/nvcc','/usr/local/cuda-12/bin/nvcc','/usr/bin/nvcc']:
        if os.path.exists(p): _nvcc = p; break

CUDA_OK = False
if _nvcc:
    cu = 'bnps_fast.cu' if os.path.exists('bnps_fast.cu') else 'bnps.cu'
    if os.path.exists(cu):
        ret = subprocess.run([_nvcc, '-O2', '-o', 'bnps_cuda', cu], capture_output=True)
        CUDA_OK = (ret.returncode == 0)
        print(f"  CUDA compiled OK ({cu})" if CUDA_OK else f"  CUDA compile FAILED: {ret.stderr.decode()[:200]}")
    else:
        print(f"  No .cu file found — CUDA disabled")
else:
    print("  nvcc not found — CUDA disabled (serial only)")

# ── STARTUP OVERHEAD ──────────────────────────────────────────────────
_oh = []
for _ in range(10):
    t = time.time()
    subprocess.run([sys.executable, 'bnps3.py'], capture_output=True, timeout=15)
    _oh.append((time.time()-t)*1000)
PY_OH = float(np.median(_oh))
print(f"  bnps3.py startup overhead: {PY_OH:.1f} ms\n")

# ── MAIN SWEEP ────────────────────────────────────────────────────────
results = {}   # key: (nm, steps)

print(f"{'Mems':>5} {'Steps':>6} {'Serial(ms)':>12} {'CUDA(ms)':>10} "
      f"{'Speedup':>9} {'Acc':>7} {'F1':>7}")
print("-" * 60)

for nm in MEMBRANE_COUNTS:
    Xm, ym = stratified_sample(X_train, y_train, nm)
    _idx   = slice(None, nm)

    for steps in STEPS_SWEEP:
        key = f"{nm}_{steps}"

        # ── Generate .pep file ────────────────────────────────────────
        pf = os.path.join(CWD, f'bc_{nm}_{steps}.pep')
        make_pep(Xm, ym, nm, NF, pf, lr=BNPS_LR)

        # ── BNPS Serial ───────────────────────────────────────────────
        ser_ts = []; last_out = ''
        for _ in range(RUNS):
            t0 = time.time()
            r  = subprocess.run([sys.executable, 'bnps3.py', pf, '-n', str(steps)],
                                capture_output=True, timeout=600, cwd=CWD)
            ser_ts.append(max((time.time()-t0)*1000 - PY_OH, 1.0))
            last_out = r.stdout.decode(errors='replace')

        ser_ms  = float(np.median(ser_ts))
        ser_std = float(np.std(ser_ts))

        # Extract weights → accuracy
        w, b = extract_weights(last_out, NF)
        acc_s = f1_s = prec_s = rec_s = float('nan')
        if w is not None:
            logits = X_test @ w + b
            prob   = 1 / (1 + np.exp(-logits))
            m_s    = metrics(y_test, prob)
            acc_s, f1_s = m_s['acc'], m_s['f1']
            prec_s, rec_s = m_s['prec'], m_s['rec']

        # ── BNPS CUDA ─────────────────────────────────────────────────
        cuda_ms = cuda_std = float('nan')
        if CUDA_OK:
            inp = os.path.join(CWD, 'input.txt')
            # Generate input.txt via -p flag
            subprocess.run([sys.executable, 'bnps3.py', pf, '-p', str(steps)],
                           capture_output=True, timeout=600, cwd=CWD)
            # Warmup
            subprocess.run(['./bnps_cuda', inp], capture_output=True, timeout=60, cwd=CWD)
            cuda_ts = []
            for _ in range(RUNS):
                t0  = time.time()
                rc  = subprocess.run(['./bnps_cuda', inp], capture_output=True,
                                     timeout=300, cwd=CWD)
                wall = (time.time()-t0)*1000
                out  = rc.stdout.decode(errors='replace')
                mx   = re.search(r'[Tt]ime[^:]*:\s*([\d.]+)\s*(ms|us|s)?', out)
                if mx:
                    v = float(mx.group(1)); u = (mx.group(2) or 'ms').lower()
                    cuda_ts.append(v*1e-3 if u=='us' else (v*1e3 if u=='s' else v))
                else:
                    cuda_ts.append(wall)
            cuda_ms  = float(np.median(cuda_ts))
            cuda_std = float(np.std(cuda_ts))

        speedup = ser_ms / cuda_ms if (CUDA_OK and cuda_ms > 0) else float('nan')

        results[key] = dict(
            nm=nm, steps=steps,
            serial_ms=ser_ms, serial_std=ser_std,
            cuda_ms=cuda_ms, cuda_std=cuda_std,
            speedup=speedup,
            acc=acc_s, f1=f1_s, prec=prec_s, rec=rec_s
        )

        acc_str = f"{acc_s:.4f}" if not np.isnan(acc_s) else "N/A"
        f1_str  = f"{f1_s:.4f}"  if not np.isnan(f1_s)  else "N/A"
        spd_str = f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A"
        cud_str = f"{cuda_ms:.1f}" if not np.isnan(cuda_ms) else "N/A"
        print(f"{nm:>5} {steps:>6} {ser_ms:>12.1f} {cud_str:>10} "
              f"{spd_str:>9} {acc_str:>7} {f1_str:>7}")

# ── SAVE ─────────────────────────────────────────────────────────────
with open('wisc_bnps_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: wisc_bnps_results.json")

# ── QUICK SUMMARY ─────────────────────────────────────────────────────
valid = [(v['acc'], k) for k, v in results.items() if not np.isnan(v['acc'])]
if valid:
    best_acc, best_key = max(valid)
    bv = results[best_key]
    print(f"\nBest BNPS Serial: nm={bv['nm']} steps={bv['steps']} "
          f"Acc={bv['acc']:.4f} F1={bv['f1']:.4f} Time={bv['serial_ms']:.1f}ms")
    if CUDA_OK and not np.isnan(bv['cuda_ms']):
        print(f"   CUDA: {bv['cuda_ms']:.1f}ms  Speedup={bv['speedup']:.2f}x")
