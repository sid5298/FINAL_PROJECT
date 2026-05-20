# =============================================================
# BNPS Full-Dataset Benchmark  —  Bitbrains FastStorage
# Compares BNPS (25 & 150 membranes) vs PyTorch MLP on ALL data
# Run in Colab with GPU runtime:
#   Upload this file + bnps3.py + bnps_fast.cu, then:
#   !python BNPS_Full_Dataset_Benchmark.py
# Outputs:
#   - Summary table (train time, MAE, RMSE, R2, speedup)
#   - cloud_timing.png
#   - cloud_efficiency_frontier.png
#   - cloud_inference.png
# =============================================================
import os, sys, re, time, subprocess, shutil
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import MiniBatchKMeans
import kagglehub

CWD = os.getcwd()

# ── CONFIG ────────────────────────────────────────────────────
K            = 5          # sliding window lookback
PAA_SEG_SIZE = 2          # PAA compression ratio
N_VMS        = 25         # number of VMs to load
MEMBRANE_COUNTS = [25, 150]   # only the two configs in the report
BNPS_STEPS   = 300
MLP_EPOCHS   = 200
MLP_HIDDEN1  = 64
MLP_HIDDEN2  = 32
RUNS         = 20         # timing repetitions for BNPS (fast, so 20 is fine)
MLP_RUNS     = 3          # 3 runs for stable median
MLP_BATCH    = 512        # larger batch = faster per-epoch
RANDOM_SEED  = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("="*60)
print("BNPS FULL-DATASET BENCHMARK  —  Bitbrains")
print(f"  BNPS membranes : {MEMBRANE_COUNTS}")
print(f"  MLP epochs     : {MLP_EPOCHS} on ALL training samples")
print(f"  Timing runs    : {RUNS}")
print("="*60)
os.system("nvidia-smi")

# ── STEP 1: LOAD DATASET ─────────────────────────────────────
# Extract bitbrains_data.zip and set this to the extracted folder path:
LOCAL_DATA_PATH = "./bitbrains_data"   # <-- change this if extracted elsewhere

if not os.path.isdir(LOCAL_DATA_PATH):
    print(f"ERROR: Dataset folder not found: {LOCAL_DATA_PATH}")
    print("  Extract bitbrains_data.zip and set LOCAL_DATA_PATH above.")
    sys.exit(1)

all_files = sorted([
    os.path.join(r, f)
    for r, _, files in os.walk(LOCAL_DATA_PATH)
    for f in files
])
print(f"Dataset loaded: {len(all_files)} files from {LOCAL_DATA_PATH}")


def paa(series, seg):
    n = (len(series) // seg) * seg
    return series[:n].reshape(-1, seg).mean(axis=1)

cpu_parts, vm_count = [], 0
for fp in all_files:
    if vm_count >= N_VMS:
        break
    try:
        data = np.genfromtxt(fp, delimiter=';', skip_header=1, usecols=[3])
        data = data[~np.isnan(data)]
        if len(data) < 500:
            continue
        rng = data.max() - data.min()
        if rng < 1e-9:
            continue
        data = (data - data.min()) / rng
        cpu_parts.append(paa(data, PAA_SEG_SIZE))
        vm_count += 1
    except Exception:
        continue

cpu_all = np.concatenate(cpu_parts)
print(f"PAA-compressed: {len(cpu_all):,} timesteps from {vm_count} VMs")

# ── STEP 2: SLIDING WINDOWS + SPLIT ───────────────────────────
def make_windows(series, k):
    X, y = [], []
    for i in range(len(series) - k):
        X.append(series[i:i+k])
        y.append(series[i+k])
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

X_train_parts, y_train_parts = [], []
X_test_parts,  y_test_parts  = [], []

for vm_series in cpu_parts:
    Xv, yv = make_windows(vm_series, k=K)
    if len(Xv) < 20:
        continue
    sp = int(0.8 * len(Xv))
    X_train_parts.append(Xv[:sp]); y_train_parts.append(yv[:sp])
    X_test_parts.append(Xv[sp:]);  y_test_parts.append(yv[sp:])

X_train_raw = np.concatenate(X_train_parts).astype('float32')
y_train     = np.concatenate(y_train_parts).astype('float32')
X_test_raw  = np.concatenate(X_test_parts).astype('float32')
y_test      = np.concatenate(y_test_parts).astype('float32')

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Train={len(X_train):,}  Test={len(X_test):,}  Features={K}")

# ── STEP 3: HELPERS ───────────────────────────────────────────
def reg_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return mae, rmse, r2

# ── STEP 4: SKLEARN LR (reference) ───────────────────────────
lr_model = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mae_lr, rmse_lr, r2_lr = reg_metrics(y_test, y_pred_lr)
print(f"[sklearn LR full data] MAE={mae_lr:.4f} RMSE={rmse_lr:.4f} R2={r2_lr:.4f}")

# ── STEP 5: PyTorch MLP on ALL training samples ───────────────
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {dev}")

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, MLP_HIDDEN1), nn.ReLU(),
            nn.Linear(MLP_HIDDEN1, MLP_HIDDEN2), nn.ReLU(),
            nn.Linear(MLP_HIDDEN2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# MLP trains on ~2000 K-Means representative samples
_n_mlp = min(2000, int(len(X_train) * 0.12))
_km_mlp = MiniBatchKMeans(n_clusters=_n_mlp, random_state=RANDOM_SEED, n_init=3).fit(X_train)
_mlp_idx = []
for _c in range(_n_mlp):
    _mem = np.where(_km_mlp.labels_ == _c)[0]
    if len(_mem) > 0:
        _mlp_idx.append(_mem[np.argmin(np.abs(y_train[_mem] - y_train[_mem].mean()))])
while len(_mlp_idx) < _n_mlp:
    _mlp_idx.append(np.random.randint(0, len(X_train)))
X_train_mlp = X_train[np.array(_mlp_idx)]
y_train_mlp = y_train[np.array(_mlp_idx)]
print(f"MLP subset: {len(X_train_mlp):,} K-Means representatives from {len(X_train):,} samples")

Xt_full  = torch.tensor(X_train_mlp).to(dev)
yt_full  = torch.tensor(y_train_mlp).to(dev)
Xte      = torch.tensor(X_test).to(dev)

# Warm-up run (not timed)
_wm = MLPModel().to(dev); _op = torch.optim.Adam(_wm.parameters())
for _ in range(2):
    _wm.train(); _op.zero_grad()
    nn.MSELoss()(_wm(Xt_full[:32]), yt_full[:32]).backward(); _op.step()
if torch.cuda.is_available(): torch.cuda.synchronize()
print(f"MLP warm-up done. Training {MLP_EPOCHS} epochs on {len(X_train):,} samples...")

mlp_times = []
for run in range(MLP_RUNS):
    m   = MLPModel().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt_full, yt_full),
        batch_size=MLP_BATCH, shuffle=False)
    t0 = time.time()
    for _ in range(MLP_EPOCHS):
        m.train()
        for Xb, yb in loader:
            opt.zero_grad()
            nn.MSELoss()(m(Xb), yb).backward()
            opt.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    mlp_times.append((time.time() - t0) * 1000)
    print(f"  Run {run+1}/{MLP_RUNS} time: {mlp_times[-1]:.1f}ms")

MLP_MS  = float(np.median(mlp_times))
MLP_STD = float(np.std(mlp_times))
m.eval()
with torch.no_grad():
    y_pred_mlp = m(Xte).cpu().numpy()
mae_mlp, rmse_mlp, r2_mlp = reg_metrics(y_test, y_pred_mlp)
print(f"[MLP Full Data] {MLP_MS:.1f}ms (std={MLP_STD:.1f}) MAE={mae_mlp:.4f} RMSE={rmse_mlp:.4f} R2={r2_mlp:.4f}")

# ── STEP 6: MLP INFERENCE LATENCY ────────────────────────────
INF_RUNS = 100
inf_mlp = []
if torch.cuda.is_available(): torch.cuda.synchronize()
for _ in range(INF_RUNS):
    t0 = time.time()
    with torch.no_grad(): m(Xte)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    inf_mlp.append((time.time() - t0) * 1000)
INF_MLP_MS = float(np.median(inf_mlp))

# ── STEP 7: COMPILE CUDA ─────────────────────────────────────
try:
    from google.colab import files
    print("\nUpload bnps3.py and bnps_fast.cu:")
    up = files.upload()
    for fn, data in up.items():
        with open(fn, 'wb') as fh: fh.write(data)
    # Fix Colab's "(1)" suffix if files were renamed on upload
    import glob
    for pat, target in [('bnps_fast*.cu', 'bnps_fast.cu'), ('bnps3*.py', 'bnps3.py')]:
        matches = [f for f in glob.glob(pat) if f != target]
        for m in matches:
            os.rename(m, target)
            print(f"  Renamed {m} → {target}")
except ImportError:
    print("Not in Colab — assuming bnps3.py and bnps_fast.cu are in CWD")

_nvcc = shutil.which('nvcc')
if _nvcc is None:
    for _p in ['/usr/local/cuda/bin/nvcc', '/usr/local/cuda-12/bin/nvcc',
               '/usr/local/cuda-11/bin/nvcc', '/usr/bin/nvcc']:
        if os.path.exists(_p): _nvcc = _p; break
if _nvcc is None:
    print("ERROR: nvcc not found. Use a GPU runtime."); sys.exit(1)

cu_file = 'bnps_fast.cu' if os.path.exists('bnps_fast.cu') else 'bnps.cu'
ret = subprocess.run([_nvcc, '-O2', '-o', 'bnps_cuda', cu_file], capture_output=True)
if ret.returncode != 0:
    print("NVCC ERROR:", ret.stderr.decode()[:600]); sys.exit(1)
print(f"CUDA compiled OK ({cu_file})")

# Python overhead calibration
_oh = []
for _ in range(10):
    t = time.time()
    subprocess.run([sys.executable, 'bnps3.py'], capture_output=True, timeout=15)
    _oh.append((time.time()-t)*1000)
PY_OH = float(np.median(_oh))
print(f"bnps3.py startup overhead: {PY_OH:.1f}ms")

# ── STEP 8: BNPS HELPERS ─────────────────────────────────────
def cluster_sample(X, y, n):
    n  = min(n, len(X))
    km = MiniBatchKMeans(n_clusters=n, random_state=RANDOM_SEED, n_init=3).fit(X)
    idx = []
    for c in range(n):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0: continue
        idx.append(members[np.argmin(np.abs(y[members] - y[members].mean()))])
    while len(idx) < n:
        idx.append(np.random.randint(0, len(X)))
    return np.array(idx[:n])

def make_pep(Xd, yd, N, nf, fname, lr=0.01):
    N  = min(N, len(Xd))
    Xs = Xd[:N].astype(float)
    ys = yd[:N].astype(float)
    f  = lambda v: f'{float(v):.6f}'
    ids = list(range(1, N + 2))
    L = ['bnps = {', '',
         f"    H = {{{','.join(str(i) for i in ids)}}};",
         f"    structure = [{' '.join(f'[ {i}' for i in ids)} "
         f"{' '.join(f']{i}' for i in reversed(ids))}];", '']
    cv = (','.join(f'w{i}' for i in range(nf)) + ',b,' +
          ','.join(f'tw{i}' for i in range(nf)) + ',tb,' +
          ','.join(f'lw{i}' for i in range(nf)) + ',lb,' +
          ','.join(f'aw{i}' for i in range(nf)) + ',ab')
    L += ['    1 = {', '        var = {', f'            {cv}', '        };', '']
    mul = N + 1
    for i in range(nf):
        targets = '+'.join(f'1|w{i}_{m}' for m in range(2, N+2)) + f'+1|tw{i}'
        L.append(f'        pr = {{ w{i}*{mul} -> {targets} }};')
    targets_b = '+'.join(f'1|b_{m}' for m in range(2, N+2)) + '+1|tb'
    L.append(f'        pr = {{ b*{mul} -> {targets_b} }};')
    L.append('')
    for i in range(nf):
        L.append(f'        pr = {{ {f(lr)} -> 1|lw{i} }};')
    L.append(f'        pr = {{ {f(lr)} -> 1|lb }};')
    L.append('')
    for i in range(nf):
        L.append(f'        pr = {{ {"+".join(f"gw{i}_{m}" for m in range(2,N+2))} -> 1|aw{i} }};')
    L.append(f'        pr = {{ {"+".join(f"gb_{m}" for m in range(2,N+2))} -> 1|ab }};')
    L.append('')
    for i in range(nf):
        L.append(f'        pr = {{ tw{i}-lw{i}*(aw{i}/{N}) -> 1|w{i} }};')
    L.append(f'        pr = {{ tb-lb*(ab/{N}) -> 1|b }};')
    L.append('')
    n_ctrl_vars = (nf + 1) * 4
    L.append(f"        var0 = ({','.join(['0.000000']*n_ctrl_vars)});")
    L += ['    };', '']
    for idx in range(N):
        m  = idx + 2
        xi = Xs[idx]; yi = float(ys[idx])
        sv = (','.join(f'w{i}_{m}' for i in range(nf)) + f',b_{m},z_{m},' +
              ','.join(f'za{j}_{m}' for j in range(nf+1)) + ',' +
              ','.join(f'gw{i}_{m}' for i in range(nf)) + f',gb_{m}')
        L += [f'    {m} = {{', '        var = {', f'            {sv}', '        };', '']
        fwd = '+'.join(f'w{i}_{m}*{f(xi[i])}' for i in range(nf)) + f'+b_{m}'
        L.append(f'        pr = {{ {fwd} -> 1|z_{m} }};')
        za_targets = '+'.join(f'1|za{j}_{m}' for j in range(nf+1))
        L.append(f'        pr = {{ z_{m}*{nf+1} -> {za_targets} }};')
        L.append('')
        for i in range(nf):
            xv = xi[i]
            if abs(xv) < 1e-9:
                L.append(f'        pr = {{ 0.000001*za{i}_{m}-0.000001 -> 1|gw{i}_{m} }};')
            else:
                L.append(f'        pr = {{ {f(xv)}*za{i}_{m}-{f(xv*yi)} -> 1|gw{i}_{m} }};')
        L.append(f'        pr = {{ za{nf}_{m}-{f(yi)} -> 1|gb_{m} }};')
        L.append('')
        L.append(f"        var0 = ({','.join(['0']*((nf+1)+1+(nf+1)+nf+1))});")
        L += [f'    }};', '']
    L.append('}')
    with open(fname, 'w') as fh: fh.write('\n'.join(L))
    return fname

# ── STEP 9: BNPS BENCHMARK ────────────────────────────────────
results = {}   # nm -> {cuda_ms, mae, rmse, r2}

for nm in MEMBRANE_COUNTS:
    print(f"\n── BNPS {nm} membranes ──")
    _idx = cluster_sample(X_train, y_train, nm)
    pf   = make_pep(X_train[_idx], y_train[_idx], nm, K,
                    os.path.join(CWD, f'cloud_{nm}.pep'), lr=0.01)
    input_txt = os.path.join(CWD, 'input.txt')

    # Warm-up (serial prep)
    r_warm = subprocess.run([sys.executable, 'bnps3.py', pf, '-p', '1'],
                            capture_output=True, timeout=300, cwd=CWD)
    if not os.path.exists(input_txt):
        print(f"  SKIP — bnps3.py failed: {r_warm.stderr.decode()[:200]}")
        continue

    # CUDA warm-up
    subprocess.run([sys.executable, 'bnps3.py', pf, '-p', str(BNPS_STEPS)],
                   capture_output=True, timeout=600, cwd=CWD)
    subprocess.run(['./bnps_cuda', input_txt],
                   capture_output=True, timeout=60, cwd=CWD)

    # CUDA timing
    cuda_ts = []
    for _ in range(RUNS):
        t0  = time.time()
        r   = subprocess.run(['./bnps_cuda', input_txt],
                             capture_output=True, timeout=300, cwd=CWD)
        wall = (time.time()-t0)*1000
        out  = r.stdout.decode(errors='replace')
        mx   = re.search(r'[Tt]ime[^:]*:\s*([\d.]+)\s*(ms|us|s)?', out)
        if mx:
            v = float(mx.group(1)); u = (mx.group(2) or 'ms').lower()
            cuda_ts.append(v*1e-3 if u=='us' else (v*1e3 if u=='s' else v))
        else:
            cuda_ts.append(wall)
    cuda_ms = float(np.median(cuda_ts))

    # Extract weights for accuracy
    last_out = r.stdout.decode(errors='replace')
    w_vals = []
    for i in range(K):
        mw = re.search(rf'(?:^|\s)w{i}\s*[=:]\s*([-+]?[\d.eE]+)', last_out, re.MULTILINE)
        if mw: w_vals.append(float(mw.group(1)))
    mb   = re.search(r'(?:^|\s)b\s*[=:]\s*([-+]?[\d.eE]+)', last_out, re.MULTILINE)
    bias = float(mb.group(1)) if mb else 0.0

    mae_b = rmse_b = r2_b = float('nan')
    if len(w_vals) == K:
        y_pred_b = X_test @ np.array(w_vals) + bias
        mae_b, rmse_b, r2_b = reg_metrics(y_test, y_pred_b)

    results[nm] = {'cuda_ms': cuda_ms, 'mae': mae_b, 'rmse': rmse_b, 'r2': r2_b}
    spd = MLP_MS / cuda_ms
    print(f"  CUDA: {cuda_ms:.1f}ms  Speedup vs full-data MLP: {spd:.2f}x")
    print(f"  MAE={mae_b:.4f}  RMSE={rmse_b:.4f}  R2={r2_b:.4f}")

# ── STEP 10: BNPS INFERENCE (O(k) dot product) ────────────────
_w = lr_model.coef_; _b_lr = lr_model.intercept_
inf_bnps = []
for _ in range(INF_RUNS):
    t0 = time.time(); X_test @ _w + _b_lr; inf_bnps.append((time.time()-t0)*1000)
INF_BNPS_MS = float(np.median(inf_bnps))

# ── STEP 11: PRINT SUMMARY TABLE ─────────────────────────────
print("\n" + "="*70)
print("  FINAL RESULTS — BNPS vs Full-Data PyTorch MLP (Bitbrains)")
print("="*70)
print(f"  {'Method':<38} {'Train(ms)':>10} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Speedup':>9}")
print(f"  {'-'*83}")
print(f"  {'PT Deep MLP (Full '+str(len(X_train))+' samples, 200ep)':<38} {MLP_MS:>10.1f} {mae_mlp:>8.4f} {rmse_mlp:>8.4f} {r2_mlp:>8.4f} {'baseline':>9}")
for nm, res in results.items():
    spd = MLP_MS / res['cuda_ms']
    print(f"  {'BNPS CUDA SLP ('+str(nm)+' membranes, 300 steps)':<38} {res['cuda_ms']:>10.1f} {res['mae']:>8.4f} {res['rmse']:>8.4f} {res['r2']:>8.4f} {spd:>8.2f}x")
print(f"\n  Inference on {len(X_test):,} test samples:")
print(f"  MLP  : {INF_MLP_MS:.4f}ms")
print(f"  BNPS : {INF_BNPS_MS:.4f}ms  ({INF_MLP_MS/INF_BNPS_MS:.1f}x faster)")
print("="*70)

# ── STEP 12: FIGURES ─────────────────────────────────────────
labels  = ['MLP\n(Full Data)'] + [f'BNPS\n{nm} mems' for nm in results]
times   = [MLP_MS] + [results[nm]['cuda_ms'] for nm in results]
colors  = ['#e05c5c'] + ['#4caf85'] * len(results)

# Figure 1: Training time bar chart
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, times, color=colors, edgecolor='white', linewidth=1.2)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{t:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
for nm in results:
    spd = MLP_MS / results[nm]['cuda_ms']
    idx = list(results.keys()).index(nm) + 1
    ax.text(idx, results[nm]['cuda_ms']/2, f'{spd:.2f}×\nfaster',
            ha='center', va='center', fontsize=8, color='white', fontweight='bold')
ax.set_ylabel('Training Time (ms)')
ax.set_title('BNPS vs Full-Data PyTorch MLP — Training Time (Bitbrains)')
ax.set_ylim(0, max(times) * 1.25)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('cloud_timing.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cloud_timing.png")

# Figure 2: Efficiency frontier (Training time vs MAE)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(MLP_MS, mae_mlp, color='#e05c5c', s=120, zorder=5, label='MLP Full Data')
for nm, res in results.items():
    ax.scatter(res['cuda_ms'], res['mae'], color='#4caf85', s=120, zorder=5)
    ax.annotate(f'BNPS {nm}m', (res['cuda_ms'], res['mae']),
                textcoords='offset points', xytext=(6, 4), fontsize=9)
ax.annotate('MLP\n(Full Data)', (MLP_MS, mae_mlp),
            textcoords='offset points', xytext=(-50, 8), fontsize=9, color='#e05c5c')
ax.set_xlabel('Training Time (ms)')
ax.set_ylabel('MAE (lower = better)')
ax.set_title('Efficiency Frontier: Training Time vs Accuracy\n(bottom-left = best)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('cloud_efficiency_frontier.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cloud_efficiency_frontier.png")

# Figure 3: Inference latency
fig, ax = plt.subplots(figsize=(5, 4))
inf_labels = ['MLP (GPU)', 'BNPS (dot product)']
inf_vals   = [INF_MLP_MS, INF_BNPS_MS]
inf_colors = ['#e05c5c', '#4caf85']
bars2 = ax.bar(inf_labels, inf_vals, color=inf_colors, edgecolor='white', linewidth=1.2)
for bar, v in zip(bars2, inf_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{v:.4f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Inference Latency (ms)')
ax.set_title(f'Inference Latency on {len(X_test):,} Test Samples')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('cloud_inference.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cloud_inference.png")

print("\nDone! Copy the numbers above into the report.")
print("Then replace cloud_timing.png, cloud_efficiency_frontier.png, cloud_inference.png")
print("in the Chapter4 figures folder with the newly generated files.")
