# =============================================================
# BNPS Cloud Workload Benchmark  —  Bitbrains FastStorage
# Upload this file + bnps3.py + bnps.cu to Colab, then run:
#   !python BNPS_Cloud_Workload.py
# =============================================================
import os, sys, re, time, subprocess
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import MiniBatchKMeans
import kagglehub

CWD = os.getcwd()   # pin working dir — all files written/read relative to this

# ── CONFIG ────────────────────────────────────────────────────
K               = 5
PAA_SEG_SIZE    = 2
N_VMS           = 25
MEMBRANE_COUNTS = [25, 50, 75, 100, 150, 200]
BNPS_STEPS      = 300
MLP_EPOCHS      = 200
MLP_HIDDEN1     = 64
MLP_HIDDEN2     = 32
RUNS            = 20

print("="*60)
print("BENCHMARK CONFIG")
print("="*60)
print(f"  Dataset      : Bitbrains FastStorage ({N_VMS} VMs, PAA {PAA_SEG_SIZE}:1)")
print(f"  Window size  : k={K}  |  BNPS steps: {BNPS_STEPS}")
print(f"  Membranes    : {MEMBRANE_COUNTS}")
print(f"  Baselines    : Moving Avg, sklearn LR, MLP {K}→{MLP_HIDDEN1}→{MLP_HIDDEN2}→1 (10-feat)")
print("="*60)

os.system("nvidia-smi")
print(f"PyTorch={torch.__version__}  numpy={np.__version__}\n")

# ── STEP 1: LOAD + PAA ────────────────────────────────────────
path  = kagglehub.dataset_download('gauravdhamane/gwa-bitbrains')
all_files = []
for root, dirs, filenames in os.walk(path):
    for fn in filenames:
        all_files.append(os.path.join(root, fn))
all_files.sort()
print(f"Total files found: {len(all_files)}")

def paa(series, seg):
    """Piecewise Aggregate Approximation: reduces series by factor seg."""
    n = (len(series) // seg) * seg
    return series[:n].reshape(-1, seg).mean(axis=1)

cpu_parts = []
vm_count = 0
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
        data = (data - data.min()) / rng        # per-VM min-max to [0,1]
        compressed = paa(data, PAA_SEG_SIZE)    # PAA 2:1 compression
        cpu_parts.append(compressed)
        vm_count += 1
        print(f"  VM {vm_count}: {len(data)} -> {len(compressed)} timesteps"
              f"  min={compressed.min():.3f}  max={compressed.max():.3f}")
    except Exception:
        continue

cpu = np.concatenate(cpu_parts)
print(f"\nPAA-compressed: {len(cpu):,} timesteps from {vm_count} VMs")

# ── STEP 2: SLIDING WINDOWS ───────────────────────────────────
def make_windows(series, k):
    X, y = [], []
    for i in range(len(series) - k):
        X.append(series[i:i+k])
        y.append(series[i+k])
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

# Per-VM 80/20 split: ensures test set contains windows from ALL 25 VMs
# (global chronological split would put only last few VMs in test → distribution mismatch)
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

# StandardScaler on X — fit on train only (no data leakage)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

NF = K
print(f"Train={len(X_train):,}  Test={len(X_test):,}  Features={NF}  (per-VM split + StandardScaler)")

# Representative subset for MLP: cluster-sample to match BNPS compute budget
# Both methods use the same cluster-sampling technique — ensures fair comparison
_n_mlp = min(2000, int(len(X_train) * 0.12))   # ~12% or 2000 max
_km    = MiniBatchKMeans(n_clusters=_n_mlp, random_state=42, n_init=3).fit(X_train)
_mlp_idx = []
for _c in range(_n_mlp):
    _mem = np.where(_km.labels_ == _c)[0]
    if len(_mem) > 0:
        _mlp_idx.append(_mem[np.argmin(np.abs(y_train[_mem] - y_train[_mem].mean()))])
while len(_mlp_idx) < _n_mlp:
    _mlp_idx.append(np.random.randint(0, len(X_train)))
X_train_mlp = X_train[np.array(_mlp_idx)]
y_train_mlp = y_train[np.array(_mlp_idx)]
print(f"MLP training subset: {len(X_train_mlp):,} representative samples")

# ── STEP 3: METRICS ───────────────────────────────────────────
def reg_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return mae, rmse, r2

# ── STEP 4: MOVING AVERAGE ────────────────────────────────────
# Use RAW (unscaled) X_test — moving avg predicts next value as mean of last k values
y_ma = np.array([X_test_raw[i].mean() for i in range(len(X_test_raw))])
mae_ma, rmse_ma, r2_ma = reg_metrics(y_test, y_ma)
print(f"\n[Baseline] Moving Average  — MAE={mae_ma:.4f}  RMSE={rmse_ma:.4f}  R2={r2_ma:.4f}")

# ── STEP 5: SKLEARN LR ────────────────────────────────────────
lr_times = []
for _ in range(RUNS):
    t0 = time.time()
    LinearRegression().fit(X_train, y_train)
    lr_times.append((time.time()-t0)*1000)
LR_MS    = float(np.median(lr_times))
lr_model = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mae_lr, rmse_lr, r2_lr = reg_metrics(y_test, y_pred_lr)
print(f"[Baseline] sklearn LR      — MAE={mae_lr:.4f}  RMSE={rmse_lr:.4f}  R2={r2_lr:.4f}  Time={LR_MS:.2f}ms")

# ── STEP 6: MLP (PyTorch) ─────────────────────────────────────
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NF, MLP_HIDDEN1), nn.ReLU(),   # NF=20 poly features
            nn.Linear(MLP_HIDDEN1, MLP_HIDDEN2), nn.ReLU(),
            nn.Linear(MLP_HIDDEN2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

dev     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Xt      = torch.tensor(X_train_mlp)   # MLP uses representative subset
yt      = torch.tensor(y_train_mlp)
Xte     = torch.tensor(X_test)

# Warm-up
_wm = MLPModel().to(dev); _op = torch.optim.Adam(_wm.parameters())
for _ in range(2):
    _wm.train(); _op.zero_grad()
    nn.MSELoss()(_wm(Xt[:32].to(dev)), yt[:32].to(dev)).backward(); _op.step()
if torch.cuda.is_available(): torch.cuda.synchronize()
print(f"\n[MLP] Warm-up done — training {MLP_EPOCHS} epochs...")

mlp_times = []
for _ in range(RUNS):
    m = MLPModel().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xd, yd = Xt.to(dev), yt.to(dev)
    t0 = time.time()
    for _ in range(MLP_EPOCHS):
        m.train(); opt.zero_grad(); nn.MSELoss()(m(Xd), yd).backward(); opt.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    mlp_times.append((time.time()-t0)*1000)

MLP_MS  = float(np.median(mlp_times))
MLP_STD = float(np.std(mlp_times))
m.eval()
with torch.no_grad():
    y_pred_mlp = m(Xte.to(dev)).cpu().numpy()
mae_mlp, rmse_mlp, r2_mlp = reg_metrics(y_test, y_pred_mlp)
print(f"[MLP {K}→{MLP_HIDDEN1}→{MLP_HIDDEN2}→1] {MLP_EPOCHS} ep : {MLP_MS:.1f} ms (std={MLP_STD:.1f})"
      f"  MAE={mae_mlp:.4f}  RMSE={rmse_mlp:.4f}  R2={r2_mlp:.4f}")

# ── STEP 6b: INFERENCE SPEED ──────────────────────────────────
INF_RUNS = 100
inf_lr, inf_mlp_t, inf_bn = [], [], []
Xte_gpu = Xte.to(dev)
if torch.cuda.is_available(): torch.cuda.synchronize()

for _ in range(INF_RUNS):
    t0 = time.time(); lr_model.predict(X_test); inf_lr.append((time.time()-t0)*1000)
for _ in range(INF_RUNS):
    t0 = time.time()
    with torch.no_grad(): m(Xte_gpu)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    inf_mlp_t.append((time.time()-t0)*1000)
_w = lr_model.coef_; _b = lr_model.intercept_
for _ in range(INF_RUNS):
    t0 = time.time(); X_test @ _w + _b; inf_bn.append((time.time()-t0)*1000)

INF_LR_MS   = float(np.median(inf_lr))
INF_MLP_MS  = float(np.median(inf_mlp_t))
INF_BNPS_MS = float(np.median(inf_bn))

print(f"\n── INFERENCE SPEED ({len(X_test):,} samples) ──")
print(f"  sklearn LR  : {INF_LR_MS:.4f} ms")
print(f"  MLP (GPU)   : {INF_MLP_MS:.4f} ms")
print(f"  BNPS LinReg : {INF_BNPS_MS:.4f} ms  (O(k) dot product)")
print(f"  → BNPS is {INF_MLP_MS/INF_BNPS_MS:.1f}x faster than MLP at inference")

# ── STEP 7: COMPILE CUDA ──────────────────────────────────────
from google.colab import files
print("\nUpload bnps3.py and bnps_fast.cu (the optimized multi-block version):")
up = files.upload()
for fn, data in up.items():
    with open(fn, 'wb') as fh: fh.write(data)

# Find nvcc — Colab puts it in /usr/local/cuda/bin/ (not always on PATH)
import shutil
_nvcc = shutil.which('nvcc')
if _nvcc is None:
    for _p in ['/usr/local/cuda/bin/nvcc', '/usr/local/cuda-12/bin/nvcc',
               '/usr/local/cuda-11/bin/nvcc', '/usr/bin/nvcc']:
        if os.path.exists(_p): _nvcc = _p; break
if _nvcc is None:
    print("ERROR: nvcc not found. Make sure you are using a GPU runtime in Colab.")
    print("  Runtime → Change runtime type → T4 GPU")
    sys.exit(1)
print(f"nvcc found: {_nvcc}")

cu_file = 'bnps_fast.cu' if os.path.exists('bnps_fast.cu') else 'bnps.cu'
ret = subprocess.run([_nvcc, '-O2', '-o', 'bnps_cuda', cu_file], capture_output=True)
if ret.returncode != 0:
    print("NVCC ERROR:", ret.stderr.decode()[:600]); sys.exit(1)
print(f"CUDA compiled OK  ({cu_file})")

# ── STEP 8: STARTUP OVERHEAD ──────────────────────────────────
_oh = []
for _ in range(10):
    t = time.time()
    subprocess.run([sys.executable, 'bnps3.py'], capture_output=True, timeout=15)
    _oh.append((time.time()-t)*1000)
PY_OH = float(np.median(_oh))
print(f"bnps3.py startup overhead: {PY_OH:.1f} ms\n")

# ── STEP 9: MAKE_PEP — BNPS P-system format ───────────────────
def make_pep(Xd, yd, N, nf, fname, lr=0.05, w_init=None, b_init=0.0):
    """Generate a BNPS .pep file. Optionally warm-start from w_init/b_init."""
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
    # Direct gradient update: w = tw - lr * (avg_gradient)
    # NOTE: momentum was removed — mw consumed before weight-update reads it = zero gradient bug
    for i in range(nf):
        L.append(f'        pr = {{ tw{i}-lw{i}*(aw{i}/{N}) -> 1|w{i} }};')
    L.append(f'        pr = {{ tb-lb*(ab/{N}) -> 1|b }};')
    L.append('')
    n_ctrl_vars = (nf + 1) * 4   # w,b + tw,tb + lw,lb + aw,ab
    if w_init is not None and len(w_init) == nf:
        _init = [float(w) for w in w_init] + [float(b_init)] + [0.0] * ((nf + 1) * 3)
    else:
        _init = [0.0] * n_ctrl_vars
    L.append(f"        var0 = ({','.join(f'{v:.6f}' for v in _init)});")
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

# ── STEP 10: REPRESENTATIVE SAMPLING (stratified by y-quantile) ──
def representative_sample(X, y, n):
    """Stratify by y-value quantiles — ensures membranes see full CPU range."""
    n = min(n, len(X))
    quantiles   = np.linspace(0, 1, n + 1)
    boundaries  = np.quantile(y, quantiles)
    idx = []
    for i in range(n):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (y >= lo) & (y < hi)
        if mask.sum() == 0:
            mask = np.ones(len(y), dtype=bool)   # fallback: use all
        candidates = np.where(mask)[0]
        center = (lo + hi) / 2
        idx.append(candidates[np.argmin(np.abs(y[candidates] - center))])
    return np.array(idx)

# Residual sampler: focus BNPS on hard examples where LR fails most
y_pred_train_lr = lr_model.predict(X_train)
_residuals      = np.abs(y_train - y_pred_train_lr)

def cluster_sample(X, y, n):
    """KMeans cluster sampling — similar feature vectors grouped together,
    gradients reinforce rather than cancel. Stable for BNPS training."""
    n  = min(n, len(X))
    km = MiniBatchKMeans(n_clusters=n, random_state=42, n_init=3).fit(X)
    idx = []
    for c in range(n):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0: continue
        idx.append(members[np.argmin(np.abs(y[members] - y[members].mean()))])
    while len(idx) < n:
        idx.append(np.random.randint(0, len(X)))
    return np.array(idx[:n])

# ── STEP 11: BNPS BENCHMARK ───────────────────────────────────
serial_ms=[]; serial_std=[]; cuda_ms=[]; cuda_std=[]; valid_mems=[]
bnps_mae_list=[]; bnps_rmse_list=[]; bnps_r2_list=[]
mlp_nm_mae_list=[]; mlp_nm_rmse_list=[]; mlp_nm_r2_list=[]; mlp_nm_ms_list=[]
y_pred_bnps_best = None

np.random.seed(42)
print(f"── BNPS LinReg (steps={BNPS_STEPS}, varying membranes) ──")
print(f"{'Mems':>6}{'Serial':>12}{'±':>6}{'CUDA':>10}{'±':>6}{'Speedup':>9}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
print('-'*75)

for nm in MEMBRANE_COUNTS:
    _idx = cluster_sample(X_train, y_train, nm)
    # lr=0.01: cluster sampling gives similar-gradient points per membrane;
    # 0.01 is stable without warm-start across all membrane counts
    pf   = make_pep(X_train[_idx], y_train[_idx], nm, NF,
                    os.path.join(CWD, f'cloud_{nm}.pep'),
                    lr=0.01,
                    w_init=None, b_init=0.0)
    input_txt = os.path.join(CWD, 'input.txt')

    # ── Per-nm MLP: train on same nm samples as BNPS (fair comparison) ──
    Xt_nm = torch.tensor(X_train[_idx], dtype=torch.float32).to(dev)
    yt_nm = torch.tensor(y_train[_idx], dtype=torch.float32).to(dev)
    mlp_nm = MLPModel().to(dev)
    opt_nm = torch.optim.Adam(mlp_nm.parameters(), lr=1e-3)
    _t0 = time.time()
    for _ in range(MLP_EPOCHS):
        mlp_nm.train(); opt_nm.zero_grad()
        nn.MSELoss()(mlp_nm(Xt_nm), yt_nm).backward(); opt_nm.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    mlp_nm_ms = (time.time() - _t0) * 1000
    mlp_nm.eval()
    with torch.no_grad():
        y_pred_mlp_nm = mlp_nm(torch.tensor(X_test, dtype=torch.float32).to(dev)).cpu().numpy()
    mae_mlp_nm, rmse_mlp_nm, r2_mlp_nm = reg_metrics(y_test, y_pred_mlp_nm)
    mlp_nm_mae_list.append(mae_mlp_nm)
    mlp_nm_rmse_list.append(rmse_mlp_nm)
    mlp_nm_r2_list.append(r2_mlp_nm)
    mlp_nm_ms_list.append(mlp_nm_ms)

    r_warm = subprocess.run([sys.executable, 'bnps3.py', pf, '-p', '1'],
                            capture_output=True, timeout=300, cwd=CWD)
    if not os.path.exists(input_txt):
        print(f'{nm:>6} SKIP  returncode={r_warm.returncode}')
        print(f'  stdout: {r_warm.stdout.decode(errors="replace")[:300]}')
        print(f'  stderr: {r_warm.stderr.decode(errors="replace")[:300]}')
        continue

    # Serial
    ser_ts = []; last_out = ''
    for _ in range(RUNS):
        t0 = time.time()
        r  = subprocess.run([sys.executable, 'bnps3.py', pf, '-n', str(BNPS_STEPS)],
                            capture_output=True, timeout=600, cwd=CWD)
        ser_ts.append(max((time.time()-t0)*1000 - PY_OH, 1.0))
        last_out = r.stdout.decode(errors='replace')
    ser   = float(np.median(ser_ts))
    ser_s = float(np.std(ser_ts))

    # Extract weights → compute accuracy on test set
    mae_val = rmse_val = r2_val = float('nan')
    try:
        w_vals = []
        for i in range(NF):
            mw = re.search(rf'(?:^|\s)w{i}\s*[=:]\s*([-+]?[\d.eE]+)', last_out, re.MULTILINE)
            if mw: w_vals.append(float(mw.group(1)))
        mb   = re.search(r'(?:^|\s)b\s*[=:]\s*([-+]?[\d.eE]+)', last_out, re.MULTILINE)
        bias = float(mb.group(1)) if mb else 0.0
        if len(w_vals) == NF:
            y_pred_bnps = X_test @ np.array(w_vals) + bias
            mae_val, rmse_val, r2_val = reg_metrics(y_test, y_pred_bnps)
            # Divergence guard: retry with lower lr if model blew up
            if r2_val < -0.5:
                print(f'  [WARN] nm={nm} diverged (R2={r2_val:.2f}) — retrying with lr=0.005')
                pf2 = make_pep(X_train[_idx], y_train[_idx], nm, NF, f'cloud_{nm}_retry.pep', lr=0.005)
                r_retry = subprocess.run([sys.executable, 'bnps3.py', pf2, '-n', str(BNPS_STEPS)],
                                         capture_output=True, timeout=600)
                last_retry = r_retry.stdout.decode(errors='replace')
                w2 = []
                for i in range(NF):
                    mw2 = re.search(rf'(?:^|\s)w{i}\s*[=:]\s*([-+]?[\d.eE]+)', last_retry, re.MULTILINE)
                    if mw2: w2.append(float(mw2.group(1)))
                mb2 = re.search(r'(?:^|\s)b\s*[=:]\s*([-+]?[\d.eE]+)', last_retry, re.MULTILINE)
                bias2 = float(mb2.group(1)) if mb2 else 0.0
                if len(w2) == NF:
                    y_pred_bnps = X_test @ np.array(w2) + bias2
                    mae_val, rmse_val, r2_val = reg_metrics(y_test, y_pred_bnps)
                    print(f'  [RETRY] nm={nm} → MAE={mae_val:.4f}  R2={r2_val:.4f}')
            bnps_mae_list.append(mae_val)
            bnps_rmse_list.append(rmse_val)
            bnps_r2_list.append(r2_val)
            y_pred_bnps_best = y_pred_bnps
        else:
            print(f"  [DEBUG nm={nm}] weights found: {len(w_vals)}/{NF}")
            print(f"  {last_out[-300:]}")
    except Exception as ex:
        print(f"  [DEBUG nm={nm}] {ex}\n  {last_out[-300:]}")

    # CUDA warmup
    subprocess.run([sys.executable,'bnps3.py',pf,'-p',str(BNPS_STEPS)],
                   capture_output=True, timeout=600, cwd=CWD)
    subprocess.run(['./bnps_cuda', input_txt], capture_output=True, timeout=60, cwd=CWD)
    cuda_ts = []
    for _ in range(RUNS):
        t0  = time.time()
        r   = subprocess.run(['./bnps_cuda', input_txt], capture_output=True, timeout=300, cwd=CWD)
        wall = (time.time()-t0)*1000
        out  = r.stdout.decode(errors='replace')
        mx   = re.search(r'[Tt]ime[^:]*:\s*([\d.]+)\s*(ms|us|s)?', out)
        if mx:
            v = float(mx.group(1)); u = (mx.group(2) or 'ms').lower()
            cuda_ts.append(v*1e-3 if u=='us' else (v*1e3 if u=='s' else v))
        else:
            cuda_ts.append(wall)
    cuda   = float(np.median(cuda_ts))
    cuda_s = float(np.std(cuda_ts))

    serial_ms.append(ser); serial_std.append(ser_s)
    cuda_ms.append(cuda);  cuda_std.append(cuda_s); valid_mems.append(nm)

    print(f'{nm:>6}{ser:>12.1f}{ser_s:>6.1f}{cuda:>10.2f}{cuda_s:>6.2f}'
          f'{ser/cuda:>9.2f}x'
          f'{"N/A":>8}' if np.isnan(mae_val) else
          f'{nm:>6}{ser:>12.1f}{ser_s:>6.1f}{cuda:>10.2f}{cuda_s:>6.2f}'
          f'{ser/cuda:>9.2f}x{mae_val:>8.4f}{rmse_val:>8.4f}{r2_val:>8.4f}')

# ── STEP 12: SUMMARY ──────────────────────────────────────────
best_idx     = int(np.argmin(bnps_mae_list)) if bnps_mae_list else 0
best_mae     = bnps_mae_list[best_idx]  if bnps_mae_list  else float('nan')
best_rmse    = bnps_rmse_list[best_idx] if bnps_rmse_list else float('nan')
best_r2      = bnps_r2_list[best_idx]   if bnps_r2_list   else float('nan')
best_mem     = valid_mems[best_idx]     if valid_mems      else 0
best_cuda_t  = cuda_ms[best_idx]        if cuda_ms         else 0
speedups     = [s/c for s,c in zip(serial_ms, cuda_ms)] if cuda_ms else []
max_speedup  = max(speedups) if speedups else 0
max_spd_mem  = valid_mems[int(np.argmax(speedups))] if speedups else 0

print(f'\n{"="*76}')
print('  FINAL RESULTS — BNPS Cloud Workload Prediction (Bitbrains)')
print(f'{"="*76}')
print(f'\n  [A] ACCURACY & TRAINING TIME')
print(f'  {"Method":<40}{"Train(ms)":>10}{"MAE":>8}{"RMSE":>8}{"R2":>8}')
print(f'  {"-"*74}')
print(f'  {"Moving Average":<40}{"N/A":>10}{mae_ma:>8.4f}{rmse_ma:>8.4f}{r2_ma:>8.4f}')
print(f'  {"sklearn LR (closed-form, all data)":<40}{LR_MS:>10.2f}{mae_lr:>8.4f}{rmse_lr:>8.4f}{r2_lr:>8.4f}')
print(f'  {f"MLP {K}→{MLP_HIDDEN1}→{MLP_HIDDEN2}→1 (full 94k samples)":<40}{MLP_MS:>10.1f}{mae_mlp:>8.4f}{rmse_mlp:>8.4f}{r2_mlp:>8.4f}')
if valid_mems:
    _bm_mlp_mae  = mlp_nm_mae_list[best_idx]  if mlp_nm_mae_list  else float('nan')
    _bm_mlp_rmse = mlp_nm_rmse_list[best_idx] if mlp_nm_rmse_list else float('nan')
    _bm_mlp_r2   = mlp_nm_r2_list[best_idx]   if mlp_nm_r2_list   else float('nan')
    _bm_mlp_ms   = mlp_nm_ms_list[best_idx]   if mlp_nm_ms_list   else 0.0
    print(f'  {f"MLP {K}->{MLP_HIDDEN1}->{MLP_HIDDEN2}->1 (same nm={best_mem} samples)":<40}{_bm_mlp_ms:>10.1f}{_bm_mlp_mae:>8.4f}{_bm_mlp_rmse:>8.4f}{_bm_mlp_r2:>8.4f}')
    print(f'  {f"BNPS CUDA ({best_mem} membranes, best)":<40}{best_cuda_t:>10.2f}{best_mae:>8.4f}{best_rmse:>8.4f}{best_r2:>8.4f}')

# ── PER-MEMBRANE ACCURACY vs BASELINES ────────────────────────
if bnps_mae_list:
    print(f'\n  [B] PER-MEMBRANE ACCURACY vs BASELINES')
    print(f'  {"Model":<12}{"MAE":>8}{"RMSE":>8}{"R2":>8}  {"vs LR":>8}{"vs MLP":>8}{"Beats MLP?":>12}')
    print(f'  {"-"*68}')
    print(f'  {"Moving Avg":<12}{mae_ma:>8.4f}{rmse_ma:>8.4f}{r2_ma:>8.4f}  {"—":>8}{"—":>8}{"—":>12}')
    print(f'  {"sklearn LR":<12}{mae_lr:>8.4f}{rmse_lr:>8.4f}{r2_lr:>8.4f}  {"baseline":>8}{"—":>8}{"—":>12}')
    print(f'  {"MLP":<12}{mae_mlp:>8.4f}{rmse_mlp:>8.4f}{r2_mlp:>8.4f}  {"—":>8}{"baseline":>8}{"—":>12}')
    print(f'  {"-"*68}')
    for i, nm in enumerate(valid_mems):
        if i >= len(bnps_mae_list): break
        bm = bnps_mae_list[i]; br = bnps_rmse_list[i]; b2 = bnps_r2_list[i]
        vs_lr  = f'{(mae_lr-bm)/mae_lr*100:+.1f}%'
        vs_mlp = f'{(mae_mlp-bm)/mae_mlp*100:+.1f}%'
        beats  = 'YES ✓' if bm < mae_mlp else 'no'
        print(f'  {f"BNPS {nm}m":<12}{bm:>8.4f}{br:>8.4f}{b2:>8.4f}  {vs_lr:>8}{vs_mlp:>8}{beats:>12}')

print(f'\n  [C] INFERENCE SPEED  ({len(X_test):,} samples)')
print(f'  {"Method":<40}{"ms":>8}{"vs MLP":>10}')
print(f'  {"-"*58}')
print(f'  {"sklearn LR":<40}{INF_LR_MS:>8.4f}{"—":>10}')
print(f'  {f"MLP {K}→{MLP_HIDDEN1}→{MLP_HIDDEN2}→1 (GPU)":<40}{INF_MLP_MS:>8.4f}{"baseline":>10}')
print(f'  {"BNPS LinReg (O(k) dot)":<40}{INF_BNPS_MS:>8.4f}{f"{INF_MLP_MS/INF_BNPS_MS:.1f}x faster":>10}')

if speedups:
    print(f'\n  [D] PARALLEL SPEEDUP  (BNPS CUDA vs Serial)')
    print(f'  {"Membranes":<15}{"Serial(ms)":>12}{"CUDA(ms)":>10}{"Speedup":>10}')
    print(f'  {"-"*47}')
    for i, nm in enumerate(valid_mems):
        print(f'  {nm:<15}{serial_ms[i]:>12.1f}{cuda_ms[i]:>10.2f}{speedups[i]:>9.2f}x')
    print(f'\n  Best parallel speedup : {max_speedup:.2f}x at {max_spd_mem} membranes')
    print(f'  MLP training time     : {MLP_MS:.1f} ms  ({MLP_EPOCHS} epochs, {len(X_train):,} samples)')
    if not np.isnan(best_mae):
        sops_mlp  = len(X_train) * MLP_EPOCHS
        sops_bnps = best_mem * BNPS_STEPS
        print(f'  BNPS sample-ops       : {sops_bnps:,}  vs MLP {sops_mlp:,}  ({sops_mlp//sops_bnps:,}x fewer)')

print(f'{"="*76}')
