"""
wisc_config.py  —  Shared config, data, helpers for Wisconsin benchmark.
Import this in wisc_bnps.py and wisc_baselines.py.

Sweep parameters
----------------
MEMBRANE_COUNTS : [25, 50, 100, 150, 200]
    Number of BNPS membranes = number of training samples used.
    All baselines (PT/TF) are ALSO trained on the SAME N samples
    so the comparison is perfectly fair.

STEPS_SWEEP : [10, 25, 50, 100]
    BNPS steps ≡ training epochs for baselines.
    Every method uses the same number of update iterations.

Dataset
-------
Breast Cancer Wisconsin (UCI), 569 samples, 30 features.
80/20 stratified split → 455 train / 114 test.
StandardScaler fit on train only.
"""

import numpy as np
import os, re, subprocess, sys
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from statsmodels.stats.proportion import proportion_confint

# ── SWEEP CONFIG ──────────────────────────────────────────────────────
MEMBRANE_COUNTS = [25, 50, 100, 150, 200]   # also = N training samples per run
STEPS_SWEEP     = [10, 25, 50, 100]          # BNPS steps = DL epochs
RUNS            = 10                         # timing repetitions (median taken)
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

# BNPS hyper-params
BNPS_LR       = 0.05
BNPS_MOMENTUM = 0.85

# ── DATA ──────────────────────────────────────────────────────────────
def load_data():
    data  = load_breast_cancer()
    X_all = data.data.astype('float32')   # (569, 30)
    y_all = data.target.astype('float32') # 1=benign, 0=malignant

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE, stratify=y_all,
        random_state=RANDOM_STATE)

    sc      = StandardScaler()
    X_train = sc.fit_transform(X_tr_raw).astype('float32')
    X_test  = sc.transform(X_te_raw).astype('float32')
    return X_train, X_test, y_train, y_test, sc

X_train, X_test, y_train, y_test, scaler = load_data()
NF       = X_train.shape[1]   # 30
N_TRAIN  = len(X_train)       # 455
N_TEST   = len(X_test)        # 114

# ── SAMPLING ──────────────────────────────────────────────────────────
def stratified_sample(X, y, n, seed=42):
    """Balanced class sample of size n from (X, y)."""
    n   = min(n, len(X))
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.where(y == 1)[0], n // 2, replace=False)
    neg = rng.choice(np.where(y == 0)[0], n - n // 2, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return X[idx], y[idx]

# ── METRICS ───────────────────────────────────────────────────────────
def metrics(y_true, y_pred_prob, thresh=0.5):
    yp  = (y_pred_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, yp)
    lo, hi = proportion_confint(int(acc * len(y_true)), len(y_true),
                                alpha=0.05, method='wilson')
    return dict(
        acc  = acc,
        prec = precision_score(y_true, yp, zero_division=0),
        rec  = recall_score(y_true, yp, zero_division=0),
        f1   = f1_score(y_true, yp, zero_division=0),
        cm   = confusion_matrix(y_true, yp).tolist(),
        ci   = (lo, hi)
    )

# ── BNPS SIGMOID ──────────────────────────────────────────────────────
def piecewise_sigmoid(z):
    """3-segment piecewise sigmoid. Better than 0.5+0.25z for large |z|."""
    sig = np.where(np.abs(z) <= 2, 0.25*z + 0.5, 0.1*z + 0.5)
    return np.clip(sig, 0.01, 0.99)

# ── BNPS SERIAL SLP ───────────────────────────────────────────────────
class MembraneSLP:
    """
    Python simulation of BNPS SLP.
    Each sample membrane does forward + gradient in parallel.
    Controller membrane aggregates and updates weights.
    Uses piecewise sigmoid + SGD with momentum.
    """
    def __init__(self, X, y, lr=BNPS_LR, momentum=BNPS_MOMENTUM):
        self.X  = X.astype('float64')
        self.y  = y.astype('float64')
        self.lr = lr
        self.mu = momentum
        self.F  = X.shape[1]
        self.reset()

    def reset(self):
        self.w  = np.zeros(self.F)
        self.b  = 0.0
        self.vw = np.zeros(self.F)
        self.vb = 0.0
        self.loss_history = []

    def step(self):
        z     = self.X @ self.w + self.b
        sigma = piecewise_sigmoid(z)
        error = sigma - self.y
        grad_w = (error[:, None] * self.X).mean(axis=0)
        grad_b = error.mean()
        loss   = -np.mean(self.y*np.log(sigma) + (1-self.y)*np.log(1-sigma))
        self.loss_history.append(float(loss))
        self.vw = self.mu * self.vw + self.lr * grad_w
        self.vb = self.mu * self.vb + self.lr * grad_b
        self.w -= self.vw
        self.b -= self.vb

    def train(self, steps):
        self.reset()
        for _ in range(steps):
            self.step()

    def predict_prob(self, X):
        return piecewise_sigmoid(X.astype('float64') @ self.w + self.b)

# ── MAKE PEP (for BNPS CUDA) ──────────────────────────────────────────
def make_pep(Xd, yd, N, nf, fname, lr=0.01):
    """
    Generate BNPS .pep file for logistic SLP.
    Same format as BNPS_Cloud_Workload.py but for classification.
    """
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
    tb = '+'.join(f'1|b_{m}' for m in range(2, N+2)) + '+1|tb'
    L.append(f'        pr = {{ b*{mul} -> {tb} }};')
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
    _init = [0.0] * ((nf + 1) * 4)
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
        za_t = '+'.join(f'1|za{j}_{m}' for j in range(nf+1))
        L.append(f'        pr = {{ z_{m}*{nf+1} -> {za_t} }};')
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
    with open(fname, 'w') as fh:
        fh.write('\n'.join(L))
    return fname

def extract_weights(output_text, nf):
    """Parse w0..wN-1 and b from bnps3.py stdout."""
    w_vals = []
    for i in range(nf):
        m = re.search(rf'(?:^|\s)w{i}\s*[=:]\s*([-+]?[\d.eE]+)', output_text, re.MULTILINE)
        if m: w_vals.append(float(m.group(1)))
    mb   = re.search(r'(?:^|\s)b\s*[=:]\s*([-+]?[\d.eE]+)', output_text, re.MULTILINE)
    bias = float(mb.group(1)) if mb else 0.0
    return np.array(w_vals) if len(w_vals) == nf else None, bias

if __name__ == '__main__':
    print(f"Dataset loaded: {N_TRAIN} train / {N_TEST} test, {NF} features")
    print(f"MEMBRANE_COUNTS : {MEMBRANE_COUNTS}")
    print(f"STEPS_SWEEP     : {STEPS_SWEEP}")
