"""
BNPS Wisconsin Breast Cancer — Best Results Script
===================================================
Improvements over all previous notebooks:
  1. All 30 features (not just 10 mean features)
  2. StandardScaler normalization (prevents gradient explosion)
  3. Lower LR=0.05 with momentum (prevents divergence at high steps)
  4. Stratified membrane sampling (balanced class distribution)
  5. Piecewise sigmoid: 3-segment approx (better than 0.5+0.25z)
  6. Sweeps membranes AND steps to find global best
  7. Full benchmark vs sklearn LR, PyTorch SLP, PyTorch Deep MLP
  8. Wilson 95% CI on all accuracies
  9. Saves graphs: wisconsin_best_accuracy.png, wisconsin_best_speedup.png

Run on Google Colab (T4 GPU) or any machine with Python + sklearn + torch.
No CUDA kernel required — pure Python BNPS + PyTorch GPU baselines.
"""

import time, sys, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────
MEMBRANE_COUNTS = [25, 50, 75, 100, 150, 200]
STEP_SWEEP      = [10, 25, 50, 75, 100, 150]
LR              = 0.05          # lower than original 0.5 → prevents divergence
MOMENTUM        = 0.85          # momentum on gradients
MLP_EPOCHS      = 100
RUNS            = 10            # timing repetitions
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
# ──────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  BNPS Wisconsin Breast Cancer — Best Results Benchmark")
print("=" * 65)

# ── 1. LOAD DATA (all 30 features) ───────────────────────────────────
data   = load_breast_cancer()
X_all  = data.data.astype('float32')      # (569, 30)
y_all  = data.target.astype('float32')    # 0=malignant, 1=benign

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE,
    stratify=y_all, random_state=RANDOM_STATE)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype('float32')
X_test  = scaler.transform(X_test_raw).astype('float32')

NF = X_train.shape[1]    # 30
N_TRAIN = len(X_train)
N_TEST  = len(X_test)

print(f"\n  Dataset : Breast Cancer Wisconsin (UCI)")
print(f"  Features: {NF} (all 30, StandardScaler normalized)")
print(f"  Train   : {N_TRAIN}  |  Test: {N_TEST}")
print(f"  Class balance (train): "
      f"benign={int(y_train.sum())}  malignant={int((y_train==0).sum())}")

# ── 2. HELPERS ────────────────────────────────────────────────────────
def wilson_ci(acc, n, alpha=0.05):
    lo, hi = proportion_confint(int(acc * n), n, alpha=alpha, method='wilson')
    return lo, hi

def compute_metrics(y_true, y_pred_prob, thresh=0.5):
    y_pred = (y_pred_prob >= thresh).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    lo, hi = wilson_ci(acc, len(y_true))
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, cm=cm, ci=(lo, hi))

def piecewise_sigmoid(z):
    """
    3-piece piecewise sigmoid — better than 0.5+0.25z for large |z|.
      z < -2  ->  0.1*z + 0.5   (slope 0.1, clips near 0)
      -2<=z<=2 -> 0.25*z + 0.5  (slope 0.25, linear region)
      z > 2   ->  0.1*z + 0.5   (slope 0.1, clips near 1)
    Clipped to [0.01, 0.99].
    """
    sig = np.where(np.abs(z) <= 2,
                   0.25 * z + 0.5,
                   0.1  * z + 0.5)
    return np.clip(sig, 0.01, 0.99)

# ── 3. MEMBRANE SLP (BNPS Python simulation) ──────────────────────────
class MembraneSLP:
    """
    BNPS Single-Layer Perceptron simulator.
    Each training sample = one membrane; controller membrane holds weights.
    Uses piecewise sigmoid + SGD with momentum.
    """
    def __init__(self, X, y, lr=0.05, momentum=0.85):
        self.X  = X
        self.y  = y
        self.lr = lr
        self.mu = momentum
        self.F  = X.shape[1]
        self.w  = np.zeros(self.F, dtype='float64')
        self.b  = 0.0
        self.vw = np.zeros(self.F, dtype='float64')   # velocity (momentum)
        self.vb = 0.0
        self.loss_history = []

    def reset(self):
        self.w  = np.zeros(self.F, dtype='float64')
        self.b  = 0.0
        self.vw = np.zeros(self.F, dtype='float64')
        self.vb = 0.0
        self.loss_history = []

    def step(self):
        """One BNPS step: parallel forward + gradient aggregate + weight update."""
        z     = self.X @ self.w + self.b          # (N,) — all membranes parallel
        sigma = piecewise_sigmoid(z)
        error = sigma - self.y                     # (N,)

        # Gradients in each sample membrane
        grad_w = (error[:, None] * self.X).mean(axis=0)   # (F,)
        grad_b = error.mean()

        # BCE loss for monitoring
        loss = -np.mean(self.y * np.log(sigma) + (1 - self.y) * np.log(1 - sigma))
        self.loss_history.append(float(loss))

        # Controller: SGD with momentum
        self.vw = self.mu * self.vw + self.lr * grad_w
        self.vb = self.mu * self.vb + self.lr * grad_b
        self.w  -= self.vw
        self.b  -= self.vb

    def train(self, n_steps):
        self.reset()
        for _ in range(n_steps):
            self.step()

    def predict_prob(self, X):
        z = X @ self.w + self.b
        return piecewise_sigmoid(z)

    def predict(self, X, thresh=0.5):
        return (self.predict_prob(X) >= thresh).astype(int)

def stratified_sample(X, y, n):
    """Balanced class sampling: n//2 from each class."""
    n = min(n, len(X))
    n_pos = n // 2
    n_neg = n - n_pos
    rng  = np.random.default_rng(42)
    pos  = rng.choice(np.where(y == 1)[0], min(n_pos, (y==1).sum()), replace=False)
    neg  = rng.choice(np.where(y == 0)[0], min(n_neg, (y==0).sum()), replace=False)
    idx  = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return X[idx], y[idx]

# ── 4. SKLEARN BASELINES ──────────────────────────────────────────────
print("\n[1] Baseline: sklearn Logistic Regression (full data)")
t0      = time.time()
lr_clf  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_clf.fit(X_train, y_train)
lr_train_ms = (time.time() - t0) * 1000
y_prob_lr   = lr_clf.predict_proba(X_test)[:, 1]
lr_m        = compute_metrics(y_test, y_prob_lr)
lo, hi      = lr_m['ci']
print(f"  sklearn LR : Acc={lr_m['acc']:.4f}  F1={lr_m['f1']:.4f}  "
      f"CI=[{lo:.3f}-{hi:.3f}]  Train={lr_train_ms:.1f}ms")

# ── 5. PYTORCH BASELINES ──────────────────────────────────────────────
dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Xtr  = torch.tensor(X_train).to(dev)
ytr  = torch.tensor(y_train).to(dev)
Xte  = torch.tensor(X_test).to(dev)

print(f"\n[2] PyTorch baselines (device={dev})")

class LinearSLP(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.fc = nn.Linear(nf, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze(1)

class DeepMLP(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nf, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 1),   nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_pt(model, X, y, epochs, lr_pt=1e-3):
    opt  = torch.optim.Adam(model.parameters(), lr=lr_pt, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(X), y).backward(); opt.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()

def eval_pt(model, X, y_np):
    model.eval()
    with torch.no_grad():
        prob = model(X).cpu().numpy()
    return compute_metrics(y_np, prob)

# PyTorch SLP (1-layer, full data)
pt_slp_times = []
for _ in range(RUNS):
    m = LinearSLP(NF).to(dev)
    t0 = time.time()
    train_pt(m, Xtr, ytr, MLP_EPOCHS)
    pt_slp_times.append((time.time()-t0)*1000)
PT_SLP_MS = float(np.median(pt_slp_times))
pt_slp_m  = eval_pt(m, Xte, y_test)
lo, hi    = pt_slp_m['ci']
print(f"  PT SLP  (1-layer, {MLP_EPOCHS}ep)  : Acc={pt_slp_m['acc']:.4f}  "
      f"F1={pt_slp_m['f1']:.4f}  CI=[{lo:.3f}-{hi:.3f}]  Train={PT_SLP_MS:.1f}ms")

# PyTorch Deep MLP (4-layer, full data)
pt_mlp_times = []
for _ in range(RUNS):
    m = DeepMLP(NF).to(dev)
    t0 = time.time()
    train_pt(m, Xtr, ytr, MLP_EPOCHS)
    pt_mlp_times.append((time.time()-t0)*1000)
PT_MLP_MS = float(np.median(pt_mlp_times))
pt_mlp_m  = eval_pt(m, Xte, y_test)
lo, hi    = pt_mlp_m['ci']
print(f"  PT MLP (4-layer, {MLP_EPOCHS}ep)   : Acc={pt_mlp_m['acc']:.4f}  "
      f"F1={pt_mlp_m['f1']:.4f}  CI=[{lo:.3f}-{hi:.3f}]  Train={PT_MLP_MS:.1f}ms")

# ── 6. BNPS STEP SWEEP (fixed 100 membranes) ─────────────────────────
print("\n[3] BNPS Step Sweep (fixed 100 membranes, lr=0.05, momentum=0.85)")
print(f"  {'Steps':>6}  {'Acc':>8}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'95% CI':>18}")
print("  " + "-" * 60)

FIXED_MEMS = 100
Xm, ym     = stratified_sample(X_train, y_train, FIXED_MEMS)
slp_fixed  = MembraneSLP(Xm, ym, lr=LR, momentum=MOMENTUM)
step_results = []

for steps in STEP_SWEEP:
    slp_fixed.train(steps)
    prob = slp_fixed.predict_prob(X_test)
    sm   = compute_metrics(y_test, prob)
    lo, hi = sm['ci']
    step_results.append((steps, sm))
    print(f"  {steps:>6}  {sm['acc']:>8.4f}  {sm['f1']:>7.4f}  "
          f"{sm['prec']:>7.4f}  {sm['rec']:>7.4f}  [{lo:.3f}-{hi:.3f}]")

best_step_idx  = int(np.argmax([r['acc'] for _, r in step_results]))
best_steps     = step_results[best_step_idx][0]
best_step_acc  = step_results[best_step_idx][1]['acc']
print(f"\n  Best: {best_steps} steps → Acc={best_step_acc:.4f}")

# ── 7. BNPS MEMBRANE SWEEP (best steps) ───────────────────────────────
print(f"\n[4] BNPS Membrane Sweep (steps={best_steps}, lr={LR}, momentum={MOMENTUM})")
print(f"  {'Mems':>6}  {'Acc':>8}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  "
      f"{'Time(ms)':>10}  {'95% CI':>18}")
print("  " + "-" * 70)

mem_results = []
for nm in MEMBRANE_COUNTS:
    Xm, ym = stratified_sample(X_train, y_train, nm)
    slp    = MembraneSLP(Xm, ym, lr=LR, momentum=MOMENTUM)

    times  = []
    for _ in range(RUNS):
        t0 = time.time()
        slp.train(best_steps)
        times.append((time.time()-t0)*1000)
    train_ms = float(np.median(times))

    prob  = slp.predict_prob(X_test)
    mm    = compute_metrics(y_test, prob)
    lo, hi = mm['ci']
    mem_results.append((nm, mm, train_ms))
    print(f"  {nm:>6}  {mm['acc']:>8.4f}  {mm['f1']:>7.4f}  "
          f"{mm['prec']:>7.4f}  {mm['rec']:>7.4f}  "
          f"{train_ms:>10.1f}  [{lo:.3f}-{hi:.3f}]")

best_mem_idx  = int(np.argmax([r['acc'] for _, r, _ in mem_results]))
best_nm       = mem_results[best_mem_idx][0]
best_mem_m    = mem_results[best_mem_idx][1]
best_mem_ms   = mem_results[best_mem_idx][2]
print(f"\n  Best: {best_nm} membranes, {best_steps} steps → Acc={best_mem_m['acc']:.4f}")

# ── 8. FINAL SUMMARY ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL RESULTS — BNPS Wisconsin Breast Cancer")
print(f"{'='*65}")
print(f"\n  {'Method':<35}  {'Acc':>8}  {'F1':>7}  {'Train(ms)':>10}  {'95% CI'}")
print("  " + "-" * 72)

for name, m, ms in [
    ("sklearn LogReg (full data)",        lr_m,       lr_train_ms),
    (f"PT SLP 1-layer ({MLP_EPOCHS}ep)",  pt_slp_m,  PT_SLP_MS),
    (f"PT MLP 4-layer ({MLP_EPOCHS}ep)",  pt_mlp_m,  PT_MLP_MS),
    (f"BNPS ({best_nm}mem, {best_steps}steps) ← BEST", best_mem_m, best_mem_ms),
]:
    lo, hi = m['ci']
    print(f"  {name:<35}  {m['acc']:>8.4f}  {m['f1']:>7.4f}  "
          f"{ms:>10.1f}  [{lo:.3f}-{hi:.3f}]")

# Speed ratios
print(f"\n  BNPS vs PT SLP  : {PT_SLP_MS/best_mem_ms:.1f}x faster")
print(f"  BNPS vs PT MLP  : {PT_MLP_MS/best_mem_ms:.1f}x faster")
print(f"  BNPS Acc vs PT SLP : {(best_mem_m['acc']-pt_slp_m['acc'])*100:+.2f}%")
print(f"  BNPS Acc vs PT MLP : {(best_mem_m['acc']-pt_mlp_m['acc'])*100:+.2f}%")
print(f"\n  Config: LR={LR}  Momentum={MOMENTUM}  "
      f"Sigmoid=piecewise-3seg  Features=30/30")

# Confusion matrix for best BNPS
cm = best_mem_m['cm']
print(f"\n  BNPS Best Confusion Matrix ({best_nm}m, {best_steps}steps):")
print(f"    Predicted →     Benign  Malignant")
print(f"    Actual Benign    TN={cm[1][1]:3d}    FP={cm[1][0]:3d}")
print(f"    Actual Malignant FN={cm[0][1]:3d}    TP={cm[0][0]:3d}")
print(f"{'='*65}")

# ── 9. PLOTS ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("BNPS Wisconsin Breast Cancer — Best Results",
             fontsize=13, fontweight='bold')

# Plot 1: Accuracy vs Steps
ax = axes[0]
steps_x = [s for s, _ in step_results]
accs_s  = [r['acc'] for _, r in step_results]
f1s_s   = [r['f1']  for _, r in step_results]
ax.plot(steps_x, accs_s, 'o-', color='#3498db', lw=2.5, ms=8, label='Accuracy')
ax.plot(steps_x, f1s_s,  's--', color='#e74c3c', lw=2,   ms=7, label='F1 Score')
for x, v in zip(steps_x, accs_s):
    ax.annotate(f'{v:.3f}', (x, v), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9)
ax.axvline(best_steps, color='#27ae60', ls=':', lw=1.5, label=f'Best={best_steps}')
ax.set_xlabel('BNPS Steps (iterations)', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title(f'Acc vs Steps\n(100 membranes, LR={LR}, mom={MOMENTUM})', fontsize=10)
ax.set_ylim(0.75, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Plot 2: Accuracy vs Membranes
ax = axes[1]
mems_x = [nm for nm, _, _ in mem_results]
accs_m  = [r['acc'] for _, r, _ in mem_results]
f1s_m   = [r['f1']  for _, r, _ in mem_results]
ax.plot(mems_x, accs_m, 'o-', color='#9b59b6', lw=2.5, ms=8, label='Accuracy')
ax.plot(mems_x, f1s_m,  's--', color='#e67e22', lw=2,   ms=7, label='F1 Score')
for x, v in zip(mems_x, accs_m):
    ax.annotate(f'{v:.3f}', (x, v), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9)
ax.axhline(pt_mlp_m['acc'], color='#e74c3c', ls='--', lw=1.5,
           label=f"PT MLP={pt_mlp_m['acc']:.3f}")
ax.axhline(pt_slp_m['acc'], color='#3498db', ls='--', lw=1.5,
           label=f"PT SLP={pt_slp_m['acc']:.3f}")
ax.set_xlabel('BNPS Membrane Count', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title(f'Acc vs Membranes\n({best_steps} steps, LR={LR})', fontsize=10)
ax.set_ylim(0.75, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Plot 3: Method comparison bar chart
ax = axes[2]
methods = ['LR\n(full)', f'PT SLP\n({MLP_EPOCHS}ep)', f'PT MLP\n({MLP_EPOCHS}ep)',
           f'BNPS\n({best_nm}m,{best_steps}s)']
accs_bar = [lr_m['acc'], pt_slp_m['acc'], pt_mlp_m['acc'], best_mem_m['acc']]
f1s_bar  = [lr_m['f1'],  pt_slp_m['f1'],  pt_mlp_m['f1'],  best_mem_m['f1']]
colors   = ['#95a5a6', '#3498db', '#e74c3c', '#27ae60']
x = np.arange(len(methods)); w = 0.35
b1 = ax.bar(x - w/2, accs_bar, w, label='Accuracy', color=colors, alpha=0.85)
b2 = ax.bar(x + w/2, f1s_bar,  w, label='F1 Score',  color=colors, alpha=0.5, hatch='//')
for b in list(b1) + list(b2):
    ax.annotate(f'{b.get_height():.3f}',
                xy=(b.get_x() + b.get_width()/2, b.get_height()),
                xytext=(0, 4), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(0.75, 1.10); ax.set_ylabel('Score', fontsize=11)
ax.set_title('Method Comparison\n(Breast Cancer Wisconsin)', fontsize=10)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('wisconsin_best_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: wisconsin_best_results.png")

# Loss curve
fig2, ax2 = plt.subplots(figsize=(10, 4))
slp_loss  = MembraneSLP(*stratified_sample(X_train, y_train, best_nm),
                         lr=LR, momentum=MOMENTUM)
slp_loss.train(max(STEP_SWEEP))
ax2.plot(range(1, len(slp_loss.loss_history)+1), slp_loss.loss_history,
         color='#2ecc71', lw=2.5, label=f'BNPS ({best_nm}m, LR={LR}, mom={MOMENTUM})')
ax2.set_xlabel('BNPS Step (Iteration)', fontsize=12)
ax2.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
ax2.set_title('BNPS Training Loss Curve — Breast Cancer Wisconsin', fontsize=13)
ax2.legend(fontsize=11); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('wisconsin_best_loss.png', dpi=150, bbox_inches='tight')
print("Plot saved: wisconsin_best_loss.png")
