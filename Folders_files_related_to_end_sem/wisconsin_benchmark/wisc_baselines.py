# ═══════════════════════════════════════════════════════════════
# wisc_baselines.py  —  Self-contained: PT SLP/MLP + TF SLP/MLP
# Paste as ANY cell in Colab — defines everything it needs.
# Outputs: wisc_final_comparison.png, wisc_dl_results.json
# ═══════════════════════════════════════════════════════════════
import time, json, os, warnings
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ── CONFIG (edit here if needed) ──────────────────────────────
MEMBRANE_COUNTS = [25, 50, 100, 150, 200]
STEPS_SWEEP     = [10, 25, 50, 100]
RUNS            = 10
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

# ── DATA ──────────────────────────────────────────────────────
_d = load_breast_cancer()
_X, _y = _d.data.astype('float32'), _d.target.astype('float32')
X_train, X_test, y_train, y_test = train_test_split(
    _X, _y, test_size=TEST_SIZE, stratify=_y, random_state=RANDOM_STATE)
_sc = StandardScaler()
X_train = _sc.fit_transform(X_train).astype('float32')
X_test  = _sc.transform(X_test).astype('float32')
NF = X_train.shape[1]   # 30

# ── HELPERS ───────────────────────────────────────────────────
def stratified_sample(X, y, n, seed=42):
    n   = min(n, len(X))
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.where(y==1)[0], n//2, replace=False)
    neg = rng.choice(np.where(y==0)[0], n-n//2, replace=False)
    idx = np.concatenate([pos, neg]); rng.shuffle(idx)
    return X[idx], y[idx]

def metrics(y_true, y_pred_prob, thresh=0.5):
    yp  = (y_pred_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, yp)
    return dict(
        acc  = acc,
        prec = precision_score(y_true, yp, zero_division=0),
        rec  = recall_score(y_true, yp, zero_division=0),
        f1   = f1_score(y_true, yp, zero_division=0),
    )

# ── DEVICE ────────────────────────────────────────────────────

# Optional TensorFlow import
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    TF_OK = True
except ImportError:
    TF_OK = False
    print("TensorFlow not available — TF baselines skipped")

dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Xte  = torch.tensor(X_test).to(dev)
print(f"Device: {dev}  |  TF available: {TF_OK}\n")

# ── PyTorch models ────────────────────────────────────────────────────
class PT_SLP(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.fc = nn.Linear(nf, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze(1)

class PT_MLP(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nf, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),  nn.Sigmoid())
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_pt(model, Xd, yd, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fn  = nn.BCELoss()
    t0  = time.time()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        fn(model(Xd), yd).backward(); opt.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.time()-t0)*1000

def eval_pt(model):
    model.eval()
    with torch.no_grad():
        prob = model(Xte).cpu().numpy()
    return metrics(y_test, prob)

# ── TF models ─────────────────────────────────────────────────────────
def make_tf_slp(nf):
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(nf,))])
    m.compile(optimizer='adam', loss='binary_crossentropy')
    return m

def make_tf_mlp(nf):
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(nf,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')])
    m.compile(optimizer='adam', loss='binary_crossentropy')
    return m

def train_tf(model, Xd, yd, epochs):
    t0 = time.time()
    model.fit(Xd, yd, epochs=epochs, batch_size=len(Xd), verbose=0)
    return (time.time()-t0)*1000

def eval_tf(model):
    prob = model.predict(X_test, verbose=0).flatten()
    return metrics(y_test, prob)

# ── SWEEP ─────────────────────────────────────────────────────────────
dl_results = {}

print(f"{'Mems':>5} {'Eps':>5} | "
      f"{'PT-SLP':>9} {'ms':>7} | "
      f"{'PT-MLP':>9} {'ms':>7} | "
      f"{'TF-SLP':>9} {'ms':>7} | "
      f"{'TF-MLP':>9} {'ms':>7}")
print("-" * 80)

for nm in MEMBRANE_COUNTS:
    Xm, ym = stratified_sample(X_train, y_train, nm)
    Xm_t   = torch.tensor(Xm).to(dev)
    ym_t   = torch.tensor(ym).to(dev)

    for epochs in STEPS_SWEEP:
        key = f"{nm}_{epochs}"
        row = dict(nm=nm, epochs=epochs)

        # ── PyTorch SLP ──────────────────────────────────────────────
        pt_slp_ms_list = []
        for _ in range(RUNS):
            m = PT_SLP(NF).to(dev)
            pt_slp_ms_list.append(train_pt(m, Xm_t, ym_t, epochs))
        row['pt_slp_ms'] = float(np.median(pt_slp_ms_list))
        row.update({f'pt_slp_{k}': v for k, v in eval_pt(m).items()})

        # ── PyTorch MLP ──────────────────────────────────────────────
        pt_mlp_ms_list = []
        for _ in range(RUNS):
            m = PT_MLP(NF).to(dev)
            pt_mlp_ms_list.append(train_pt(m, Xm_t, ym_t, epochs))
        row['pt_mlp_ms'] = float(np.median(pt_mlp_ms_list))
        row.update({f'pt_mlp_{k}': v for k, v in eval_pt(m).items()})

        # ── TF SLP ───────────────────────────────────────────────────
        if TF_OK:
            tf_slp_ms_list = []
            for _ in range(RUNS):
                m = make_tf_slp(NF)
                tf_slp_ms_list.append(train_tf(m, Xm, ym, epochs))
            row['tf_slp_ms'] = float(np.median(tf_slp_ms_list))
            tm = eval_tf(m)
            row.update({f'tf_slp_{k}': v for k, v in tm.items()})

            # ── TF MLP ───────────────────────────────────────────────
            tf_mlp_ms_list = []
            for _ in range(RUNS):
                m = make_tf_mlp(NF)
                tf_mlp_ms_list.append(train_tf(m, Xm, ym, epochs))
            row['tf_mlp_ms'] = float(np.median(tf_mlp_ms_list))
            tm = eval_tf(m)
            row.update({f'tf_mlp_{k}': v for k, v in tm.items()})

        dl_results[key] = row

        tf_s = f"{row.get('tf_slp_acc', float('nan')):.4f}/{row.get('tf_slp_ms', 0):.0f}ms" if TF_OK else "N/A"
        tf_m = f"{row.get('tf_mlp_acc', float('nan')):.4f}/{row.get('tf_mlp_ms', 0):.0f}ms" if TF_OK else "N/A"
        print(f"{nm:>5} {epochs:>5} | "
              f"{row['pt_slp_acc']:.4f} {row['pt_slp_ms']:>6.0f}ms | "
              f"{row['pt_mlp_acc']:.4f} {row['pt_mlp_ms']:>6.0f}ms | "
              f"{tf_s:>17} | {tf_m:>17}")

with open('wisc_dl_results.json', 'w') as f:
    json.dump(dl_results, f, indent=2)
print("\nSaved: wisc_dl_results.json")

# ── LOAD BNPS RESULTS ─────────────────────────────────────────────────
bnps_results = {}
if os.path.exists('wisc_bnps_results.json'):
    with open('wisc_bnps_results.json') as f:
        bnps_results = json.load(f)
else:
    print("wisc_bnps_results.json not found — run wisc_bnps.py first")

# ── PLOTS ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('BNPS vs PyTorch vs TensorFlow — Wisconsin Breast Cancer\n'
             'Fair Comparison: Same N Samples, Same Steps/Epochs',
             fontsize=13, fontweight='bold')

COLORS = {
    'BNPS Serial': '#27ae60',
    'BNPS CUDA':   '#2ecc71',
    'PT SLP':      '#3498db',
    'PT MLP':      '#e74c3c',
    'TF SLP':      '#9b59b6',
    'TF MLP':      '#e67e22',
}

# ── Plot 1: Acc vs Membranes (fixed steps = max STEPS_SWEEP) ──────────
ax = axes[0, 0]
fix_steps = max(STEPS_SWEEP)
for label, src, acc_key, ms_key in [
    ('BNPS Serial', bnps_results, 'acc',        'serial_ms'),
    ('BNPS CUDA',   bnps_results, 'acc',        'cuda_ms'),
    ('PT SLP',      dl_results,   'pt_slp_acc', 'pt_slp_ms'),
    ('PT MLP',      dl_results,   'pt_mlp_acc', 'pt_mlp_ms'),
]:
    xs, ys = [], []
    for nm in MEMBRANE_COUNTS:
        k = f"{nm}_{fix_steps}"
        if k in src and not np.isnan(src[k].get(acc_key, float('nan'))):
            xs.append(nm); ys.append(src[k][acc_key])
    if xs:
        ax.plot(xs, ys, 'o-', color=COLORS[label], lw=2, ms=7, label=label)

if TF_OK:
    for label, acc_key in [('TF SLP','tf_slp_acc'),('TF MLP','tf_mlp_acc')]:
        xs, ys = [], []
        for nm in MEMBRANE_COUNTS:
            k = f"{nm}_{fix_steps}"
            if k in dl_results: xs.append(nm); ys.append(dl_results[k].get(acc_key, float('nan')))
        if xs: ax.plot(xs, ys, 's--', color=COLORS[label], lw=2, ms=7, label=label)

ax.set_xlabel('Membrane Count (= Training Samples)', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title(f'Accuracy vs Membrane Count\n(steps/epochs = {fix_steps})', fontsize=10)
ax.set_ylim(0.70, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Plot 2: Acc vs Steps (fixed nm = median membrane count) ───────────
ax = axes[0, 1]
fix_nm = MEMBRANE_COUNTS[len(MEMBRANE_COUNTS)//2]
for label, src, acc_key in [
    ('BNPS Serial', bnps_results, 'acc'),
    ('PT SLP',      dl_results,   'pt_slp_acc'),
    ('PT MLP',      dl_results,   'pt_mlp_acc'),
]:
    xs, ys = [], []
    for steps in STEPS_SWEEP:
        k = f"{fix_nm}_{steps}"
        if k in src and not np.isnan(src[k].get(acc_key, float('nan'))):
            xs.append(steps); ys.append(src[k][acc_key])
    if xs: ax.plot(xs, ys, 'o-', color=COLORS[label], lw=2, ms=7, label=label)

if TF_OK:
    for label, acc_key in [('TF SLP','tf_slp_acc'),('TF MLP','tf_mlp_acc')]:
        xs, ys = [], []
        for steps in STEPS_SWEEP:
            k = f"{fix_nm}_{steps}"
            if k in dl_results: xs.append(steps); ys.append(dl_results[k].get(acc_key, float('nan')))
        if xs: ax.plot(xs, ys, 's--', color=COLORS[label], lw=2, ms=7, label=label)

ax.set_xlabel('Steps / Epochs', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title(f'Accuracy vs Steps/Epochs\n(membranes = {fix_nm})', fontsize=10)
ax.set_ylim(0.70, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Plot 3: Training Time vs Membranes ────────────────────────────────
ax = axes[1, 0]
for label, src, ms_key in [
    ('BNPS Serial', bnps_results, 'serial_ms'),
    ('BNPS CUDA',   bnps_results, 'cuda_ms'),
    ('PT SLP',      dl_results,   'pt_slp_ms'),
    ('PT MLP',      dl_results,   'pt_mlp_ms'),
]:
    xs, ys = [], []
    for nm in MEMBRANE_COUNTS:
        k = f"{nm}_{fix_steps}"
        if k in src and not np.isnan(src[k].get(ms_key, float('nan'))):
            xs.append(nm); ys.append(src[k][ms_key])
    if xs: ax.plot(xs, ys, 'o-', color=COLORS[label], lw=2, ms=7, label=label)

if TF_OK:
    for label, ms_key in [('TF SLP','tf_slp_ms'),('TF MLP','tf_mlp_ms')]:
        xs, ys = [], []
        for nm in MEMBRANE_COUNTS:
            k = f"{nm}_{fix_steps}"
            if k in dl_results: xs.append(nm); ys.append(dl_results[k].get(ms_key, 0))
        if xs: ax.plot(xs, ys, 's--', color=COLORS[label], lw=2, ms=7, label=label)

ax.set_xlabel('Membrane Count (= Training Samples)', fontsize=11)
ax.set_ylabel('Training Time (ms)', fontsize=11)
ax.set_title(f'Training Time vs Membrane Count\n(steps/epochs = {fix_steps})', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Plot 4: Best result bar chart (best cell per method) ──────────────
ax = axes[1, 1]
method_best = {}
for label, src, acc_key, ms_key in [
    ('BNPS\nSerial',  bnps_results, 'acc',        'serial_ms'),
    ('BNPS\nCUDA',    bnps_results, 'acc',        'cuda_ms'),
    ('PT\nSLP',       dl_results,   'pt_slp_acc', 'pt_slp_ms'),
    ('PT\nMLP',       dl_results,   'pt_mlp_acc', 'pt_mlp_ms'),
]:
    best_acc, best_ms = float('-inf'), 0
    for k, v in src.items():
        a = v.get(acc_key, float('nan'))
        if not np.isnan(a) and a > best_acc:
            best_acc = a
            best_ms  = v.get(ms_key, 0) or 0
    if best_acc > float('-inf'):
        method_best[label] = (best_acc, best_ms)

if TF_OK:
    for label, acc_key, ms_key in [
        ('TF\nSLP','tf_slp_acc','tf_slp_ms'),
        ('TF\nMLP','tf_mlp_acc','tf_mlp_ms')
    ]:
        best_acc, best_ms = float('-inf'), 0
        for k, v in dl_results.items():
            a = v.get(acc_key, float('nan'))
            if not np.isnan(a) and a > best_acc:
                best_acc = a; best_ms = v.get(ms_key, 0)
        if best_acc > float('-inf'):
            method_best[label] = (best_acc, best_ms)

labels   = list(method_best.keys())
accs     = [method_best[l][0] for l in labels]
col_list = [COLORS.get(l.replace('\n', ' '), '#7f8c8d') for l in labels]
bars = ax.bar(labels, accs, color=col_list, alpha=0.85, edgecolor='white')
for b in bars:
    ax.annotate(f'{b.get_height():.4f}',
                xy=(b.get_x()+b.get_width()/2, b.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0.75, 1.10)
ax.set_ylabel('Best Accuracy', fontsize=11)
ax.set_title('Best Accuracy per Method\n(over all membrane/step combinations)', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('wisc_final_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: wisc_final_comparison.png")

# ── TEXT SUMMARY ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL COMPARISON — Best per Method")
print(f"{'='*65}")
print(f"  {'Method':<20} {'Best Acc':>10} {'Train(ms)':>12}")
print("  " + "-"*44)
for label in labels:
    a, ms = method_best[label]
    print(f"  {label.replace(chr(10),' '):<20} {a:>10.4f} {ms:>12.1f}")
print(f"{'='*65}")
