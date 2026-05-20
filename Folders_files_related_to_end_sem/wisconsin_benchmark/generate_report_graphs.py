"""
generate_report_graphs.py — Report-quality graphs from BNPS_Wisconsin_Best.py results
=======================================================================================
Generates 4 publication-ready figures for chapter4 of the LaTeX report:

  1. wisconsin_accuracy_bars.png  — Method comparison bar chart (main figure)
  2. wisconsin_step_sweep.png     — BNPS accuracy vs steps (convergence)
  3. wisconsin_membrane_sweep.png — BNPS accuracy vs membrane count
  4. wisconsin_speedup.png        — BNPS Serial vs CUDA speedup (existing data)

Copy output PNGs to:
  END_REVIEW/.../CS22B1099/Chapter4/
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize':10,
    'figure.dpi':     150,
})

CHAP4 = r"END_REVIEW\Boolean_Numerical_P_System_Based_Model_for_Cloud_Workload_Prediction__6_\CS22B1099\Chapter4"

COLORS = {
    'BNPS':     '#2ecc71',
    'PT SLP':   '#3498db',
    'PT MLP':   '#e74c3c',
    'sklearn':  '#9b59b6',
    'accent':   '#f39c12',
}

# ── Data from BNPS_Wisconsin_Best.py run ────────────────────────────

# [3] Step sweep (100 membranes fixed)
steps   = [10,    25,    50,    75,    100,   150]
acc_s   = [0.9035,0.9561,0.9649,0.9737,0.9649,0.9649]
f1_s    = [0.9209,0.9650,0.9718,0.9790,0.9722,0.9722]

# [4] Membrane sweep (75 steps fixed)
mems    = [25,    50,    75,    100,   150,   200]
acc_m   = [0.9298,0.9298,0.9298,0.9737,0.9386,0.9474]
time_m  = [5.9,   6.2,   6.5,   6.8,   7.6,   8.4]   # ms

# [Final] Head-to-head (best each)
methods = ['sklearn LR\n(full data)', 'BNPS SLP\n(100m, 75s)', 'PT MLP\n(4-layer)', 'PT SLP\n(1-layer)']
acc_f   = [0.9825,                    0.9737,                   0.9561,              0.9298]
time_f  = [20.9,                      6.8,                      197.7,               101.5]
cols_f  = [COLORS['sklearn'], COLORS['BNPS'], COLORS['PT MLP'], COLORS['PT SLP']]

# [Speedup] Serial vs CUDA (from old BNPS_02 notebook)
spd_mems    = [10,      25,      50,      100,     200]
serial_ms   = [161.4,   1616.0,  1763.9,  7080.5,  25015.6]
cuda_ms     = [14.04,   31.71,   34.54,   80.14,   179.57]
speedups    = [s/c for s, c in zip(serial_ms, cuda_ms)]

import os
os.makedirs(CHAP4, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# FIGURE 1: Main method comparison bar chart
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))
x   = np.arange(len(methods))
bars = ax.bar(x, acc_f, color=cols_f, width=0.55, edgecolor='white', linewidth=1.5, zorder=3)

# Annotate bars
for i, (b, t) in enumerate(zip(bars, time_f)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
            f'{b.get_height():.4f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(b.get_x() + b.get_width()/2, b.get_height()/2,
            f'{t:.1f} ms', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

# BNPS arrow annotation
ax.annotate('BNPS beats\nPT MLP by +1.76%\n14.9–29× faster',
            xy=(1, 0.9737), xytext=(2.5, 0.9600),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
            fontsize=9, color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', fc='#ffeaa7', ec='#fdcb6e', alpha=0.9))

ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(0.88, 1.02)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Wisconsin Breast Cancer — Method Comparison\n'
             '(Train time shown inside bars; BNPS uses only 100 of 455 training samples)',
             fontsize=11, fontweight='bold')
ax.axhline(0.9737, color=COLORS['BNPS'], ls='--', lw=1.2, alpha=0.5, zorder=2)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
out1 = os.path.join(CHAP4, 'wisconsin_accuracy_bars.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.savefig('wisconsin_accuracy_bars.png', dpi=150, bbox_inches='tight')
print(f"Saved: {out1}")

# ════════════════════════════════════════════════════════════════════
# FIGURE 2: Step sweep — convergence curve
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, acc_s, 'o-', color=COLORS['BNPS'], lw=2.5, ms=9, zorder=3, label='BNPS SLP Accuracy')
ax.fill_between(steps, acc_s, 0.88, alpha=0.12, color=COLORS['BNPS'])

# Mark best
best_step = steps[np.argmax(acc_s)]
best_acc  = max(acc_s)
ax.annotate(f'Best: {best_acc:.4f}\n@ {best_step} steps',
            xy=(best_step, best_acc), xytext=(best_step + 15, best_acc - 0.012),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
            fontsize=10, color='#e74c3c', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#ffeaa7', ec='#fdcb6e', alpha=0.9))

# Reference lines
for label, val, col in [('sklearn LR', 0.9825, COLORS['sklearn']),
                          ('PT MLP',     0.9561, COLORS['PT MLP']),
                          ('PT SLP',     0.9298, COLORS['PT SLP'])]:
    ax.axhline(val, ls='--', lw=1.3, color=col, alpha=0.75, label=label)

ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('BNPS Accuracy vs.\\ Training Steps\n(100 Membranes, LR=0.05, Momentum=0.85, 30 Features)',
             fontsize=11, fontweight='bold')
ax.set_xlim(5, 160); ax.set_ylim(0.88, 1.00)
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
out2 = os.path.join(CHAP4, 'wisconsin_step_sweep.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.savefig('wisconsin_step_sweep.png', dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")

# ════════════════════════════════════════════════════════════════════
# FIGURE 3: Membrane sweep — accuracy + time dual axis
# ════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

l1, = ax1.plot(mems, acc_m, 'o-', color=COLORS['BNPS'], lw=2.5, ms=9, label='BNPS Accuracy')
ax1.fill_between(mems, acc_m, 0.90, alpha=0.12, color=COLORS['BNPS'])
l2, = ax2.plot(mems, time_m, 's--', color=COLORS['accent'], lw=2, ms=8, label='Training Time (ms)')

# Mark best
best_m_idx = int(np.argmax(acc_m))
ax1.annotate(f'Best: {acc_m[best_m_idx]:.4f}\n@ {mems[best_m_idx]} mems',
             xy=(mems[best_m_idx], acc_m[best_m_idx]),
             xytext=(mems[best_m_idx] + 20, acc_m[best_m_idx] - 0.015),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
             fontsize=10, color='#e74c3c', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='#ffeaa7', ec='#fdcb6e', alpha=0.9))

ax1.axhline(0.9561, ls='--', lw=1.3, color=COLORS['PT MLP'], alpha=0.7, label='PT MLP baseline')
ax1.set_xlabel('Membrane Count (= Training Samples Used)', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12, color=COLORS['BNPS'])
ax2.set_ylabel('Training Time (ms)', fontsize=12, color=COLORS['accent'])
ax1.set_title('BNPS Accuracy & Training Time vs.\\ Membrane Count\n(75 Steps, LR=0.05, Momentum=0.85)',
              fontsize=11, fontweight='bold')
ax1.set_ylim(0.90, 1.00)
lines = [l1, l2, mpatches.Patch(color=COLORS['PT MLP'], alpha=0.7, label='PT MLP baseline')]
ax1.legend(handles=lines, fontsize=9, loc='upper left')
ax1.grid(alpha=0.3); ax1.spines['top'].set_visible(False)
plt.tight_layout()
out3 = os.path.join(CHAP4, 'wisconsin_membrane_sweep.png')
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.savefig('wisconsin_membrane_sweep.png', dpi=150, bbox_inches='tight')
print(f"Saved: {out3}")

# ════════════════════════════════════════════════════════════════════
# FIGURE 4: CUDA vs Serial speedup (log scale)
# ════════════════════════════════════════════════════════════════════
fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 5))

# Left: execution time
ax_t.semilogy(spd_mems, serial_ms, 'o-', color='#e74c3c', lw=2.5, ms=9, label='Serial CPU')
ax_t.semilogy(spd_mems, cuda_ms,   's-', color=COLORS['BNPS'], lw=2.5, ms=9, label='BNPS CUDA GPU')
ax_t.set_xlabel('Membrane Count', fontsize=12)
ax_t.set_ylabel('Execution Time (ms, log scale)', fontsize=12)
ax_t.set_title('Execution Time vs.\\ Membrane Count', fontsize=11, fontweight='bold')
ax_t.legend(fontsize=10); ax_t.grid(alpha=0.3)
ax_t.spines['top'].set_visible(False); ax_t.spines['right'].set_visible(False)

# Right: speedup bar
bars_s = ax_s.bar(spd_mems, speedups, color=COLORS['BNPS'], width=28,
                   edgecolor='white', alpha=0.88)
for b, v in zip(bars_s, speedups):
    ax_s.text(b.get_x() + b.get_width()/2, b.get_height() + 3,
              f'{v:.0f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_s.set_xlabel('Membrane Count', fontsize=12)
ax_s.set_ylabel('Speedup (Serial / CUDA)', fontsize=12)
ax_s.set_title('BNPS CUDA Parallel Speedup\n(vs.\\ Sequential CPU Execution)',
               fontsize=11, fontweight='bold')
ax_s.grid(axis='y', alpha=0.3)
ax_s.spines['top'].set_visible(False); ax_s.spines['right'].set_visible(False)

plt.suptitle('BNPS GPU Parallelism: Up to 139× Speedup at 200 Membranes',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
out4 = os.path.join(CHAP4, 'wisconsin_speedup.png')
plt.savefig(out4, dpi=150, bbox_inches='tight')
plt.savefig('wisconsin_speedup.png', dpi=150, bbox_inches='tight')
print(f"Saved: {out4}")

print("\n✅ All 4 report figures generated:")
print(f"   {out1}")
print(f"   {out2}")
print(f"   {out3}")
print(f"   {out4}")
print("\nThey are also saved locally for Colab upload.")
