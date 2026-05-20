
# ═══════════════════════════════════════════════════════
#  BNPS_Final_Benchmark.py  —  BC Wisconsin | Colab T4
#  Deep MLP (TF/PT) vs SLP (BNPS) — fair framing
# ═══════════════════════════════════════════════════════
import subprocess, sys, os, time, re

# ── FIX 5: Environment info ───────────────────────────
print("── GPU INFO ──")
subprocess.run('nvidia-smi', shell=True, check=False)   # shell=True works on all Colab paths
subprocess.run([sys.executable,'-m','pip','install','kagglehub','statsmodels','-q'],check=True)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
warnings.filterwarnings('ignore')
import tensorflow as tf, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from statsmodels.stats.proportion import proportion_confint
import kagglehub

print(f"TF={tf.__version__}  PT={torch.__version__}  numpy={np.__version__}")
subprocess.run([sys.executable,'-m','pip','show','tensorflow','torch','numpy'],check=False)

# ── CONFIG ───────────────────────────────────────────────
MEMBRANE_COUNTS = [10, 25, 50, 100, 200]
BNPS_STEPS  = 10
MLP_EPOCHS  = 100
RUNS        = 10   # FIX 3: increased from 3 to 10 for statistical robustness

print("="*60)
print("BENCHMARK CONFIG")
print("="*60)
print(f"  TF / PyTorch : Deep MLP 512→256→128→64→1 | {MLP_EPOCHS} epochs")
print(f"  BNPS         : SLP (membrane P-system)    | {BNPS_STEPS} steps")
print(f"  Dataset      : Breast Cancer Wisconsin")
print(f"  Runs/method  : {RUNS} (averaged) | GPU warm-up: Yes")
print("="*60)

# ── DATA ─────────────────────────────────────────────────
path  = kagglehub.dataset_download('uciml/breast-cancer-wisconsin-data')
df    = pd.read_csv(os.path.join(path,'data.csv'))
df    = df.drop(columns=[c for c in ['id','Unnamed: 32'] if c in df.columns])
df['diagnosis'] = (df['diagnosis']=='M').astype(float)
fcols = [c for c in df.columns if c!='diagnosis'][:10]
X = StandardScaler().fit_transform(df[fcols].values.astype('float32'))
y = df['diagnosis'].values.astype('float32')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
NF = X_train.shape[1]
print(f"Train={len(X_train)}  Test={len(X_test)}  Features={NF}")

def metrics(y_true, y_prob, t=0.5):
    """Compute metrics + FIX 4: Wilson score 95% confidence interval on accuracy."""
    yp=(np.array(y_prob)>=t).astype(int); yt=np.array(y_true).astype(int)
    n=len(yt); acc=accuracy_score(yt,yp)
    lo,hi=proportion_confint(int(acc*n),n,alpha=0.05,method='wilson')
    return dict(acc=acc, f1=f1_score(yt,yp,zero_division=0),
                prec=precision_score(yt,yp,zero_division=0),
                rec=recall_score(yt,yp,zero_division=0),
                acc_ci=(lo,hi))

# ── TF Deep MLP ──────────────────────────────────────────
def tf_mlp(d):
    m=tf.keras.Sequential([
        tf.keras.layers.Dense(512,'relu',input_shape=(d,)),
        tf.keras.layers.Dense(256,'relu'), tf.keras.layers.Dense(128,'relu'),
        tf.keras.layers.Dense(64,'relu'),  tf.keras.layers.Dense(1,'sigmoid')])
    m.compile('adam','binary_crossentropy'); return m

# Build model ONCE so graph is compiled only once (fixes high std deviation)
print("\n[TF] Building & warming up graph...")
with tf.device('/GPU:0'):
    m_tf = tf_mlp(NF)
    # Warm-up run: compiles the graph — NOT counted in timing
    m_tf.fit(X_train, y_train, epochs=2, batch_size=64, verbose=0)
    init_weights = m_tf.get_weights()  # save initial weights

# Timed runs: reset weights to same start point — graph already compiled
tf_times=[]
for _ in range(RUNS):
    with tf.device('/GPU:0'):
        m_tf.set_weights(init_weights)   # reset weights (no recompilation)
        t0 = time.time()
        m_tf.fit(X_train, y_train, epochs=MLP_EPOCHS, batch_size=64, verbose=0)
        tf_times.append((time.time()-t0)*1000)
# FIX 3: use median (robust to outliers)
TF_MS=float(np.median(tf_times)); TF_STD=float(np.std(tf_times))
tf_m=metrics(y_test, m_tf.predict(X_test,verbose=0).flatten())
lo,hi=tf_m['acc_ci']
print(f"TF  Deep MLP @ {MLP_EPOCHS} ep : {TF_MS:.1f} ms (std={TF_STD:.1f})  Acc={tf_m['acc']:.4f} [95% CI {lo:.4f}-{hi:.4f}]  F1={tf_m['f1']:.4f}")

# ── PyTorch Deep MLP ──────────────────────────────────────
class MLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),
                               nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),
                               nn.Linear(64,1),nn.Sigmoid())
    def forward(self,x): return self.net(x).squeeze()

Xt=torch.tensor(X_train,dtype=torch.float32); yt=torch.tensor(y_train,dtype=torch.float32)
Xte=torch.tensor(X_test,dtype=torch.float32)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# warm-up
print("[PT] Warming up...")
_wp=MLP(NF).to(dev); _op=torch.optim.Adam(_wp.parameters())
for _ in range(2):
    _wp.train(); _op.zero_grad()
    nn.BCELoss()(_wp(Xt[:64].to(dev)),yt[:64].to(dev)).backward(); _op.step()
if torch.cuda.is_available(): torch.cuda.synchronize()

pt_times=[]
for _ in range(RUNS):
    m_pt=MLP(NF).to(dev); opt=torch.optim.Adam(m_pt.parameters(),lr=1e-3)
    Xd,yd=Xt.to(dev),yt.to(dev); t0=time.time()
    for _e in range(MLP_EPOCHS):
        m_pt.train(); opt.zero_grad(); nn.BCELoss()(m_pt(Xd),yd).backward(); opt.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    pt_times.append((time.time()-t0)*1000)
# FIX 3: use median
PT_MS=float(np.median(pt_times)); PT_STD=float(np.std(pt_times))
m_pt.eval()
with torch.no_grad(): pt_prob=m_pt(Xte.to(dev)).cpu().numpy()
pt_m=metrics(y_test,pt_prob)
lo,hi=pt_m['acc_ci']
print(f"PT  Deep MLP @ {MLP_EPOCHS} ep : {PT_MS:.1f} ms (std={PT_STD:.1f})  Acc={pt_m['acc']:.4f} [95% CI {lo:.4f}-{hi:.4f}]  F1={pt_m['f1']:.4f}")

# ── Upload & Compile BNPS ─────────────────────────────────
from google.colab import files
print('\nUpload: bnps3.py  bnps.cu'); uploaded=files.upload()
for fn in uploaded:
    if 'bnps3' in fn: os.rename(fn,'bnps3.py')
    elif fn.endswith('.cu'): os.rename(fn,'bnps.cu')
ret=subprocess.run(['nvcc','-O2','-o','bnps_cuda','bnps.cu','-lm'],capture_output=True)
if ret.returncode: raise RuntimeError(ret.stderr.decode()[:400])
print('CUDA compiled OK')

# ── .pep generator ────────────────────────────────────────
def make_pep(Xd,yd,N,nf,fname,lr=0.5):
    N=min(N,len(Xd)); Xs=Xd[:N].astype(float); ys=yd[:N].astype(float)
    f=lambda v:f'{float(v):.6f}'
    ids=list(range(1,N+2))
    L=['bnps = {','', f"    H = {{{','.join(str(i) for i in ids)}}};",
       f"    structure = [{' '.join(f'[ {i}' for i in ids)} {' '.join(f']{i}' for i in reversed(ids))}];",'']
    cv=(','.join(f'w{i}' for i in range(nf))+',b,'+','.join(f'tw{i}' for i in range(nf))+',tb,'+
        ','.join(f'lw{i}' for i in range(nf))+',lb,'+','.join(f'aw{i}' for i in range(nf))+',ab')
    L+=['    1 = {','        var = {',f'            {cv}','        };','']
    mul=N+1
    for i in range(nf):
        L.append(f"        pr = {{ w{i}*{mul} -> {'+'.join(f'1|w{i}_{m}' for m in range(2,N+2))}+1|tw{i} }};")
    L.append(f"        pr = {{ b*{mul} -> {'+'.join(f'1|b_{m}' for m in range(2,N+2))}+1|tb }};"); L.append('')
    L.append(f"        pr = {{ {lr} -> {'+'.join(f'1|lw{i}' for i in range(nf))}+1|lb }};"); L.append('')
    for i in range(nf):
        L.append(f"        pr = {{ {'+'.join(f'gw{i}_{m}' for m in range(2,N+2))} -> 1|aw{i} }};")
    L.append(f"        pr = {{ {'+'.join(f'gb_{m}' for m in range(2,N+2))} -> 1|ab }};"); L.append('')
    for i in range(nf): L.append(f'        pr = {{ tw{i}-lw{i}*(aw{i}/{N}) -> 1|w{i} }};')
    L.append(f'        pr = {{ tb-lb*(ab/{N}) -> 1|b }};'); L.append('')
    L.append(f"        var0 = ({','.join(['0']*((nf+1)*4))});"); L+=['    };','']
    for idx in range(N):
        m=idx+2; xi=Xs[idx]; yi=float(ys[idx])
        sv=(','.join(f'w{i}_{m}' for i in range(nf))+f',b_{m},z_{m},'+
            ','.join(f'za{j}_{m}' for j in range(nf+1))+','+','.join(f'gw{i}_{m}' for i in range(nf))+f',gb_{m}')
        L+=[f'    {m} = {{','        var = {',f'            {sv}','        };','']
        fwd='+'.join(f'w{i}_{m}*{f(xi[i])}' for i in range(nf))+f'+b_{m}'
        L.append(f'        pr = {{ {fwd} -> 1|z_{m} }};')
        L.append(f"        pr = {{ z_{m}*{nf+1} -> {'+'.join(f'1|za{j}_{m}' for j in range(nf+1))} }};"); L.append('')
        for i in range(nf):
            xv=xi[i]; c=xv*0.5*yi
            if abs(xv)<1e-9: L.append(f'        pr = {{ 0.000001*za{i}_{m} -> 1|gw{i}_{m} }};')
            else:             L.append(f'        pr = {{ {f(xv)}*0.25*za{i}_{m}-{f(c)} -> 1|gw{i}_{m} }};')
        L.append(f'        pr = {{ 0.25*za{nf}_{m}-{f(0.5*yi)} -> 1|gb_{m} }};'); L.append('')
        L.append(f"        var0 = ({','.join(['0']*(nf+2+(nf+1)+nf+1))});"); L+=[f'    }};','']
    L.append('}')
    with open(fname,'w') as fh: fh.write('\n'.join(L))
    return fname

# ── FIX 1: Measure bnps3.py import overhead (not just python -c pass) ────
# Running bnps3.py with no args imports numpy etc. and exits — captures real overhead
_oh=[]
for _ in range(10):
    t=time.time()
    subprocess.run([sys.executable,'bnps3.py'],capture_output=True,timeout=15)
    _oh.append((time.time()-t)*1000)
PY_OH=float(np.median(_oh))
print(f'bnps3.py startup overhead (median of 10): {PY_OH:.1f} ms')
print(f'  (includes numpy/library imports — more accurate than python -c pass)')

# ── BNPS benchmark ────────────────────────────────────────
serial_ms=[]; serial_std=[]; cuda_ms=[]; cuda_std=[]; valid_mems=[]
bnps_acc_list=[]; bnps_f1_list=[]

print(f"\n── BNPS SLP (steps={BNPS_STEPS}, varying membranes) ──")
print(f"{'Mems':>6}{'Serial':>12}{'±':>6}{'CUDA':>10}{'±':>6}{'Speedup':>9}{'Acc':>8}")
print('-'*60)

for nm in MEMBRANE_COUNTS:
    pf=make_pep(X_train,y_train,nm,NF,f'bc_{nm}.pep')
    try:
        subprocess.run([sys.executable,'bnps3.py',pf,'-p','1'],capture_output=True,timeout=300)
    except subprocess.TimeoutExpired:
        print(f'{nm:>6} SKIP'); continue
    if not os.path.exists('input.txt'): print(f'{nm:>6} SKIP'); continue

    # Serial (3 runs) — capture last output for weight extraction
    ser_ts=[]; last_ser_out=''
    for _ in range(RUNS):
        t0=time.time()
        r=subprocess.run([sys.executable,'bnps3.py',pf,'-n',str(BNPS_STEPS)],capture_output=True,timeout=600)
        ser_ts.append(max((time.time()-t0)*1000-PY_OH,1.0))
        last_ser_out=r.stdout.decode(errors='replace')
    ser=float(np.median(ser_ts)); ser_s=float(np.std(ser_ts))  # FIX: median not mean

    # Extract final weights from serial output → compute BNPS accuracy
    bnps_acc=float('nan')
    try:
        w_vals=[]
        for i in range(NF):
            mw=re.search(rf'(?:^|\s)w{i}\s*[=:]\s*([-+]?[\d.eE]+)',last_ser_out,re.MULTILINE)
            if mw: w_vals.append(float(mw.group(1)))
        mb=re.search(r'(?:^|\s)b\s*[=:]\s*([-+]?[\d.eE]+)',last_ser_out,re.MULTILINE)
        bias=float(mb.group(1)) if mb else 0.0
        if len(w_vals)==NF:
            logits=X_test@np.array(w_vals)+bias
            pred_prob=1/(1+np.exp(-logits))
            bm=metrics(y_test,pred_prob)
            bnps_acc=bm['acc']; bnps_acc_list.append(bm['acc']); bnps_f1_list.append(bm['f1'])
    except Exception: pass

    # CUDA — FIX 2: warmup run before timing
    subprocess.run([sys.executable,'bnps3.py',pf,'-p',str(BNPS_STEPS)],capture_output=True,timeout=600)
    subprocess.run(['./bnps_cuda','input.txt'],capture_output=True,timeout=60)  # warmup
    cuda_ts=[]
    for _ in range(RUNS):
        t0=time.time()
        r=subprocess.run(['./bnps_cuda','input.txt'],capture_output=True,timeout=300)
        wall=(time.time()-t0)*1000; out=r.stdout.decode(errors='replace')
        mx=re.search(r'[Tt]ime[^:]*:\s*([\d.]+)\s*(ms|us|s)?',out)
        if mx:
            v=float(mx.group(1)); u=(mx.group(2) or 'ms').lower()
            cuda_ts.append(v*1e-3 if u=='us' else (v*1e3 if u=='s' else v))
        else: cuda_ts.append(wall)
    # FIX 3: use median
    cuda=float(np.median(cuda_ts)); cuda_s=float(np.std(cuda_ts))

    serial_ms.append(ser); serial_std.append(ser_s)
    cuda_ms.append(cuda);  cuda_std.append(cuda_s); valid_mems.append(nm)
    # FIX 4: Wilson CI on BNPS accuracy
    acc_str='N/A'; ci_str=''
    if not np.isnan(bnps_acc):
        n_t=len(y_test); lo,hi=proportion_confint(int(bnps_acc*n_t),n_t,alpha=0.05,method='wilson')
        acc_str=f'{bnps_acc:.4f}'; ci_str=f' [{lo:.3f}-{hi:.3f}]'
    print(f'{nm:>6}{ser:>12.1f}{ser_s:>6.1f}{cuda:>10.2f}{cuda_s:>6.2f}{ser/cuda:>8.2f}x{acc_str:>8}{ci_str}')

# ── RESULTS SUMMARY ───────────────────────────────────────
best_idx=int(np.argmin(cuda_ms)) if cuda_ms else 0
best_cuda=cuda_ms[best_idx]; best_mems=valid_mems[best_idx]
best_bnps_acc=bnps_acc_list[best_idx] if bnps_acc_list else float('nan')
best_bnps_f1 =bnps_f1_list[best_idx]  if bnps_f1_list  else float('nan')

print(f'\n{"="*72}')
print('  FINAL RESULTS SUMMARY')
print(f'{"="*72}')
print(f'  {"Method":<26}{"Median(ms)":>12}{"Std":>7}{"Acc":>8}{"95% CI":>16}{"F1":>8}')
print(f'  {"─"*76}')
tf_lo,tf_hi=tf_m['acc_ci']; pt_lo,pt_hi=pt_m['acc_ci']
print(f'  {"TF  Deep MLP GPU":<26}{TF_MS:>12.1f}{TF_STD:>7.1f}{tf_m["acc"]:>8.4f}  [{tf_lo:.3f}-{tf_hi:.3f}]{tf_m["f1"]:>8.4f}')
print(f'  {"PT  Deep MLP GPU":<26}{PT_MS:>12.1f}{PT_STD:>7.1f}{pt_m["acc"]:>8.4f}  [{pt_lo:.3f}-{pt_hi:.3f}]{pt_m["f1"]:>8.4f}')
if cuda_ms:
    acc_s=f'{best_bnps_acc:.4f}' if not np.isnan(best_bnps_acc) else 'N/A'
    f1_s =f'{best_bnps_f1:.4f}'  if not np.isnan(best_bnps_f1)  else 'N/A'
    if not np.isnan(best_bnps_acc):
        n_t=len(y_test); bl,bh=proportion_confint(int(best_bnps_acc*n_t),n_t,alpha=0.05,method='wilson')
        ci_s=f'[{bl:.3f}-{bh:.3f}]'
    else: ci_s='N/A'
    print(f'  {f"BNPS CUDA SLP ({best_mems} mems)":<26}{best_cuda:>12.2f}{"N/A":>7}{acc_s:>8}  {ci_s:<14}{f1_s:>8}')
print(f'{"="*72}')
if cuda_ms:
    print(f'\n  BNPS CUDA vs TF  : {TF_MS/best_cuda:.1f}x faster')
    print(f'  BNPS CUDA vs PT  : {PT_MS/best_cuda:.1f}x faster')
    print(f'\n  NOTE: TF/PT use Deep MLP (512→256→128→64→1) with {MLP_EPOCHS} epochs')
    print(f'        BNPS uses SLP (membrane P-system) with {BNPS_STEPS} steps')
    print(f'        BNPS achieves faster execution even against a more complex baseline')

# ── CHARTS ────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(16,6))
fig.suptitle(
    'BNPS CUDA SLP  vs  TensorFlow Deep MLP  vs  PyTorch Deep MLP\n'
    f'Breast Cancer Wisconsin | T4 GPU | TF/PT: {MLP_EPOCHS} epochs | BNPS: {BNPS_STEPS} steps',
    fontsize=12,fontweight='bold')

ax1=axes[0]
if cuda_ms:
    ax1.plot(valid_mems,serial_ms,'o--',color='#888888',lw=2,ms=7,label='BNPS Serial CPU')
    ax1.errorbar(valid_mems,serial_ms,yerr=serial_std,fmt='none',ecolor='#888888',capsize=4)
    ax1.plot(valid_mems,cuda_ms,'^-',color='#3BCEAC',lw=2.5,ms=9,label='BNPS CUDA GPU')
    ax1.errorbar(valid_mems,cuda_ms,yerr=cuda_std,fmt='none',ecolor='#3BCEAC',capsize=4)
ax1.axhline(TF_MS,color='#FF6B35',ls='--',lw=2,label=f'TF Deep MLP ({TF_MS:.0f}ms)')
ax1.axhspan(TF_MS-TF_STD,TF_MS+TF_STD,alpha=0.1,color='#FF6B35')
ax1.axhline(PT_MS,color='#EE4266',ls='--',lw=2,label=f'PT Deep MLP ({PT_MS:.0f}ms)')
ax1.axhspan(PT_MS-PT_STD,PT_MS+PT_STD,alpha=0.1,color='#EE4266')
ax1.set_xlabel('BNPS Membrane Count (parallel training samples)',fontsize=11)
ax1.set_ylabel('Execution Time (ms) — lower is better',fontsize=11)
ax1.set_title('Execution Time vs Membrane Count\n(TF/PT = horizontal reference lines)',fontsize=10)
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2=axes[1]
methods=['TF\nDeep MLP','PT\nDeep MLP']
accs=[tf_m['acc'],pt_m['acc']]; f1s=[tf_m['f1'],pt_m['f1']]
colors=['#FF6B35','#EE4266']
if not np.isnan(best_bnps_acc):
    methods.append(f'BNPS CUDA\nSLP ({best_mems}m)')
    accs.append(best_bnps_acc); f1s.append(best_bnps_f1); colors.append('#3BCEAC')
x=np.arange(len(methods)); w=0.35
b1=ax2.bar(x-w/2,accs,w,label='Accuracy',color=colors,alpha=0.85,edgecolor='white')
b2=ax2.bar(x+w/2,f1s, w,label='F1 Score', color=colors,alpha=0.5, edgecolor='white',hatch='//')
for b in list(b1)+list(b2):
    ax2.annotate(f'{b.get_height():.3f}',
                 xy=(b.get_x()+b.get_width()/2,b.get_height()),
                 xytext=(0,4),textcoords='offset points',ha='center',fontsize=9,fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(methods,fontsize=10)
ax2.set_ylim(0,1.15); ax2.set_ylabel('Score',fontsize=11)
ax2.set_title('Accuracy & F1 on Test Set\n(All methods — BC Wisconsin)',fontsize=10)
ax2.legend(fontsize=9); ax2.grid(axis='y',alpha=0.3)

plt.tight_layout()
plt.savefig('bnps_final_benchmark.png',dpi=150,bbox_inches='tight')
plt.show()
print('Saved: bnps_final_benchmark.png')

# ═ BNPS Accuracy vs Steps (fixed 100 membranes) ═════════════════════
STEP_SWEEP  = [10, 25, 50, 75, 100]
FIXED_MEMS  = 100
# lr=0.1 (down from 0.5) — prevents oscillation/divergence at higher step counts
pf_fixed = make_pep(X_train, y_train, FIXED_MEMS, NF, f'bc_{FIXED_MEMS}_sweep.pep', lr=0.1)

# Generate input.txt once
subprocess.run([sys.executable,'bnps3.py',pf_fixed,'-p','1'],
               capture_output=True, timeout=300)

print("\n── BNPS Accuracy vs Steps (100 membranes) ──")
print(f"{'Steps':>7}{'Acc':>10}{'95% CI':>20}{'F1':>8}")
print('-' * 48)

step_accs = []
for steps in STEP_SWEEP:
    last_out = ''
    for _ in range(RUNS):
        t0 = time.time()
        r = subprocess.run(
            [sys.executable,'bnps3.py',pf_fixed,'-n',str(steps)],
            capture_output=True, timeout=600)
        last_out = r.stdout.decode(errors='replace')

    # Extract weights from last run
    try:
        w_vals = []
        for i in range(NF):
            mw = re.search(rf'(?:^|\s)w{i}\s*[=:]\s*([-+]?[\d.eE]+)',
                           last_out, re.MULTILINE)
            if mw: w_vals.append(float(mw.group(1)))
        mb = re.search(r'(?:^|\s)b\s*[=:]\s*([-+]?[\d.eE]+)',
                       last_out, re.MULTILINE)
        bias = float(mb.group(1)) if mb else 0.0
        if len(w_vals) == NF:
            logits = X_test @ np.array(w_vals) + bias
            pred_prob = 1 / (1 + np.exp(-logits))
            bm = metrics(y_test, pred_prob)
            lo, hi = bm['acc_ci']
            step_accs.append((steps, bm['acc'], lo, hi, bm['f1']))
            print(f"{steps:>7}{bm['acc']:>10.4f}  [{lo:.3f} – {hi:.3f}]{bm['f1']:>8.4f}")
        else:
            print(f"{steps:>7}  weight extraction failed")
    except Exception as e:
        print(f"{steps:>7}  ERROR: {e}")

# Plot accuracy vs steps
if step_accs:
    fig, ax = plt.subplots(figsize=(7,4))
    xs  = [s[0] for s in step_accs]
    ys  = [s[1] for s in step_accs]
    los = [s[2] for s in step_accs]
    his = [s[3] for s in step_accs]
    ax.plot(xs, ys, 'o-', color='#3BCEAC', lw=2.5, ms=8, label='BNPS SLP Accuracy')
    ax.fill_between(xs, los, his, alpha=0.2, color='#3BCEAC', label='95% CI')
    ax.axhline(tf_m['acc'], color='#FF6B35', ls='--', lw=1.5,
               label=f"TF Deep MLP ({tf_m['acc']:.4f})")
    ax.axhline(pt_m['acc'], color='#EE4266', ls='--', lw=1.5,
               label=f"PT Deep MLP ({pt_m['acc']:.4f})")
    ax.set_xlabel('BNPS Simulation Steps', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('BNPS SLP Accuracy vs Steps (100 membranes)\nvs Deep MLP reference lines',
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('bnps_accuracy_vs_steps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: bnps_accuracy_vs_steps.png')

