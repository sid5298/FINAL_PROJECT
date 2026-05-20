# ─────────────────────────────────────────────────────────────────────────────
# BNPS Colab Runner
# Upload and run .py / .cu / .pep files directly in Google Colab
# ─────────────────────────────────────────────────────────────────────────────

import os, subprocess, sys
from google.colab import files

# ── 1. Upload files ───────────────────────────────────────────────────────────
print("📂 Upload your .py / .cu / .pep files (you can select multiple):")
uploaded = files.upload()

if not uploaded:
    print("❌ No files uploaded.")
else:
    for filename in uploaded:
        ext = os.path.splitext(filename)[1].lower()
        print(f"\n{'='*60}")
        print(f"📄 File: {filename}  ({ext})")
        print('='*60)

        # ── .pep → run with bnps3.py ─────────────────────────────────────────
        if ext == '.pep':
            # Check if bnps3.py is present; if not, ask user to upload it too
            if not os.path.exists('bnps3.py'):
                print("⚠️  bnps3.py not found in current dir.")
                print("    Please also upload bnps3.py (your BNPS simulator).")
                print("    Re-run this cell after uploading bnps3.py.")
            else:
                steps = input("   How many simulation steps? [default: 10]: ").strip()
                steps = steps if steps.isdigit() else "10"
                mode  = input("   Mode — serial (s) or parallel (p)? [default: s]: ").strip().lower()
                flag  = '-p' if mode == 'p' else '-n'
                cmd   = [sys.executable, 'bnps3.py', filename, flag, steps]
                print(f"\n▶  Running: {' '.join(cmd)}\n")
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("STDERR:\n", result.stderr)

        # ── .py → run directly ───────────────────────────────────────────────
        elif ext == '.py':
            cmd = [sys.executable, filename]
            print(f"\n▶  Running: {' '.join(cmd)}\n")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:\n", result.stderr)

        # ── .cu → compile with nvcc then run ────────────────────────────────
        elif ext == '.cu':
            binary = filename.replace('.cu', '')
            # Check nvcc
            nvcc_check = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
            if not nvcc_check.stdout.strip():
                print("⚠️  nvcc not found. Installing CUDA toolkit...")
                subprocess.run(['apt-get', 'install', '-y', 'nvidia-cuda-toolkit'],
                               capture_output=True)

            print(f"\n🔨 Compiling: nvcc {filename} -o {binary}")
            compile_result = subprocess.run(
                ['nvcc', filename, '-o', binary],
                capture_output=True, text=True
            )
            if compile_result.returncode != 0:
                print("❌ Compilation failed:\n", compile_result.stderr)
            else:
                print(f"✅ Compiled successfully → ./{binary}")
                print(f"\n▶  Running: ./{binary}\n")
                run_result = subprocess.run([f'./{binary}'], capture_output=True, text=True)
                print(run_result.stdout)
                if run_result.stderr:
                    print("STDERR:\n", run_result.stderr)

        else:
            print(f"⚠️  Unknown extension '{ext}'. Skipping.")

print("\n✅ Done.")
