import subprocess, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
ENV = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}

PATTERN, N, PROCS = "glider_gun", 20, [3, 4, 5, 6, 7, 8]

SCRIPTS = {
    "Sequential": ("Sequential/seq.py", False),
    "Line":        ("Line/Line.py",                    True),
    "Column":      ("Column/Column.py",                True),
    "LineAndCol":  ("LineAndColumn/LINE_COLUMN.py",    True),
}

RE = re.compile(r"(\d[\d.e+\-]+)", re.I)

def get_times(script, parallel, n_procs=1):
    cmd = (["mpirun", "-n", str(n_procs), "python3", script, PATTERN, "800", "800"]
           if parallel else ["python3", script, PATTERN, "800", "800", str(N)])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=ENV)
    totals, count = [], 0
    for line in proc.stdout:
        if "calcul" in line.lower() and "affichage" in line.lower():
            nums = RE.findall(line)
            if len(nums) >= 2:
                totals.append(float(nums[0]) + float(nums[1]))
                count += 1
                if count >= N: break
    proc.kill(); proc.wait()
    return np.mean(totals) if totals else 0

# ── Run ───────────────────────────────────────────────────────────────────────
results = {}
for name, (script, parallel) in SCRIPTS.items():
    if not parallel:
        results[name] = {1: get_times(script, False)}
        print(f"{name}: {results[name][1]:.3e}s")
    else:
        results[name] = {}
        for p in PROCS:
            t = get_times(script, True, p)
            results[name][p] = t
            print(f"{name} {p}p: {t:.3e}s  speedup={results['Sequential'][1]/t:.2f}x")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = {"Line": "#1E88E5", "Column": "#43A047", "LineAndCol": "#FB8C00"}
seq_t = results["Sequential"][1]

for name, col in colors.items():
    times    = [results[name][p] for p in PROCS]
    speedups = [seq_t / t for t in times]
    ax1.plot(PROCS, times,    "o-", color=col, lw=2, ms=7, label=name)
    ax2.plot(PROCS, speedups, "o-", color=col, lw=2, ms=7, label=name)

ax1.axhline(seq_t, color="red", ls="--", lw=1.5, label=f"Sequential ({seq_t:.2e}s)")
ax2.plot(PROCS, [p - 1 for p in PROCS], "k--", lw=1, alpha=0.5, label="Idéal")

for ax, title, ylabel in [(ax1, "Temps total", "Secondes"), (ax2, "Speedup", "×")]:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Nombre de processus MPI"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(PROCS)

plt.tight_layout()
plt.savefig("benchmark.png", dpi=150)
print("✅ benchmark.png sauvegardé")