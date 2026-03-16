import subprocess
import time
import matplotlib
matplotlib.use('Agg')  # Backend non interactif pour éviter les erreurs Qt/Snap
import matplotlib.pyplot as plt

# Scripts à tester
scripts = {
    "seq": "./Sequential/seq.py",
    "line": "./Line/LINE_gol.py",
    "column": "./Column/COLUMN_gol.py",
    "line_column": "./LineAndColumn/LINE_COLUMN.py"
}

# Paramètres
pattern = "glider"
resx, resy = 200, 200
iterations = 10
n_process_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 1 processus affichage + N-1 workers

results = {}

# Exécution des tests
for nproc in n_process_list:
    results[nproc] = {}
    print(f"\n=== Test avec {nproc} processus ===")
    
    for name, script_path in scripts.items():
        times = []
        for it in range(iterations):
            start_time = time.time()
            subprocess.run(
                ["mpiexec", "-n", str(nproc), "python", script_path, pattern, str(resx), str(resy)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        avg_time = sum(times) / len(times)
        results[nproc][name] = avg_time
        print(f"{name}: {avg_time:.6f}s (moyenne sur {iterations} itérations)")

# Création du tableau final
header = ["N proc"] + list(scripts.keys())
row_format = "{:<8}" + "{:<15}" * len(scripts)

# Affichage au terminal
print("\n=== Tableau des performances (temps moyen en secondes) ===")
print(row_format.format(*header))
for nproc in n_process_list:
    row = [str(nproc)] + [f"{results[nproc][name]:.6f}" for name in scripts.keys()]
    print(row_format.format(*row))

# Sauvegarde dans un fichier TXT
txt_file = "gol_performance.txt"
with open(txt_file, "w") as f:
    f.write("=== Tableau des performances (temps moyen en secondes) ===\n")
    f.write(row_format.format(*header) + "\n")
    for nproc in n_process_list:
        row = [str(nproc)] + [f"{results[nproc][name]:.6f}" for name in scripts.keys()]
        f.write(row_format.format(*row) + "\n")

print(f"\nTableau enregistré dans {txt_file}")

# ------------------------
# Création et sauvegarde du plot
# ------------------------
plt.figure(figsize=(10, 6))
for name in scripts.keys():
    y = [results[nproc][name] for nproc in n_process_list]
    plt.plot(n_process_list, y, marker='o', label=name)

plt.xlabel("Nombre de processus")
plt.ylabel("Temps moyen (s)")
plt.title("Performance MPI des scripts Game of Life")
plt.xticks(n_process_list)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

plot_file = "gol_performance.png"
plt.savefig(plot_file)
print(f"Plot sauvegardé dans {plot_file}")
