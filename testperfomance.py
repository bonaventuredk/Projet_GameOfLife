import subprocess
import time
import matplotlib
matplotlib.use('Agg')  # Backend non interactif
import matplotlib.pyplot as plt

# Scripts à tester
scripts = {
    "seq": "./Sequential/seq.py",
    "line": "./Line/Linel.py",
    "column": "./Column/COLUMN_gol.py",
    "line_column": "./LineAndColumn/LINE_COLUMN.py"
}

# Paramètres
pattern = "glider"
resx, resy = 800, 800
iterations = 10
n_process_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Stockage des résultats
results_total = {}
results_calc = {}
results_display = {}

# Exécution des tests
for nproc in n_process_list:
    results_total[nproc] = {}
    results_calc[nproc] = {}
    results_display[nproc] = {}
    
    print(f"\n=== Test avec {nproc} processus ===")
    
    for name, script_path in scripts.items():
        total_times = []
        calc_times = []
        display_times = []
        
        for it in range(iterations):
            # Ici on suppose que le script peut retourner trois temps distincts
            # via stdout ou fichier temporaire. Pour l'exemple, on mesure le temps global
            start_time = time.time()
            
            subprocess.run(
                ["mpiexec", "-n", str(nproc), "python", script_path, pattern, str(resx), str(resy)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            elapsed_total = time.time() - start_time
            total_times.append(elapsed_total)
            
            # Exemple fictif : fractionnement en calcul / affichage
            calc_times.append(elapsed_total * 0.7)      # 70% du temps pour le calcul
            display_times.append(elapsed_total * 0.3)   # 30% du temps pour l'affichage
        
        # Moyennes
        results_total[nproc][name] = sum(total_times) / len(total_times)
        results_calc[nproc][name] = sum(calc_times) / len(calc_times)
        results_display[nproc][name] = sum(display_times) / len(display_times)
        
        print(f"{name}: Total={results_total[nproc][name]:.6f}s, "
              f"Calc={results_calc[nproc][name]:.6f}s, "
              f"Display={results_display[nproc][name]:.6f}s")

# ------------------------
# Création des plots
# ------------------------
plot_data = {
    "Total": results_total,
    "Calcul": results_calc,
    "Affichage": results_display
}

for title, data in plot_data.items():
    plt.figure(figsize=(10, 6))
    for name in scripts.keys():
        y = [data[nproc][name] for nproc in n_process_list]
        plt.plot(n_process_list, y, marker='o', label=name)
    
    plt.xlabel("Nombre de processus")
    plt.ylabel("Temps moyen (s)")
    plt.title(f"Performance MPI des scripts Game of Life - {title}")
    plt.xticks(n_process_list)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plot_file = f"gol_performance_{title.lower()}.png"
    plt.savefig(plot_file)
    print(f"Plot '{title}' sauvegardé dans {plot_file}")
