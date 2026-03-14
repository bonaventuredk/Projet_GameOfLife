import subprocess
import time
import numpy as np
import os
import sys
import shutil

# ------------------------------------------------------------
# Implémentation séquentielle de référence (sans MPI)
# ------------------------------------------------------------
class GrilleSequentielle:
    def __init__(self, dim_global, init_pattern=None):
        self.ny, self.nx = dim_global
        self.cells = np.zeros((self.ny, self.nx), dtype=np.uint8)
        if init_pattern:
            for (i, j) in init_pattern:
                self.cells[i, j] = 1
        else:
            self.cells = np.random.randint(2, size=(self.ny, self.nx), dtype=np.uint8)

    def compute_next_iteration(self):
        new_cells = np.zeros_like(self.cells)
        for i in range(self.ny):
            for j in range(self.nx):
                voisines = [
                    self.cells[(i-1)%self.ny, (j-1)%self.nx],
                    self.cells[(i-1)%self.ny, j],
                    self.cells[(i-1)%self.ny, (j+1)%self.nx],
                    self.cells[i, (j-1)%self.nx],
                    self.cells[i, (j+1)%self.nx],
                    self.cells[(i+1)%self.ny, (j-1)%self.nx],
                    self.cells[(i+1)%self.ny, j],
                    self.cells[(i+1)%self.ny, (j+1)%self.nx]
                ]
                nb = sum(voisines)
                if self.cells[i, j] == 1:
                    if nb in (2, 3):
                        new_cells[i, j] = 1
                else:
                    if nb == 3:
                        new_cells[i, j] = 1
        self.cells = new_cells

def run_sequentiel(dim_global, pattern, iterations, output_file):
    """Exécute la simulation séquentielle et sauvegarde la grille finale."""
    grid = GrilleSequentielle(dim_global, pattern)
    for _ in range(iterations):
        grid.compute_next_iteration()
    np.save(output_file, grid.cells)



# ------------------------------------------------------------
# Lancement d'une version parallèle MPI
# ------------------------------------------------------------
def run_mpi(script_path, np, motif, resx, resy, iterations, output_file):
    """
    Lance la commande : mpiexec -n np python script.py motif resx resy iterations
    La grille finale doit être sauvegardée dans 'grid_final.npy' par le processus 0.
    """
    cmd = ["mpiexec", "-n", str(np), "python", script_path, motif, str(resx), str(resy), str(iterations)]
    if os.path.exists("grid_final.npy"):
        os.remove("grid_final.npy")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        result = None
    # Récupérer le fichier de sortie et le renommer
    if os.path.exists("grid_final.npy"):
        shutil.move("grid_final.npy", output_file)
        return elapsed, True
    else:
        return elapsed, False

# ------------------------------------------------------------
# Programme principal de test
# ------------------------------------------------------------
def main():
    # Paramètres du test
    motif = "glider"               # pattern à utiliser
    resx, resy = 800, 800           # résolution d'affichage (non utilisé pour le calcul)
    iterations = 10                 # nombre d'itérations

    # Dictionnaire des patterns (extrait du code original)
    dico_patterns = {
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat"    : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider"  : ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        # ... d'autres patterns peuvent être ajoutés
    }
    dim_global, pattern = dico_patterns[motif]

    # Chemins vers les scripts (à adapter selon votre arborescence)
    scripts = {
        "line"        : "./Line/LINE_gol.py",
        "column"      : "./Column /COLUMN_gol.py",
        "line_column" : "./Line and Column /LINE_COLUMN.py"
    }

    # Vérification de l'existence des scripts
    for name, path in scripts.items():
        if not os.path.isfile(path):
            print(f"Erreur : le script {path} est introuvable.")
            sys.exit(1)

    # Création d'un répertoire pour les résultats
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)

    # Exécution de la référence séquentielle
    seq_file = os.path.join(results_dir, "grid_seq.npy")
    print("Génération de la référence séquentielle...")
    run_sequentiel(dim_global, pattern, iterations, seq_file)
    print("Référence séquentielle terminée.\n")

    # Fichier de rapport
    rapport = os.path.join(results_dir, "resultats.txt")
    with open(rapport, "w") as f:
        f.write("RAPPORT DE TEST DE PERFORMANCE\n")
        f.write("==============================\n")
        f.write(f"Motif : {motif}\n")
        f.write(f"Grille globale : {dim_global[0]} x {dim_global[1]}\n")
        f.write(f"Itérations : {iterations}\n\n")
        f.write("{:<15} {:<15} {:<15} {:<10}\n".format("Version", "Processus", "Temps (s)", "Correct"))

    # Boucle de test pour chaque version et nombre de processus
    for version, script_path in scripts.items():
        for np in [2, 3, 4]:
            print(f"Test : {version} avec {np} processus...")
            out_file = os.path.join(results_dir, f"grid_{version}_{np}.npy")
            temps, ok = run_mpi(script_path, np, motif, resx, resy, iterations, out_file)

            if ok and os.path.exists(out_file):
                # Comparaison avec la référence
                ref = np.load(seq_file)
                test = np.load(out_file)
                correct = np.array_equal(ref, test)
            else:
                correct = False

            with open(rapport, "a") as f:
                f.write("{:<15} {:<15} {:<15.3f} {:<10}\n".format(version, np, temps, correct))
            print(f"  -> Temps = {temps:.3f}s, Correct = {correct}\n")

    print(f"\nTests terminés. Résultats enregistrés dans {rapport}")

if __name__ == "__main__":
    main()