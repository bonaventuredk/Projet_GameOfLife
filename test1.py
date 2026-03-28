import subprocess
import shutil
import os

# Liste des scripts
scripts = [
    "./Line/Line.py",
    "./LineAndColumn/LC.py",
    "./LineAndColumn/LC2.py",
    "./Line/Line2.py",
    "./Column/Column.py",
    "./Column/Column2.py"
]

# Nombres de processus
procs_list = [3, 4, 8]

# Vérifier si xvfb-run est installé
xvfb_available = shutil.which('xvfb-run') is not None

for script in scripts:
    print(f"\n==============================")
    print(f" Script : {script}")
    print(f"==============================")
    
    for p in procs_list:
        print(f"\n--- {p} processus ---")
        
        # Construction de la commande
        if xvfb_available:
            # Préfixe avec xvfb-run pour masquer les fenêtres
            cmd = ["xvfb-run", "-a", "mpiexec", "-np", str(p), "python3", script]
        else:
            cmd = ["mpiexec", "-np", str(p), "python3", script]
            print("(Attention : xvfb-run non trouvé, les fenêtres pourraient s'afficher)")
        
        try:
            # Exécution avec timeout de 6 secondes
            subprocess.run(cmd, check=True, timeout=6)
        except subprocess.TimeoutExpired:
            print(f"✓ Exécution arrêtée après 6 secondes pour {script} ({p} processus).")
        except subprocess.CalledProcessError as e:
            print(f"Erreur avec {script} ({p} processus) : {e}")
