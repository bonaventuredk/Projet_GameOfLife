import subprocess

# Exécuter test1.py
print("Lancement de test1.py...")
subprocess.run(["python3", "test1.py"], check=True)

# Exécuter test2.py
print("Lancement de test2.py...")
subprocess.run(["python3", "test2.py"], check=True)

print("Les deux scripts ont été exécutés.")