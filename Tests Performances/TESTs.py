import subprocess

print("Lancement de sc1.py...")
subprocess.run(["python3", "./Tests/sc1.py"], check=True)

print("Lancement de sc3.py...")
subprocess.run(["python3", "./Tests/sc3.py"], check=True)

print("Lancement de sc4.py...")
subprocess.run(["python3", "./Tests/sc4.py"], check=True)

print("Lancement de sc2.py...")
subprocess.run(["python3", "./Tests/sc2.py"], check=True)

print("Lancement de sc5.py...")
subprocess.run(["python3", "./Tests/sc5.py"], check=True)

print("Les scripts ont été exécutés.")