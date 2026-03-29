# Projet_GameOfLife
Un groupe de processus qui s'occupe de l'affichage et un autre pour les calculs.

# Auteurs

- DOHEMETO Bonaventure
- BURNS Thomas


# Game of Life MPI Performance Benchmark

Ce dépôt contient des scripts Python pour tester les performances de différentes implémentations parallèles du **Game of Life** en utilisant `MPI` via `mpiexec`.

## Contenu du projet

- `Avec_Sendrecv/Line.py` – Implémentation ligne par ligne  ( avec SendRecv)
- `Sans_Sendrecv/Line2.py` – Implémentation ligne par ligne  ( sans SendRecv) 
- `Avec_Sendrecv/Column.py` – Implémentation colonne par colonne  ( avec SendRecv)  
- `Sans_Sendrecv/Column2.py` – Implémentation colonne par colonne ( sans SendRecv)
- `Avec_Sendrecv/LC.py` – Implémentation mixte ( avec SendRecv)
- `Sans_Sendrecv/LC2.py` – Implémentation mixte ( sans SendRecv)
- `gol_performance.txt` – Tableau des temps moyens des tests MPI  
- `1_comparaison_temps_moyens.png` – Graphique sans la version sequentielle
- `1_comparaison_temps_moyens_seq.png` – Graphique avec la version sequentielle
- `2_comparaison_temps_moyens.png` – Graphique sans la version sequentielle
- `2_comparaison_temps_moyens_seq.png` – Graphique avec la version sequentielle
- `TESTs.py` - Pour lancer tous les tests de performances (avoir les temps moyens, les plots, le speed-up, l'efficiency)
-  <span style="color:red;">`IMPORTANT 🔴🔴🔴!`</span> - Pour lancer le script TESTs.py, il faut aller dans les scripts sc1.py et sc3.py dans le dossier /Testsperformances et remplacer les valeurs de la liste suivante (procs_list = [3,4,5,6,7,8]), afin que le maximum des éléments ne dépasse pas le nombre de processus que votre PC peut lancer. Nous avons utilisé un max de 8 processus par défaut. Voici la partie à modifier au besoin:
 


## Description du script de benchmark

Le script principal exécute les tests de performance pour différents nombres de processus (`N proc`) et mesure le temps moyen d'exécution pour chaque implémentation.

### Paramètres testés

- **Pattern** : `glider`  
- **Résolution** : 200 × 200  
- **Itérations** : 10  
- **Nombre de processus** : 2 à 12 (1 pour l'affichage + N-1 workers)  




## 🖥️ Configuration matérielle sur lequel les test ont été réalisé (CPU)

Voici les détails de la machine utilisée pour le projet :

| Caractéristique | Détail |
|-----------------|--------|
| **Processeur** | AMD Ryzen 7 7730U with Radeon Graphics |
| **Architecture** | x86_64 (64-bit) |
| **Cœurs / Threads** | 8 cœurs / 16 threads |
| **Fréquence CPU** | 410 MHz – 2000 MHz (boost activé) |
| **Cache** | L1 : 256 KiB, L2 : 4 MiB, L3 : 16 MiB |
| **Virtualisation** | AMD-V supportée |
| **Endianness** | Little Endian |
| **Nombre de CPU en ligne** | 16 |
| **NUMA** | 1 nœud (CPU 0-15) |

## 🔒 Sécurité et vulnérabilités

## ⚡ Informations supplémentaires

- Modes CPU : 32-bit et 64-bit  
- Taille des adresses : 48 bits physiques / 48 bits virtuelles  
- BogoMIPS : 3992.71  
- Flags importants : SSE, SSE2, SSE4.1, SSE4.2, AVX, AVX2, FMA, AES, xsave, etc.

### Exécution

Pour chaque script et chaque nombre de processus, le script :

1. Lance le script via MPI (`mpiexec -n N python script.py pattern resx resy`)  
2. Mesure le temps d'exécution  
3. Calcule le temps moyen sur 10 itérations  

Les résultats sont affichés dans le terminal, sauvegardés dans `gol_performance.txt` et tracés dans `gol_performance.png`.

### Tableau de performance

| N proc | line     | column   | line_column |
| ------ | -------- | -------- | ----------- |
| 2      | 0.047774 | 0.047030 | 0.047222    |
| 3      | 0.048908 | 0.046176 | 0.046546    |
| 4      | 0.046877 | 0.045336 | 0.047212    |
| 5      | 0.047600 | 0.047676 | 0.047226    |
| 6      | 0.046189 | 0.046643 | 0.047578    |
| 7      | 0.048050 | 0.046693 | 0.047400    |
| 8      | 0.047692 | 0.047975 | 0.046646    |
| 9      | 0.047390 | 0.045339 | 0.044355    |
| 10     | 0.046153 | 0.044716 | 0.044324    |
| 11     | 0.045598 | 0.045370 | 0.044854    |
| 12     | 0.044734 | 0.046157 | 0.045549    |


### Graphique de performance

Le plot `gol_performance.png` montre l’évolution du temps moyen en fonction du nombre de processus pour chaque implémentation :

![Performance MPI](gol_performance.png)

## Exigences

- Python 3.x  
- `matplotlib`  
- MPI (OpenMPI ou MPICH)  
- Accès à un terminal avec `mpiexec`  

## Utilisation

1. Cloner le dépôt :  
```bash
git clone <url_du_repo>
cd <nom_du_repo>
