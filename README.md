# Projet_GameOfLife
Un groupe de processus qui s'occupe de l'affichage et un autre pour les calculs.

# Auteurs

- DOHEMETO Bonaventure
- BURNS Thomas


# Game of Life MPI Performance Benchmark

Ce dépôt contient des scripts Python pour tester les performances de différentes implémentations parallèles du **Game of Life** en utilisant `MPI` via `mpiexec`.

## Contenu du projet

- `Line/LINE_gol.py` – Implémentation ligne par ligne  
- `Column/COLUMN_gol.py` – Implémentation colonne par colonne  
- `LineAndColumn/LINE_COLUMN.py` – Implémentation mixte lignes et colonnes  
- `gol_performance.txt` – Tableau des temps moyens des tests MPI  
- `gol_performance.png` – Graphique des performances MPI  

## Description du script de benchmark

Le script principal exécute les tests de performance pour différents nombres de processus (`N proc`) et mesure le temps moyen d'exécution pour chaque implémentation.

### Paramètres testés

- **Pattern** : `glider`  
- **Résolution** : 200 × 200  
- **Itérations** : 10  
- **Nombre de processus** : 2 à 12 (1 pour l'affichage + N-1 workers)  

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
