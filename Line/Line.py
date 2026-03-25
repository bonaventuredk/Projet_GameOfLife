# Version parallélisée suivant les lignes
"""
    Membres du groupe:
    ------------------
        - Dohemeto Bonaventure 
        - Burns Thomas

Le jeu de la vie
################
"""
import pygame as pg

import numpy as np

import random

import time

import sys

from mpi4py import MPI

# ========================== Classes ==========================

class Grille:
    """Grille torique pour l'automate cellulaire."""
    def __init__(self, rank, worker_nbp, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        """
        Initialise une grille torique pour l'automate cellulaire.
        
        Paramètres:
            rank (int): Rang du processus MPI.
            worker_nbp (int): Nombre total de processus travailleurs.
            dim (tuple): Dimensions globales (ny_global, nx).
            init_pattern (list, optionnel): Liste de tuples (i, j) pour initialiser les cellules vivantes.
            color_life (pg.Color): Couleur des cellules vivantes.
            color_dead (pg.Color): Couleur des cellules mortes.
        """
        # Stockage des paramètres MPI
        self.rank = rank
        self.worker_nbp = worker_nbp
        ny_global, nx = dim
        
        # Calcul des dimensions locales pour la parallélisation par lignes
        base = ny_global // worker_nbp
        rest = ny_global % worker_nbp
        self.ny_loc = base + (1 if rank < rest else 0)
        self.y_start = rank * base + min(rank, rest)
        self.dimensions = (self.ny_loc + 2, nx) # 2: pour les 2 lignes fantômes

        # Initialisation du tableau de cellules
        self.cells = np.zeros(self.dimensions, dtype=np.uint8)

        # Initialisation des cellules selon le motif fourni ou aléatoirement
        if init_pattern is not None:
            for (i_glob, j_glob) in init_pattern:
                if self.y_start <= i_glob < self.y_start + self.ny_loc:
                    i_loc = i_glob - self.y_start + 1
                    self.cells[i_loc, j_glob] = 1
        else:
            self.cells[1:self.ny_loc+1, :] = np.random.randint(2, size=(self.ny_loc, nx), dtype=np.uint8)

        # Couleurs pour l'affichage
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        """
        Calcule la prochaine itération du jeu de la vie.
        
        Applique les règles du jeu de la vie à chaque cellule de la grille locale
        et retourne la liste des indices des cellules qui ont changé d'état.
        
        Retourne:
            list: Liste des indices globaux des cellules qui ont changé d'état.
        """
        # Récupération des dimensions réelles (sans les lignes fantômes)
        ny_reel = self.dimensions[0] - 2
        nx = self.dimensions[1]
        
        # Création d'un nouveau tableau pour la prochaine génération
        next_cells = np.zeros_like(self.cells)
        
        # Liste pour stocker les indices des cellules qui changent d'état
        diff = []

        # Parcours de toutes les cellules locales
        for i in range(1, ny_reel+1):
            for j in range(nx):
                # Comptage des 8 voisins
                voisines = [
                    self.cells[i-1, (j-1)%nx], self.cells[i-1, j], self.cells[i-1, (j+1)%nx],
                    self.cells[i,   (j-1)%nx],                     self.cells[i,   (j+1)%nx],
                    self.cells[i+1, (j-1)%nx], self.cells[i+1, j], self.cells[i+1, (j+1)%nx]
                ]
                nb_voisines = sum(voisines)

                # Application des règles du jeu de la vie
                if self.cells[i, j] == 1:
                    # Cellule vivante : survit si 2 ou 3 voisins
                    next_cells[i, j] = 1 if nb_voisines in (2,3) else 0
                    if nb_voisines not in (2,3):
                        diff.append((i-1)*nx + j)
                else:
                    # Cellule morte : naît si exactement 3 voisins
                    if nb_voisines == 3:
                        next_cells[i, j] = 1
                        diff.append((i-1)*nx + j)

        # Mise à jour de la grille avec la nouvelle génération
        self.cells = next_cells
        return diff

    def modificateur(self, diff):
        """
        Modifie l'état des cellules spécifiées dans la liste diff.
        
        Inverse l'état (vivant/mort) de chaque cellule dont l'indice est dans la liste.
        
        Paramètres:
            diff (list): Liste des indices des cellules à modifier.
        """
        # Récupération de la largeur de la grille
        nx = self.dimensions[1]
        
        # Pour chaque indice dans la liste des changements
        for c in diff:
            # Conversion de l'indice linéaire en coordonnées (ligne, colonne)
            nr = c // nx
            nc = c % nx

            # Inversion de l'état de la cellule (0->1 ou 1->0)
            self.cells[nr, nc] = 1 - self.cells[nr, nc]

    def exchange_ghost_lines(self, comm):
        """Échange les lignes fantômes verticales entre voisins MPI.
        
        L’objectif de cette méthode est de synchroniser les frontières avec les
        voisins du dessus et du dessous pour permettre un calcul correct des
        cellules bordures lors de l’itération du Jeu de la Vie.

        Paramètres:
            comm (MPI.Comm): Communicateur MPI des workers (représentant uniquement
                les processus de calcul, sans le rang d’affichage).

        Comportement:
            - si `size == 2`, on effectue deux `Sendrecv` séquentiels pour éviter
              blocages entre seulement deux processus.
            - sinon, on utilise `Isend` / `Irecv` + `Waitall` pour communication
              non bloquante entre plusieurs processus.
        """
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Si on a qu'un seul processus pour le calcul et l'affichage
        if size == 1:
            self.cells[0, :] = self.cells[-2, :]
            self.cells[-1, :] = self.cells[1, :]
            return

        # Calcul des indices des voisins circulaires (bords toroïdaux globalement)
        haut = (rank - 1) % size      # voisin du dessus
        bas = (rank + 1) % size   # voisin du dessous

        if size == 2:
            # On utilise Sendrecv pour éviter les deadlocks
            send_top = self.cells[-2, :]
            recv_top = np.empty(self.dimensions[1], dtype=np.uint8)
            comm.Sendrecv(sendbuf=send_top, dest=top, recvbuf=recv_top, source=top)
            self.cells[0, :] = recv_top

            send_bottom = self.cells[1, :]
            recv_bottom = np.empty(self.dimensions[1], dtype=np.uint8)
            comm.Sendrecv(sendbuf=send_bottom, dest=bottom, recvbuf=recv_bottom, source=bottom)
            self.cells[-1, :] = recv_bottom
        else:
            
            send_haut = self.cells[1, :].copy()
            recv_bas = np.empty(self.dimensions[1], dtype=np.uint8)

            comm.Sendrecv(
                sendbuf=send_haut,
                dest=haut,
                recvbuf=recv_bas,
                source=bas
            )

          
            send_bas = self.cells[-2,:].copy()
            recv_haut = np.empty(self.dimensions[1], dtype=np.uint8)

            comm.Sendrecv(
                sendbuf=send_bas,
                dest=bas,
                recvbuf=recv_haut,
                source=haut
            )

            # Mise à jour ghost cells
            self.cells[0,:] = recv_haut
            self.cells[-1,:] = recv_bas

# ========================== Application graphique ==========================

class App:
    def __init__(self, geometry, global_dim, color_life, color_dead):
        """
        Initialise l'application graphique pour afficher le jeu de la vie.
        
        Paramètres:
            geometry (tuple): Dimensions de la fenêtre (hauteur, largeur) en pixels.
            global_dim (tuple): Dimensions de la grille globale (ny_global, nx).
            color_life (pg.Color): Couleur RGB pour afficher les cellules vivantes.
            color_dead (pg.Color): Couleur RGB pour afficher les cellules mortes.
        """
        # Stockage des dimensions globales de la grille (nombre de lignes et colonnes)
        self.global_dim = global_dim
        
        # Stockage de la couleur des cellules vivantes
        self.color_life = color_life
        
        # Stockage de la couleur des cellules mortes
        self.color_dead = color_dead
        
        # Calcul de la largeur en pixels d'une cellule (résolution horizontale / nombre de colonnes)
        self.size_x = geometry[1] // global_dim[1]
        
        # Calcul de la hauteur en pixels d'une cellule (résolution verticale / nombre de lignes)
        self.size_y = geometry[0] // global_dim[0]
        
        # Couleur de la grille : gris clair si les cellules sont suffisamment grandes (>4px), sinon pas de grille
        self.draw_color = pg.Color('lightgrey') if self.size_x>4 and self.size_y>4 else None
        
        # Calcul de la largeur totale en pixels de la fenêtre
        self.width = global_dim[1] * self.size_x
        
        # Calcul de la hauteur totale en pixels de la fenêtre
        self.height = global_dim[0] * self.size_y
        
        # Création de la fenêtre Pygame avec les dimensions calculées
        self.screen = pg.display.set_mode((self.width, self.height))

    def compute_rectangle(self, i, j):
        """
        Calcule les coordonnées et dimensions du rectangle pour afficher une cellule.
        
        Paramètres:
            i (int): Indice de ligne dans la grille (0 = en haut).
            j (int): Indice de colonne dans la grille (0 = à gauche).
        
        Retourne:
            tuple: (x, y, width, height) pour dessiner le rectangle avec Pygame.
        """
        # Calcul de la position x : colonne j * taille horizontale d'une cellule
        x = self.size_x * j
        
        # Calcul de la position y : inversion verticale (y=0 en bas pour affichage inverse)
        y = self.height - self.size_y * (i + 1)
        
        # Largeur et hauteur du rectangle correspondent à la taille d'une cellule
        width = self.size_x
        height = self.size_y
        
        return (x, y, width, height)

    def compute_color(self, val):
        """
        Détermine la couleur à afficher selon l'état d'une cellule.
        
        Paramètres:
            val (int): Valeur de la cellule (1 pour vivante, 0 pour morte).
        
        Retourne:
            pg.Color: Couleur attribuée à l'état de la cellule.
        """
        # Retourne la couleur correspondant à l'état de la cellule
        return self.color_life if val else self.color_dead

    def draw_global(self, global_cells):
        """
        Affiche la grille globale avec les états des cellules sur l'écran.
        
        Paramètres:
            global_cells (np.ndarray): Matrice 2D contenant l'état de chaque cellule (1=vivante, 0=morte).
        """
        # Parcours de toutes les cellules de la grille globale
        for i in range(self.global_dim[0]):
            for j in range(self.global_dim[1]):
                # Remplissage du rectangle correspondant avec la couleur appropriée
                self.screen.fill(self.compute_color(global_cells[i,j]), self.compute_rectangle(i,j))
        
        # Affichage de la grille (lignes de démarcation entre les cellules) si suffisamment grande
        if self.draw_color is not None:
            # Dessinage des lignes horizontales de séparation
            for i in range(self.global_dim[0]):
                pg.draw.line(self.screen, self.draw_color, (0,i*self.size_y), (self.width,i*self.size_y))
            
            # Dessinage des lignes verticales de séparation
            for j in range(self.global_dim[1]):
                pg.draw.line(self.screen, self.draw_color, (j*self.size_x,0), (j*self.size_x,self.height))
        
        # Mise à jour de l'affichage Pygame
        pg.display.update()

# ========================== MPI ==========================

# Création d'une copie du communicateur global MPI pour éviter de modifier l'original
globCom = MPI.COMM_WORLD.Dup()

# Récupération du rang du processus courant (0 pour le maître, 1+ pour les workers)
rank = globCom.Get_rank()

# Récupération du nombre total de processus MPI disponibles
nbp = globCom.Get_size()

# Attribution d'une couleur : 0 pour le processus affichage (rank 0), 1 pour les workers
color = 0 if rank==0 else 1

# Utilisation du rang comme clé de tri au sein de chaque groupe de couleur
key = rank

# Division du communicateur global en deux groupes : affichage et calcul
# Les processus de même couleur formeront un nouveau communicateur
new_comm = globCom.Split(color, key)

# Création d'un communicateur pour les workers uniquement (rang 0 n'en a pas besoin)
worker_comm = new_comm if rank!=0 else None

# ========================== Main ==========================

if __name__ == '__main__':
    # Initialisation de Pygame
    pg.init()

    # Importations 
    import time
    import sys

    pg.init()

    # Dictionnaire des motifs (patterns) disponibles pour initialiser la grille globale
    dico_patterns = { # ... (identique à l'original)
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }
    choice = 'glider'
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3:
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    print(f"Pattern initial choisi : {choice}")
    print(f"resolution ecran : {resx,resy}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)
    dim, pattern = init_pattern

    # Si un seul processus, il fait à la fois le calcul et l'affichage.
    single_process = (nbp == 1)

    grid = None
    if single_process:
        # grille globale avec une seule partition
        grid = Grille(0, 1, dim, pattern)
    elif rank != 0:
        worker_rank = rank - 1
        worker_nbp = nbp - 1
        grid = Grille(worker_rank, worker_nbp, dim, pattern)

    # Création de l'application graphique pour le processus maître (et monoprocesseur)
    appli = App((resx,resy), dim, pg.Color("black"), pg.Color("white")) if rank==0 else None

    # Drapeau de boucle principale (arrêt lorsque la fenêtre est fermée)
    mustContinue = True


    # ========================== Boucle principale ==========================
    while mustContinue:
        # Mesure du temps total de l'itération (calcul + communication + affichage)
        t_total_start = time.time()

        # -------------------- Calcul --------------------
        if single_process:
            t_calc_start = time.time()
            grid.exchange_ghost_lines(globCom)
            grid.compute_next_iteration()
            t_calc_end = time.time()
            t_calc = t_calc_end - t_calc_start
            t_calc_total = t_calc

            # Pas de communication MPI pour un seul processus
            local_data = None
            local_nrows = grid.dimensions[0] - 2
        else:
            if rank != 0:
                t_calc_start = time.time()
                grid.exchange_ghost_lines(worker_comm)
                grid.compute_next_iteration()
                t_calc_end = time.time()
                t_calc = t_calc_end - t_calc_start

                local_data = grid.cells[1:grid.dimensions[0]-1, :].flatten()
                local_nrows = grid.dimensions[0]-2
            else:
                t_calc = 0
                local_data = np.array([], dtype=np.uint8)
                local_nrows = 0

            # Chaque worker a t_calc, rank 0 veut le max
            t_calc_scalar = np.array(t_calc, dtype=np.float64)
            t_calc_total = np.array(0.0, dtype=np.float64)
            globCom.Reduce(t_calc_scalar, t_calc_total, op=MPI.MAX, root=0)

            # -------------------- Rassemblement --------------------
            all_nrows = np.zeros(nbp, dtype=int) if rank==0 else None
            globCom.Gather(np.array([local_nrows],dtype=int), all_nrows, root=0)

            nx = dim[1]
            if rank==0:
                recvcounts = all_nrows * nx
                displs = np.zeros(nbp,dtype=int)
                for i in range(1,nbp):
                    displs[i] = displs[i-1] + recvcounts[i-1]
                total_cells = displs[-1] + recvcounts[-1]
                global_cells_flat = np.zeros(total_cells, dtype=np.uint8)
            else:
                recvcounts = None
                displs = None
                global_cells_flat = None

            globCom.Gatherv(local_data, [global_cells_flat, recvcounts, displs, MPI.UINT8_T], root=0)

        # -------------------- Affichage --------------------
        if single_process:
            t_display_start = time.time()
            appli.draw_global(grid.cells[1:grid.dimensions[0]-1, :])
            t_display_end = time.time()
            t_affichage = t_display_end - t_display_start

            # Gestion des événements dans le même processus
            for event in pg.event.get():
                if event.type==pg.QUIT:
                    mustContinue=False

            print(f"[Itération] Temps calcul : {t_calc_total:.6e}s | Temps affichage : {t_affichage:.6e}s")
        else:
            if rank==0:
                nx = dim[1]
                global_cells = np.zeros(dim, dtype=np.uint8)
                for i in range(1, nbp):
                    start_row = displs[i] // nx
                    end_row = start_row + all_nrows[i]
                    global_cells[start_row:end_row,:] = global_cells_flat[displs[i]:displs[i]+recvcounts[i]].reshape((all_nrows[i], nx))

                t_display_start = time.time()
                appli.draw_global(global_cells)
                t_display_end = time.time()
                t_affichage = t_display_end - t_display_start

                # -------------------- Événements --------------------
                for event in pg.event.get():
                    if event.type==pg.QUIT:
                        mustContinue=False

                # Affichage des temps
                print(f"[Itération] Temps calcul (workers) : {t_calc_total:.6e}s | Temps affichage : {t_affichage:.6e}s")

        # -------------------- Temps total --------------------
        t_total_end = time.time()
        t_total = t_total_end - t_total_start
        if rank==0 or single_process:
            print(f"[Itération] Temps total : {t_total:.6e}s")

        # Diffusion de mustContinue à tous (inutile en monoprocesseur mais sans effet)
        mustContinue = globCom.bcast(mustContinue, root=0)

    pg.quit()