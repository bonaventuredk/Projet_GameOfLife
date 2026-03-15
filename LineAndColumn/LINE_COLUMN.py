"""
Le jeu de la vie
################
...
"""
import pygame as pg
import numpy as np
from mpi4py import MPI
import time
import sys
import math

# ---------- Fonction utilitaire pour déterminer une grille 2D équilibrée ----------
def get_2d_dims(nproc):
    """
    Retourne un couple (n_rows, n_cols) tel que n_rows * n_cols == nproc,
    avec une répartition aussi équilibrée que possible.
    """
    if nproc == 1:
        return (1, 1)
    # Chercher la factorisation la plus équilibrée
    root = int(math.sqrt(nproc))
    for r in range(root, 0, -1):
        if nproc % r == 0:
            return (r, nproc // r)
    # Si pas de diviseur parfait (ne devrait pas arriver car nproc est entier)
    return (1, nproc)

# ---------- Classe Grille avec décomposition 2D ----------
class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    Version parallélisée 2D : chaque worker possède un bloc rectangulaire
    avec des lignes et colonnes fantômes pour les échanges.
    """
    def __init__(self, cart_comm, dim_global, init_pattern=None,
                 color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.cart_comm = cart_comm
        self.rank = cart_comm.Get_rank()
        self.nproc = cart_comm.Get_size()
        self.dim_global = dim_global
        ny_global, nx_global = dim_global

        # Récupérer les coordonnées du processus dans la grille cartésienne
        self.coords = cart_comm.Get_coords(self.rank)
        self.row, self.col = self.coords

        # Dimensions de la grille de processus
        dims = cart_comm.Get_topo()
        self.n_rows, self.n_cols = dims[0]  # dims = (n_rows, n_cols, periods)

        # Distribution des lignes
        base_y = ny_global // self.n_rows
        rest_y = ny_global % self.n_rows
        self.ny_loc = base_y + (1 if self.row < rest_y else 0)
        self.y_start = self.row * base_y + min(self.row, rest_y)

        # Distribution des colonnes
        base_x = nx_global // self.n_cols
        rest_x = nx_global % self.n_cols
        self.nx_loc = base_x + (1 if self.col < rest_x else 0)
        self.x_start = self.col * base_x + min(self.col, rest_x)

        # Dimensions locales avec fantômes (halo)
        self.dimensions = (self.ny_loc + 2, self.nx_loc + 2)
        self.cells = np.zeros(self.dimensions, dtype=np.uint8)

        # Initialisation
        if init_pattern is not None:
            for (i_glob, j_glob) in init_pattern:
                if (self.y_start <= i_glob < self.y_start + self.ny_loc and
                    self.x_start <= j_glob < self.x_start + self.nx_loc):
                    i_loc = i_glob - self.y_start + 1
                    j_loc = j_glob - self.x_start + 1
                    self.cells[i_loc, j_loc] = 1
        else:
            # Initialisation aléatoire sur la partie réelle
            self.cells[1:self.ny_loc+1, 1:self.nx_loc+1] = np.random.randint(
                2, size=(self.ny_loc, self.nx_loc), dtype=np.uint8
            )

        self.col_life = color_life
        self.col_dead = color_dead

        # Déterminer les rangs des voisins (pour les échanges)
        # Voisins haut / bas (direction 0)
        self.top_rank = self.cart_comm.Shift(0, -1)[0]   # source pour déplacement -1
        self.bottom_rank = self.cart_comm.Shift(0, 1)[0] # source pour déplacement +1
        # Voisins gauche / droite (direction 1)
        self.left_rank = self.cart_comm.Shift(1, -1)[0]
        self.right_rank = self.cart_comm.Shift(1, 1)[0]

    def compute_next_iteration(self):
        """Calcule l'état suivant en utilisant les fantômes déjà à jour."""
        next_cells = np.zeros_like(self.cells)
        for i in range(1, self.ny_loc + 1):
            for j in range(1, self.nx_loc + 1):
                # Les voisins sont accessibles directement grâce aux fantômes
                voisines = [
                    self.cells[i-1, j-1], self.cells[i-1, j], self.cells[i-1, j+1],
                    self.cells[i,   j-1],                     self.cells[i,   j+1],
                    self.cells[i+1, j-1], self.cells[i+1, j], self.cells[i+1, j+1]
                ]
                nb_voisines = sum(voisines)

                if self.cells[i, j] == 1:
                    if nb_voisines in (2, 3):
                        next_cells[i, j] = 1
                else:
                    if nb_voisines == 3:
                        next_cells[i, j] = 1

        self.cells = next_cells
        # On ne maintient pas de liste diff (inutilisée)
        return []
    def exchange_ghosts(self):
        # Cas particulier d'un seul worker
        if self.nproc == 1:
            self.cells[0, :] = self.cells[self.ny_loc, :]
            self.cells[self.ny_loc+1, :] = self.cells[1, :]
            self.cells[:, 0] = self.cells[:, self.nx_loc]
            self.cells[:, self.nx_loc+1] = self.cells[:, 1]
            return

        reqs = []

        # Échanges verticaux (haut/bas)
        send_top = self.cells[1, :].copy()
        recv_top = np.empty(self.nx_loc + 2, dtype=np.uint8)
        reqs.append(self.cart_comm.Irecv(recv_top, source=self.top_rank, tag=0))
        reqs.append(self.cart_comm.Isend(send_top, dest=self.top_rank, tag=0))

        send_bottom = self.cells[self.ny_loc, :].copy()
        recv_bottom = np.empty(self.nx_loc + 2, dtype=np.uint8)
        reqs.append(self.cart_comm.Irecv(recv_bottom, source=self.bottom_rank, tag=0))
        reqs.append(self.cart_comm.Isend(send_bottom, dest=self.bottom_rank, tag=0))

        MPI.Request.Waitall(reqs)
        self.cells[0, :] = recv_top
        self.cells[self.ny_loc + 1, :] = recv_bottom

        # Échanges horizontaux (gauche/droite) après mise à jour des fantômes verticaux
        reqs = []
        send_left = self.cells[:, 1].copy()
        recv_left = np.empty(self.ny_loc + 2, dtype=np.uint8)
        reqs.append(self.cart_comm.Irecv(recv_left, source=self.left_rank, tag=1))
        reqs.append(self.cart_comm.Isend(send_left, dest=self.left_rank, tag=1))

        send_right = self.cells[:, self.nx_loc].copy()
        recv_right = np.empty(self.ny_loc + 2, dtype=np.uint8)
        reqs.append(self.cart_comm.Irecv(recv_right, source=self.right_rank, tag=1))
        reqs.append(self.cart_comm.Isend(send_right, dest=self.right_rank, tag=1))

        MPI.Request.Waitall(reqs)
        self.cells[:, 0] = recv_left
        self.cells[:, self.nx_loc + 1] = recv_right
    

# ---------- Classe App (inchangée) ----------
class App:
    def __init__(self, geometry, global_dim, color_life, color_dead):
        self.global_dim = global_dim
        self.color_life = color_life
        self.color_dead = color_dead
        self.size_x = geometry[1] // global_dim[1]
        self.size_y = geometry[0] // global_dim[0]
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None
        self.width = global_dim[1] * self.size_x
        self.height = global_dim[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))

    def compute_rectangle(self, i, j):
        return (self.size_x * j, self.height - self.size_y * (i + 1), self.size_x, self.size_y)

    def compute_color(self, val):
        return self.color_life if val else self.color_dead

    def draw_global(self, global_cells):
        for i in range(self.global_dim[0]):
            for j in range(self.global_dim[1]):
                self.screen.fill(self.compute_color(global_cells[i, j]),
                                 self.compute_rectangle(i, j))
        if self.draw_color is not None:
            for i in range(self.global_dim[0]):
                pg.draw.line(self.screen, self.draw_color, (0, i*self.size_y), (self.width, i*self.size_y))
            for j in range(self.global_dim[1]):
                pg.draw.line(self.screen, self.draw_color, (j*self.size_x, 0), (j*self.size_x, self.height))
        pg.display.update()

# ---------- Initialisation MPI ----------
globCom = MPI.COMM_WORLD.Dup()
rank = globCom.Get_rank()
nbp = globCom.Get_size()

# Séparation : processus 0 pour l'affichage, les autres pour le calcul
color = 0 if rank == 0 else 1
key = rank
new_comm = globCom.Split(color, key)
if rank != 0:
    worker_comm = new_comm
    # Création d'un communicateur cartésien pour les workers
    worker_nbp = worker_comm.Get_size()
    dims_2d = get_2d_dims(worker_nbp)
    periods = (True, True)  # tore dans les deux directions
    reorder = True
    worker_cart = worker_comm.Create_cart(dims_2d, periods, reorder)
else:
    worker_comm = None
    worker_cart = None

# ---------- Programme principal ----------
if __name__ == '__main__':
    pg.init()

    # Dictionnaire des patterns (identique)
    dico_patterns = {
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
    dim_global, pattern = init_pattern
    ny_global, nx_global = dim_global

    # Création des grilles pour les workers (décomposition 2D)
    if rank != 0:
        grid = Grille(worker_cart, dim_global, pattern,
                      color_life=pg.Color("black"), color_dead=pg.Color("white"))
    else:
        grid = None

    if rank == 0:
        appli = App((resx, resy), dim_global, pg.Color("black"), pg.Color("white"))
    else:
        appli = None

    mustContinue = True
    # Premier échange de fantômes pour initialiser les halos
    if rank != 0:
        grid.exchange_ghosts()

    while mustContinue:
        if rank != 0:
            # Mesure du temps de calcul
            t1 = time.time()
            grid.exchange_ghosts()
            grid.compute_next_iteration()
            t2 = time.time()
            print(f"Worker {rank} compute time: {t2-t1:.2e} secondes")

            # Préparation des données locales (sans les fantômes)
            local_data = grid.cells[1:grid.ny_loc+1, 1:grid.nx_loc+1].flatten()
            # Métadonnées : y_start, ny_loc, x_start, nx_loc
            meta = np.array([grid.y_start, grid.ny_loc, grid.x_start, grid.nx_loc], dtype=int)
        else:
            local_data = np.array([], dtype=np.uint8)
            meta = np.zeros(4, dtype=int)  # placeholder

        # Rassemblement des métadonnées
        all_meta = None
        if rank == 0:
            all_meta = np.zeros((nbp, 4), dtype=int)
        globCom.Gather(meta, all_meta, root=0)

        # Communication des blocs par point-à-point (car Gatherv ne convient pas pour des blocs 2D)
        if rank == 0:
            # Reconstruction de la grille globale
            global_cells = np.zeros((ny_global, nx_global), dtype=np.uint8)
            # Réception des blocs de chaque worker
            for i in range(1, nbp):
                ys, nyl, xs, nxl = all_meta[i]
                # Recevoir le bloc du worker i
                block_data = np.empty(nyl * nxl, dtype=np.uint8)
                globCom.Recv(block_data, source=i, tag=100)
                # Remettre en forme 2D et copier dans la grille globale
                bloc = block_data.reshape((nyl, nxl))
                global_cells[ys:ys+nyl, xs:xs+nxl] = bloc
        else:
            # Envoyer le bloc local au processus 0
            globCom.Send(local_data, dest=0, tag=100)

        if rank == 0:
            # Affichage
            t3 = time.time()
            appli.draw_global(global_cells)
            t4 = time.time()
            print(f"Affichage : {t4-t3:.2e} secondes")

            # Gestion des événements
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False

        # Diffusion de mustContinue à tous les processus
        mustContinue = globCom.bcast(mustContinue, root=0)

    pg.quit()