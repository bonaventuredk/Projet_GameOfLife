"""
    Version combinant la parallélisation par lignes et colonnes 
"""
import pygame as pg
import numpy as np
from mpi4py import MPI
from math import sqrt

def get_2d_grid_dims(n):
    """
    Retourne (p_rows, p_cols) tel que p_rows * p_cols = n et la grille
    soit la plus carrée possible.
    """
    best = (1, n)
    best_diff = n - 1
    for i in range(1, int(sqrt(n)) + 1):
        if n % i == 0:
            j = n // i
            diff = abs(j - i)
            if diff < best_diff:
                best = (i, j)
                best_diff = diff
    return best

class Grille2D:
    """
    Grille torique découpée en blocs 2D.
    Chaque worker reçoit un bloc de dimensions (ny_loc, nx_loc) et deux
    lignes/colonnes fantômes (haut, bas, gauche, droite).
    """
    def __init__(self, worker_comm, row, col, p_rows, p_cols,
                 global_dim, init_pattern=None,
                 color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.worker_comm = worker_comm
        self.row = row          # ligne dans la grille de processus
        self.col = col          # colonne dans la grille de processus
        self.p_rows = p_rows
        self.p_cols = p_cols

        ny_glob, nx_glob = global_dim

        # Distribution des lignes entre les p_rows workers
        base_y = ny_glob // p_rows
        rest_y = ny_glob % p_rows
        self.ny_loc = base_y + (1 if row < rest_y else 0)
        self.y_start = row * base_y + min(row, rest_y)

        # Distribution des colonnes entre les p_cols workers
        base_x = nx_glob // p_cols
        rest_x = nx_glob % p_cols
        self.nx_loc = base_x + (1 if col < rest_x else 0)
        self.x_start = col * base_x + min(col, rest_x)

        # Dimensions locales avec fantômes (haut, bas, gauche, droite)
        self.dimensions = (self.ny_loc + 2, self.nx_loc + 2)
        self.cells = np.zeros(self.dimensions, dtype=np.uint8)

        # Initialisation à partir du pattern ou aléatoire
        if init_pattern is not None:
            for (i_glob, j_glob) in init_pattern:
                if (self.y_start <= i_glob < self.y_start + self.ny_loc and
                    self.x_start <= j_glob < self.x_start + self.nx_loc):
                    i_loc = i_glob - self.y_start + 1   # +1 pour fantôme haut
                    j_loc = j_glob - self.x_start + 1   # +1 pour fantôme gauche
                    self.cells[i_loc, j_loc] = 1
        else:
            # Remplir aléatoirement la partie réelle
            self.cells[1:self.ny_loc+1, 1:self.nx_loc+1] = np.random.randint(
                2, size=(self.ny_loc, self.nx_loc), dtype=np.uint8
            )

        self.col_life = color_life
        self.col_dead = color_dead

        # Calcul des rangs des voisins dans worker_comm (ordre ligne majeure)
        # Voisins verticaux (haut/bas)
        top_row = (self.row - 1) % self.p_rows
        bottom_row = (self.row + 1) % self.p_rows
        self.top_rank = top_row * self.p_cols + self.col
        self.bottom_rank = bottom_row * self.p_cols + self.col

        # Voisins horizontaux (gauche/droite)
        left_col = (self.col - 1) % self.p_cols
        right_col = (self.col + 1) % self.p_cols
        self.left_rank = self.row * self.p_cols + left_col
        self.right_rank = self.row * self.p_cols + right_col

    def exchange_ghost(self):
        """Échange complet avec les 8 voisins (4 directs + 4 coins)"""
        
        size = self.worker_comm.Get_size()
        if size == 1:
            # Un seul worker : recopie circulaire interne
            # Lignes
            self.cells[0, :] = self.cells[self.ny_loc, :]
            self.cells[self.ny_loc+1, :] = self.cells[1, :]
            # Colonnes
            self.cells[:, 0] = self.cells[:, self.nx_loc]
            self.cells[:, self.nx_loc+1] = self.cells[:, 1]
            # Coins (copie circulaire)
            self.cells[0, 0] = self.cells[self.ny_loc, self.nx_loc]
            self.cells[0, self.nx_loc+1] = self.cells[self.ny_loc, 1]
            self.cells[self.ny_loc+1, 0] = self.cells[1, self.nx_loc]
            self.cells[self.ny_loc+1, self.nx_loc+1] = self.cells[1, 1]
            return

        # ===== 1. Échanges avec les 4 voisins directs =====
        # Préparation des buffers pour les échanges horizontaux
        send_right = self.cells[:, self.nx_loc].copy()
        recv_left = np.empty(self.dimensions[0], dtype=np.uint8)
        send_left = self.cells[:, 1].copy()
        recv_right = np.empty(self.dimensions[0], dtype=np.uint8)

        # Préparation des buffers pour les échanges verticaux
        send_bottom = self.cells[self.ny_loc, :].copy()
        recv_top = np.empty(self.dimensions[1], dtype=np.uint8)
        send_top = self.cells[1, :].copy()
        recv_bottom = np.empty(self.dimensions[1], dtype=np.uint8)

        # Liste des requêtes
        reqs = []

        # Communications horizontales
        reqs.append(self.worker_comm.Irecv(recv_left, source=self.left_rank, tag=0))
        reqs.append(self.worker_comm.Irecv(recv_right, source=self.right_rank, tag=1))
        reqs.append(self.worker_comm.Isend(send_left, dest=self.left_rank, tag=1))
        reqs.append(self.worker_comm.Isend(send_right, dest=self.right_rank, tag=0))

        # Communications verticales
        reqs.append(self.worker_comm.Irecv(recv_top, source=self.top_rank, tag=2))
        reqs.append(self.worker_comm.Irecv(recv_bottom, source=self.bottom_rank, tag=3))
        reqs.append(self.worker_comm.Isend(send_top, dest=self.top_rank, tag=3))
        reqs.append(self.worker_comm.Isend(send_bottom, dest=self.bottom_rank, tag=2))

        # Attendre la fin des communications directes
        MPI.Request.Waitall(reqs)

        # Mise à jour des fantômes directs
        self.cells[:, 0] = recv_left
        self.cells[:, self.nx_loc+1] = recv_right
        self.cells[0, :] = recv_top
        self.cells[self.ny_loc+1, :] = recv_bottom

        # ===== 2. Échanges avec les 4 coins diagonaux =====
        # Calcul des rangs des voisins en diagonale
        # Nord-Ouest
        nw_row = (self.row - 1) % self.p_rows
        nw_col = (self.col - 1) % self.p_cols
        nw_rank = nw_row * self.p_cols + nw_col
        
        # Nord-Est
        ne_row = (self.row - 1) % self.p_rows
        ne_col = (self.col + 1) % self.p_cols
        ne_rank = ne_row * self.p_cols + ne_col
        
        # Sud-Ouest
        sw_row = (self.row + 1) % self.p_rows
        sw_col = (self.col - 1) % self.p_cols
        sw_rank = sw_row * self.p_cols + sw_col
        
        # Sud-Est
        se_row = (self.row + 1) % self.p_rows
        se_col = (self.col + 1) % self.p_cols
        se_rank = se_row * self.p_cols + se_col

        # Préparation des données pour les coins
        # Je vais envoyer mes 4 coins réels à mes 4 voisins diagonaux,
        # et recevoir leurs coins correspondants.
        # Les données sont des scalaires, donc pas besoin de copy, mais on utilise un tableau 1D
        send_nw = np.array([self.cells[1, 1]], dtype=np.uint8)          # coin Nord-Ouest réel
        send_ne = np.array([self.cells[1, self.nx_loc]], dtype=np.uint8) # coin Nord-Est réel
        send_sw = np.array([self.cells[self.ny_loc, 1]], dtype=np.uint8) # coin Sud-Ouest réel
        send_se = np.array([self.cells[self.ny_loc, self.nx_loc]], dtype=np.uint8) # coin Sud-Est réel

        # Buffers de réception
        recv_nw = np.empty(1, dtype=np.uint8)
        recv_ne = np.empty(1, dtype=np.uint8)
        recv_sw = np.empty(1, dtype=np.uint8)
        recv_se = np.empty(1, dtype=np.uint8)

        # Liste des requêtes pour les diagonales
        diag_reqs = []

        # Échange avec Nord-Ouest (je reçois depuis NW, j'envoie à SE)
        diag_reqs.append(self.worker_comm.Irecv(recv_nw, source=nw_rank, tag=4))
        diag_reqs.append(self.worker_comm.Isend(send_se, dest=se_rank, tag=4))
        # Échange avec Nord-Est (je reçois depuis NE, j'envoie à SW)
        diag_reqs.append(self.worker_comm.Irecv(recv_ne, source=ne_rank, tag=5))
        diag_reqs.append(self.worker_comm.Isend(send_sw, dest=sw_rank, tag=5))
        # Échange avec Sud-Ouest (je reçois depuis SW, j'envoie à NE)
        diag_reqs.append(self.worker_comm.Irecv(recv_sw, source=sw_rank, tag=6))
        diag_reqs.append(self.worker_comm.Isend(send_ne, dest=ne_rank, tag=6))
        # Échange avec Sud-Est (je reçois depuis SE, j'envoie à NW)
        diag_reqs.append(self.worker_comm.Irecv(recv_se, source=se_rank, tag=7))
        diag_reqs.append(self.worker_comm.Isend(send_nw, dest=nw_rank, tag=7))

        # Attendre la fin des communications diagonales
        MPI.Request.Waitall(diag_reqs)

        # Mise à jour des fantômes diagonaux
        self.cells[0, 0] = recv_nw[0]
        self.cells[0, self.nx_loc+1] = recv_ne[0]
        self.cells[self.ny_loc+1, 0] = recv_sw[0]
        self.cells[self.ny_loc+1, self.nx_loc+1] = recv_se[0]

    def compute_next_iteration(self):
        ny = self.ny_loc
        nx = self.nx_loc
        N = ny + 2
        M = nx + 2
        next_cells = np.zeros_like(self.cells)
        diff = []

        for i in range(1, ny+1):
            for j in range(1, nx+1):
                voisines = [
                    self.cells[(i-1)%N, (j-1)%M], self.cells[(i-1)%N, j%M], self.cells[(i-1)%N, (j+1)%M],
                    self.cells[i%N,     (j-1)%M],                            self.cells[i%N,     (j+1)%M],
                    self.cells[(i+1)%N, (j-1)%M], self.cells[(i+1)%N, j%M], self.cells[(i+1)%N, (j+1)%M]
                ]
                nb_voisines = sum(voisines)

                if self.cells[i, j] == 1:
                    if nb_voisines in (2, 3):
                        next_cells[i, j] = 1
                else:
                    if nb_voisines == 3:
                        next_cells[i, j] = 1

        self.cells = next_cells
        return diff

class App:
    """Gestion de l'affichage Pygame (inchangée)."""
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
    worker_comm = new_comm      # communicateur pour les workers (rangs 0..nbp-2 localement)
else:
    worker_comm = None

# -------------------------------------------------------------------
if __name__ == '__main__':
    import time
    import sys

    pg.init()
    dico_patterns = {   # (identique à l'original)
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
    global_dim, pattern = init_pattern   # global_dim = (ny_glob, nx_glob)

    # -----------------------------------------------------------------
    # Détermination de la grille de processus 2D pour les workers
    worker_nbp = nbp - 1
    if worker_nbp < 1:
        print("Erreur : il faut au moins 2 processus (1 pour l'affichage + 1 worker)")
        exit(1)

    p_rows, p_cols = get_2d_grid_dims(worker_nbp)
    if rank == 0:
        print(f"Grille de processus : {p_rows} lignes × {p_cols} colonnes = {worker_nbp} workers")

    # Création de la grille locale pour chaque worker
    if rank != 0:
        local_rank = rank - 1            # rang dans worker_comm (0 .. worker_nbp-1)
        row = local_rank // p_cols
        col = local_rank % p_cols
        grid = Grille2D(worker_comm, row, col, p_rows, p_cols,
                        global_dim, pattern,
                        pg.Color("black"), pg.Color("white"))
    else:
        grid = None

    # Création de l'application d'affichage (seulement sur le root)
    if rank == 0:
        appli = App((resx, resy), global_dim, pg.Color("black"), pg.Color("white"))
    else:
        appli = None

    # -----------------------------------------------------------------
    mustContinue = True
    if rank == 0:
        filename = f"temps2D2_{nbp}.txt"   # nb de workers
        f = open(filename, "w")
        t0 = time.time()   # début global
    # Premier échange pour initialiser les fantômes
    if rank != 0:
        grid.exchange_ghost()

    while mustContinue:
        t_total_start = time.time()  # début du temps total de l’itération

        # -------------------- Calcul (workers) --------------------
        if rank != 0:
            t_calc_start = time.time()
            grid.exchange_ghost()                # met à jour les fantômes
            grid.compute_next_iteration()
            t_calc_end = time.time()
            t_calc = t_calc_end - t_calc_start

            # Préparation des données locales (sans fantômes) pour l'envoi
            local_data = grid.cells[1:grid.dimensions[0]-1, 1:grid.dimensions[1]-1].flatten()
            info = np.array([grid.ny_loc, grid.nx_loc, grid.y_start, grid.x_start], dtype=int)
        else:
            t_calc = 0
            local_data = np.array([], dtype=np.uint8)
            info = np.array([0, 0, 0, 0], dtype=int)
        # Chaque worker a t_calc, rank 0 veut la somme
        t_calc_scalar = np.array(t_calc, dtype=np.float64)
        t_calc_total = np.array(0.0, dtype=np.float64)

        # Reduce avec SUM vers rank 0, en excluant rank 0 si t_calc=0
        globCom.Reduce(t_calc_scalar, t_calc_total, op=MPI.MAX, root=0)

        # -------------------- Rassemblement --------------------
        all_info = None
        if rank == 0:
            all_info = np.zeros((nbp,4), dtype=int)
        globCom.Gather(info, all_info, root=0)

        # Gatherv des données
        if rank == 0:
            recvcounts = np.zeros(nbp, dtype=int)
            for i in range(1, nbp):
                ny = all_info[i,0]
                nx = all_info[i,1]
                recvcounts[i] = ny * nx
            displs = np.zeros(nbp, dtype=int)
            for i in range(1, nbp):
                displs[i] = displs[i-1] + recvcounts[i-1]
            total = displs[-1] + recvcounts[-1]
            global_data = np.zeros(total, dtype=np.uint8)
        else:
            recvcounts = None
            displs = None
            global_data = None

        globCom.Gatherv(local_data, [global_data, recvcounts, displs, MPI.UINT8_T], root=0)

        # -------------------- Reconstruction & Affichage --------------------
        if rank == 0:
            # reconstruction de la grille globale
            global_cells = np.zeros(global_dim, dtype=np.uint8)
            for i in range(1, nbp):
                ny = all_info[i,0]
                nx = all_info[i,1]
                y_start = all_info[i,2]
                x_start = all_info[i,3]
                bloc = global_data[displs[i]:displs[i]+recvcounts[i]].reshape((ny,nx))
                global_cells[y_start:y_start+ny, x_start:x_start+nx] = bloc

            # affichage
            t_display_start = time.time()
            appli.draw_global(global_cells)
            t_display_end = time.time()
            t_affichage = t_display_end - t_display_start

            # gestion événements
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False

            # affichage des temps
            print(f"[Itération] Temps calcul (workers) : {t_calc_total:.6e}s | Temps affichage : {t_affichage:.6e}s")

        # -------------------- Temps total --------------------
        t_total_end = time.time()
        t_total = t_total_end - t_total_start
        if rank == 0:
                    print(f"[Itération] Temps total : {t_total:.6e}s")

                    # écrire seulement pendant les 10 premières secondes
                    if time.time() - t0 <= 5:
                        f.write(f"{t_calc_total}\n")
                        f.flush()

        # diffusion de mustContinue
        mustContinue = globCom.bcast(mustContinue, root=0)
    if rank == 0:
        f.close()
