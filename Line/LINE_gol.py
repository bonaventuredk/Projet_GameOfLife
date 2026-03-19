# Version parallélisée suivant les lignes avec mesure des temps
"""
    Membres du groupe:
    ------------------
        - Dohemeto Bonaventure 
        - BURNS Thomas

Le jeu de la vie
################
Automate cellulaire de Conway sur grille torique.
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
        self.rank = rank
        self.worker_nbp = worker_nbp
        ny_global, nx = dim
        
        base = ny_global // worker_nbp
        rest = ny_global % worker_nbp
        self.ny_loc = base + (1 if rank < rest else 0)
        self.y_start = rank * base + min(rank, rest)
        self.dimensions = (self.ny_loc + 2, nx)

        self.cells = np.zeros(self.dimensions, dtype=np.uint8)

        if init_pattern is not None:
            for (i_glob, j_glob) in init_pattern:
                if self.y_start <= i_glob < self.y_start + self.ny_loc:
                    i_loc = i_glob - self.y_start + 1
                    self.cells[i_loc, j_glob] = 1
        else:
            self.cells[1:self.ny_loc+1, :] = np.random.randint(2, size=(self.ny_loc, nx), dtype=np.uint8)

        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        ny_reel = self.dimensions[0] - 2
        nx = self.dimensions[1]
        next_cells = np.zeros_like(self.cells)
        diff = []

        for i in range(1, ny_reel+1):
            for j in range(nx):
                voisines = [
                    self.cells[i-1, (j-1)%nx], self.cells[i-1, j], self.cells[i-1, (j+1)%nx],
                    self.cells[i,   (j-1)%nx],                     self.cells[i,   (j+1)%nx],
                    self.cells[i+1, (j-1)%nx], self.cells[i+1, j], self.cells[i+1, (j+1)%nx]
                ]
                nb_voisines = sum(voisines)

                if self.cells[i, j] == 1:
                    next_cells[i, j] = 1 if nb_voisines in (2,3) else 0
                    if nb_voisines not in (2,3):
                        diff.append((i-1)*nx + j)
                else:
                    if nb_voisines == 3:
                        next_cells[i, j] = 1
                        diff.append((i-1)*nx + j)

        self.cells = next_cells
        return diff

    def modificateur(self, diff):
        nx = self.dimensions[1]
        for c in diff:
            nr = c//nx
            nc = c%nx
            self.cells[nr, nc] = 1 - self.cells[nr, nc]

    def exchange_ghost_lines(self, comm):
        size = comm.Get_size()
        rank = comm.Get_rank()
        top = (rank - 1) % size
        bottom = (rank + 1) % size

        send_top = self.cells[self.dimensions[0]-2, :]
        recv_top = np.empty(self.dimensions[1], dtype=np.uint8)
        comm.Sendrecv(sendbuf=send_top, dest=top, recvbuf=recv_top, source=top)
        self.cells[0, :] = recv_top

        send_bottom = self.cells[1, :]
        recv_bottom = np.empty(self.dimensions[1], dtype=np.uint8)
        comm.Sendrecv(sendbuf=send_bottom, dest=bottom, recvbuf=recv_bottom, source=bottom)
        self.cells[self.dimensions[0]-1, :] = recv_bottom

# ========================== Application graphique ==========================

class App:
    def __init__(self, geometry, global_dim, color_life, color_dead):
        self.global_dim = global_dim
        self.color_life = color_life
        self.color_dead = color_dead
        self.size_x = geometry[1] // global_dim[1]
        self.size_y = geometry[0] // global_dim[0]
        self.draw_color = pg.Color('lightgrey') if self.size_x>4 and self.size_y>4 else None
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
                self.screen.fill(self.compute_color(global_cells[i,j]), self.compute_rectangle(i,j))
        if self.draw_color is not None:
            for i in range(self.global_dim[0]):
                pg.draw.line(self.screen, self.draw_color, (0,i*self.size_y), (self.width,i*self.size_y))
            for j in range(self.global_dim[1]):
                pg.draw.line(self.screen, self.draw_color, (j*self.size_x,0), (j*self.size_x,self.height))
        pg.display.update()

# ========================== MPI ==========================

globCom = MPI.COMM_WORLD.Dup()
rank = globCom.Get_rank()
nbp = globCom.Get_size()

color = 0 if rank==0 else 1
key = rank
new_comm = globCom.Split(color, key)
worker_comm = new_comm if rank!=0 else None

# ========================== Main ==========================

if __name__ == '__main__':
    pg.init()

    import time
    import sys

    pg.init()
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

    grid = None
    if rank != 0:
        worker_rank = rank - 1
        worker_nbp = nbp - 1
        grid = Grille(worker_rank, worker_nbp, dim, pattern)

    appli = App((resx,resy), dim, pg.Color("black"), pg.Color("white")) if rank==0 else None

    mustContinue = True
    if rank != 0:
        grid.exchange_ghost_lines(worker_comm)

    # ========================== Boucle principale ==========================
    while mustContinue:
        t_total_start = time.time()  # temps total de l'itération

        # -------------------- Calcul --------------------
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
        
        # Chaque worker a t_calc, rank 0 veut la somme
        t_calc_scalar = np.array(t_calc, dtype=np.float64)
        t_calc_total = np.array(0.0, dtype=np.float64)

        # Reduce avec SUM vers rank 0, en excluant rank 0 si t_calc=0
        globCom.Reduce(t_calc_scalar, t_calc_total, op=MPI.SUM, root=0)

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
        if rank==0:
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
        if rank==0:
            print(f"[Itération] Temps total : {t_total:.6e}s")

        # Diffusion de mustContinue à tous
        mustContinue = globCom.bcast(mustContinue, root=0)

    pg.quit()