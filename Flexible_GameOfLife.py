import pygame as pg
import numpy as np
from mpi4py import MPI
import sys

# ===============================
# MPI INIT
# ===============================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp  = comm.Get_size()

class Grille:
    def __init__(self, rank, nbp, dim, init_pattern=None):
        self.rank = rank
        self.nbp = nbp
        self.global_dim = dim
        
        # --- LOGIQUE DE RÉPARTITION FLEXIBLE ---
        base_ny = dim[0] // nbp
        reste = dim[0] % nbp
        
        # Combien de lignes pour CE processeur
        self.ny_loc = base_ny + (1 if rank < reste else 0)
        
        # Calcul du point de départ global (y_start)
        # Chaque rang avant moi qui était dans le 'reste' a pris base_ny + 1
        self.y_start = rank * base_ny + min(rank, reste)
        self.y_end = self.y_start + self.ny_loc
        
        # Debug pour vérifier la répartition
        # print(f"Rank {rank}: lignes {self.y_start} à {self.y_end-1} (Total: {self.ny_loc})")

        # Grille locale + 2 lignes fantômes
        self.cells = np.zeros((self.ny_loc + 2, dim[1]), dtype=np.uint8)
        
        if init_pattern is not None:
            for (y, x) in init_pattern:
                if self.y_start <= y < self.y_end:
                    self.cells[y - self.y_start + 1, x] = 1

    def transfert_bords(self):
        top_neighbor = (self.rank - 1) % self.nbp
        bot_neighbor = (self.rank + 1) % self.nbp
        # On échange toujours une seule ligne, peu importe la taille de ny_loc
        comm.Sendrecv(sendbuf=self.cells[1, :], dest=top_neighbor, 
                      recvbuf=self.cells[-1, :], source=bot_neighbor)
        comm.Sendrecv(sendbuf=self.cells[-2, :], dest=bot_neighbor, 
                      recvbuf=self.cells[0, :], source=top_neighbor)

    def compute_next_iteration(self):
        ny = self.ny_loc
        nx = self.global_dim[1]
        next_cells = np.zeros_like(self.cells)
        for i in range(1, ny + 1):
            for j in range(nx):
                jm1, jp1 = (j - 1) % nx, (j + 1) % nx
                nb_voisines = (
                    self.cells[i-1, jm1] + self.cells[i-1, j] + self.cells[i-1, jp1] +
                    self.cells[i,   jm1]                       + self.cells[i,   jp1] +
                    self.cells[i+1, jm1] + self.cells[i+1, j] + self.cells[i+1, jp1]
                )
                if self.cells[i, j] == 1:
                    if 2 <= nb_voisines <= 3: next_cells[i, j] = 1
                elif nb_voisines == 3:
                    next_cells[i, j] = 1
        self.cells = next_cells

class App:
    def __init__(self, geometry, global_dim):
        self.grid_dim = global_dim
        self.screen = pg.display.set_mode(geometry)
        self.cell_w = geometry[0] // global_dim[1]
        self.cell_h = geometry[1] // global_dim[0]

    def draw(self, full_grid):
        self.screen.fill((255, 255, 255))
        for i in range(self.grid_dim[0]):
            for j in range(self.grid_dim[1]):
                if full_grid[i, j] == 1:
                    rect = (j * self.cell_w, i * self.cell_h, self.cell_w, self.cell_h)
                    pg.draw.rect(self.screen, (0,0,0), rect)
        pg.display.update()

if __name__ == '__main__':
    dico_patterns = {
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "glider": ((100,100),[(1,1),(2,2),(2,3),(3,1),(3,2)])
    }

    choice = sys.argv[1] if len(sys.argv) > 1 else 'space_ship'
    dim, pattern = dico_patterns.get(choice, dico_patterns['space_ship'])

    grid = Grille(rank, nbp, dim, pattern)
    
    appli = None
    if rank == 0:
        pg.init()
        appli = App((600, 600), dim)

    mustContinue = True
    while mustContinue:
        grid.transfert_bords()
        grid.compute_next_iteration()

        # Gather fonctionne très bien même si les tableaux ont des tailles différentes
        local_data = grid.cells[1:-1, :].copy()
        all_data = comm.gather(local_data, root=0)

        if rank == 0:
            # np.vstack empile les morceaux même s'ils n'ont pas le même nombre de lignes
            full_grid = np.vstack(all_data)
            appli.draw(full_grid)
            
            for event in pg.event.get():
                if event.type == pg.QUIT: mustContinue = False
        
        mustContinue = comm.bcast(mustContinue, root=0)

    if rank == 0: pg.quit()
