import pygame as pg
import numpy as np
from mpi4py import MPI
import sys
import time

# ===============================
# MPI INIT
# ===============================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp  = comm.Get_size()

class Grille:
    def __init__(self, rank, nbp, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.rank = rank
        self.nbp = nbp
        self.global_dim = dim # (rows, cols)
        
        # Distribution des lignes
        self.ny_loc = dim[0] // nbp
        self.y_start = rank * self.ny_loc
        self.y_end = self.y_start + self.ny_loc
        
        # Grille locale + 2 lignes fantômes (haut et bas)
        self.dimensions = (self.ny_loc + 2, dim[1])
        self.cells = np.zeros(self.dimensions, dtype=np.uint8)
        
        if init_pattern is not None:
            for (y, x) in init_pattern:
                if self.y_start <= y < self.y_end:
                    # On place dans la grille locale (y+1 pour sauter le fantôme du haut)
                    self.cells[y - self.y_start + 1, x] = 1
        else:
            # Si pas de pattern, remplissage aléatoire local
            # On ne remplit que les lignes réelles [1:-1]
            self.cells[1:-1, :] = np.random.randint(2, size=(self.ny_loc, dim[1]), dtype=np.uint8)

        self.col_life = color_life
        self.col_dead = color_dead

    def transfert_bords(self):
        """ Échange des lignes fantômes avec les voisins du haut et du bas """
        top_neighbor = (self.rank - 1) % self.nbp
        bot_neighbor = (self.rank + 1) % self.nbp

        # Envoyer ligne 1 (haut réel) au voisin du haut, recevoir dans ligne -1 (fantôme bas)
        comm.Sendrecv(sendbuf=self.cells[1, :], dest=top_neighbor, 
                      recvbuf=self.cells[-1, :], source=bot_neighbor)

        # Envoyer ligne -2 (bas réel) au voisin du bas, recevoir dans ligne 0 (fantôme haut)
        comm.Sendrecv(sendbuf=self.cells[-2, :], dest=bot_neighbor, 
                      recvbuf=self.cells[0, :], source=top_neighbor)

    def compute_next_iteration(self):
        ny = self.ny_loc
        nx = self.global_dim[1]
        next_cells = np.zeros_like(self.cells)

        for i in range(1, ny + 1):
            for j in range(nx):
                # Voisins horizontaux avec tore (modulo)
                jm1 = (j - 1) % nx
                jp1 = (j + 1) % nx
                
                # Somme des 8 voisins (les lignes i-1 et i+1 utilisent les fantômes si besoin)
                nb_voisines = (
                    self.cells[i-1, jm1] + self.cells[i-1, j] + self.cells[i-1, jp1] +
                    self.cells[i,   jm1]                       + self.cells[i,   jp1] +
                    self.cells[i+1, jm1] + self.cells[i+1, j] + self.cells[i+1, jp1]
                )

                if self.cells[i, j] == 1: # Cellule vivante
                    if 2 <= nb_voisines <= 3:
                        next_cells[i, j] = 1
                else: # Cellule morte
                    if nb_voisines == 3:
                        next_cells[i, j] = 1

        self.cells = next_cells

class App:
    def __init__(self, geometry, global_dim, col_life, col_dead):
        self.grid_dim = global_dim
        self.screen = pg.display.set_mode(geometry)
        self.cell_w = geometry[0] // global_dim[1]
        self.cell_h = geometry[1] // global_dim[0]
        self.col_life = col_life
        self.col_dead = col_dead

    def draw(self, full_grid):
        self.screen.fill(self.col_dead)
        for i in range(self.grid_dim[0]):
            for j in range(self.grid_dim[1]):
                if full_grid[i, j] == 1:
                    rect = (j * self.cell_w, i * self.cell_h, self.cell_w, self.cell_h)
                    pg.draw.rect(self.screen, self.col_life, rect)
        pg.display.update()

if __name__ == '__main__':
    # Configuration initiale

    # Génération d'une forêt de 100 canons (10x10)
    forest_pattern = []
    # Le motif de base d'un canon de Gosper
    base_gun = [(5,1), (5,2), (6,1), (6,2), (5,11), (6,11), (7,11), (4,12), (8,12), (3,13), (9,13), (3,14), (9,14), (6,15), (4,16), (8,16), (5,17), (6,17), (7,17), (6,18), (3,21), (4,21), (5,21), (3,22), (4,22), (5,22), (2,23), (6,23), (1,25), (2,25), (6,25), (7,25), (3,35), (4,35), (3,36), (4,36)]
    
    for row in range(10):
        for col in range(10):
            offset_y = row * 100
            offset_x = col * 100
            for y, x in base_gun:
                forest_pattern.append((y + offset_y, x + offset_x))

    dico_patterns = { # Dimension et pattern dans un tuple
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
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)]),
	"r_pentomino": ((100, 100), [(50,51), (51,50), (51,51), (52,51), (50,52)]),
	"galaxy": ((100, 100), [(47,47), (47,48), (47,49), (47,50), (47,51), (47,52), (48,47), (48,48), (48,49), (48,50), (48,51), (48,52), (50,45), (51,45), (52,45), (53,45), (54,45), (55,45), (50,46), (51,46), (52,46), (53,46), (54,46), (55,46), (53,48), (53,49), (53,50), (53,51), (53,52), (53,53), (54,48), (54,49), (54,50), (54,51), (54,52), (54,53), (45,52), (46,52), (47,52), (48,52), (49,52), (50,52), (45,53), (46,53), (47,53), (48,53), (49,53), (50,53)])
    }
    
    dico_patterns['forest'] = ((1000,1000), forest_pattern)

    choice = sys.argv[1] if len(sys.argv) > 1 else 'glider'
    dim, pattern = dico_patterns.get(choice, dico_patterns['glider'])
    
    # On s'assure que le nombre de lignes est divisible par le nombre de process
    if dim[0] % nbp != 0:
        if rank == 0: print(f"Erreur: {dim[0]} lignes non divisibles par {nbp} processus.")
        sys.exit()

    grid = Grille(rank, nbp, dim, pattern)
    
    appli = None
    if rank == 0:
        pg.init()
        appli = App((800, 800), dim, pg.Color("black"), pg.Color("white"))

    mustContinue = True
    while mustContinue:
        # 1. Échange des lignes de bord (Halo)
        grid.transfert_bords()

        # 2. Calcul de la génération suivante
        grid.compute_next_iteration()

        # 3. Rassemblement des données sur le Rank 0
        # On extrait les lignes réelles (sans fantômes)
        local_data = grid.cells[1:-1, :].copy()
        all_data = comm.gather(local_data, root=0)

        # 4. Affichage et gestion des événements sur le Rank 0
        if rank == 0:
            full_grid = np.vstack(all_data)
            appli.draw(full_grid)
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False
            
            # Limiter la vitesse pour que ce soit visible
            # time.sleep(0.02)

        # 5. Synchronisation du signal de sortie
        mustContinue = comm.bcast(mustContinue, root=0)

    if rank == 0:
        pg.quit()
