# Version parallélisé suivant les lignes
"""
    Membre du groupes:
    ------------------
        -Dohemeto Bonaventure 
        -BURNS Thomas

Le jeu de la vie
################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonales.) L'interaction se fait selon les
règles suivantes pour calculer l'irération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

On itère ensuite pour étudier la façon dont évolue la population des cellules sur la grille.
"""
import pygame as pg

import numpy as np

import random

import time

import sys


class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    En entrée lors de la création de la grille :
        - dimensions est un tuple contenant le nombre de cellules dans les deux directions (nombre lignes, nombre colonnes)
        - init_pattern est une liste de cellules initialement vivantes sur cette grille (les autres sont considérées comme mortes)
        - color_life est la couleur dans laquelle on affiche une cellule vivante
        - color_dead est la couleur dans laquelle on affiche une cellule morte
    Si aucun pattern n'est donné, on tire au hasard quels sont les cellules vivantes et les cellules mortes
    Exemple :
       grid = Grille( (10,10), init_pattern=[(2,2),(0,2),(4,2),(2,0),(2,4)], color_life=pg.Color("red"), color_dead=pg.Color("black"))
    """
    def __init__(self, rank, worker_nbp, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.rank = rank
        self.worker_nbp = worker_nbp   
        # Distribution équitable des lignes
        ny_global, nx = dim
        
        base = ny_global // worker_nbp
        rest = ny_global % worker_nbp
        self.ny_loc = base + (1 if rank < rest else 0)
        self.y_start = rank * base + min(rank, rest)
        self.dimensions = (self.ny_loc + 2, nx)

        # Initialisation des cellules
        self.cells = np.zeros(self.dimensions, dtype=np.uint8)

        if init_pattern is not None:
            for (i_glob, j_glob) in init_pattern:
                if self.y_start <= i_glob < self.y_start + self.ny_loc:
                    i_loc = i_glob - self.y_start + 1  # +1 à cause du fantôme du haut
                    self.cells[i_loc, j_glob] = 1
        else:
            # Initialisation aléatoire uniquement pour les lignes locales
            self.cells[1:self.ny_loc+1, :] = np.random.randint(2, size=(self.ny_loc, nx), dtype=np.uint8)

        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        ny_reel = self.dimensions[0] - 2   # nombre de lignes réelles
        nx = self.dimensions[1]
        next_cells = np.zeros_like(self.cells)   # on part sur du 0 partout
        diff = []   # liste des indices des cellules modifiées (facultatif)

        for i in range(1, ny_reel+1):
            for j in range(nx):
                # Extraction des 8 voisins (les fantômes sont déjà à jour)
                voisines = [
                    self.cells[i-1, (j-1)%nx], self.cells[i-1, j], self.cells[i-1, (j+1)%nx],
                    self.cells[i,   (j-1)%nx],                     self.cells[i,   (j+1)%nx],
                    self.cells[i+1, (j-1)%nx], self.cells[i+1, j], self.cells[i+1, (j+1)%nx]
                ]
                nb_voisines = sum(voisines)

                if self.cells[i, j] == 1:
                    if nb_voisines in (2, 3):
                        next_cells[i, j] = 1
                    else:
                        next_cells[i, j] = 0
                        diff.append((i-1)*nx + j)   # indice global dans le domaine local (0..ny_reel*nx-1)
                else:
                    if nb_voisines == 3:
                        next_cells[i, j] = 1
                        diff.append((i-1)*nx + j)
                    # sinon reste 0 (déjà initialisé)

        self.cells = next_cells
        return diff

    def modificateur(self, diff):
        nx = self.dimensions[1]
        for c in diff:
            nr = c//nx
            nc = c%nx
            self.cells[nr, nc] = (1 - self.cells[nr, nc])
        return None

    def exchange_ghost_lines(self, comm):
        size = comm.Get_size()
        
        """
            if size == 1:
                # Un seul worker : recopie circulaire interne
                self.cells[0, :] = self.cells[self.dimensions[0]-2, :]
                self.cells[self.dimensions[0]-1, :] = self.cells[1, :]
                return  
        """
        rank = comm.Get_rank()
        top = (rank - 1) % size
        bottom = (rank + 1) % size

        # Envoi de la dernière ligne réelle au voisin du haut (pour son fantôme du haut)
        send_top = self.cells[self.dimensions[0]-2, :]   # dernière ligne réelle
        recv_top = np.empty(self.dimensions[1], dtype=np.uint8)
        comm.Sendrecv(sendbuf=send_top, dest=top, recvbuf=recv_top, source=top)
        self.cells[0, :] = recv_top

        # Envoi de la première ligne réelle au voisin du bas (pour son fantôme du bas)
        send_bottom = self.cells[1, :]                    # première ligne réelle
        recv_bottom = np.empty(self.dimensions[1], dtype=np.uint8)
        comm.Sendrecv(sendbuf=send_bottom, dest=bottom, recvbuf=recv_bottom, source=bottom)
        self.cells[self.dimensions[0]-1, :] = recv_bottom

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
        """global_cells : tableau 2D numpy de dimensions globales"""
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


from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup()
rank = globCom.Get_rank()
nbp = globCom.Get_size()

# Division de processus : 0 pour l'affichage, 1..nbp-1 pour les workers
color = 0 if rank == 0 else 1
key = rank
new_comm = globCom.Split(color, key)
if rank != 0:
    worker_comm = new_comm
else:
    worker_comm = None

if __name__ == '__main__':
    

    pg.init()
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

    # Création des grilles pour les workers
    if rank != 0:
        worker_rank = rank - 1
        worker_nbp = nbp - 1
        grid = Grille(worker_rank, worker_nbp, dim, pattern)
    else:
        grid = None

    if rank == 0:
        appli = App((resx, resy), dim, pg.Color("black"), pg.Color("white"))
    else:
        appli = None

    mustContinue = True
    # Premier échange de fantômes pour initialiser les lignes fantômes
    if rank != 0:
        grid.exchange_ghost_lines(worker_comm)

    while mustContinue:
        if rank != 0:
            # Mesure du temps de calcul
            t1 = time.time()
            grid.exchange_ghost_lines(worker_comm)
            grid.compute_next_iteration()
            t2 = time.time()
            print(f"Worker {rank} compute time: {t2-t1:.2e} secondes")

            # Préparation des données locales (sans les fantômes)
            local_data = grid.cells[1:grid.dimensions[0]-1, :].flatten()
            local_nrows = grid.dimensions[0] - 2
        else:
            # Processus 0 : données vides
            local_data = np.array([], dtype=np.uint8)
            local_nrows = 0

        # Rassemblement des tailles locales
        all_nrows = None
        if rank == 0:
            all_nrows = np.zeros(nbp, dtype=int)
        globCom.Gather(np.array([local_nrows], dtype=int), all_nrows, root=0)

        if rank == 0:
            nx = dim[1]
            recvcounts = all_nrows * nx
            displs = np.zeros(nbp, dtype=int)
            for i in range(1, nbp):
                displs[i] = displs[i-1] + recvcounts[i-1]
            total_cells = displs[-1] + recvcounts[-1]
            global_cells_flat = np.zeros(total_cells, dtype=np.uint8)
        else:
            recvcounts = None
            displs = None
            global_cells_flat = None

        # Rassemblement des données
        globCom.Gatherv(local_data, [global_cells_flat, recvcounts, displs, MPI.UINT8_T], root=0)

        if rank == 0:
            # Reconstruction de la grille globale
            global_cells = np.zeros(dim, dtype=np.uint8)
            for i in range(1, nbp):
                start_row = displs[i] // nx
                end_row = start_row + all_nrows[i]
                global_cells[start_row:end_row, :] = global_cells_flat[displs[i]:displs[i]+recvcounts[i]].reshape((all_nrows[i], nx))

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