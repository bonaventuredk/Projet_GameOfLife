# Veuillez lancer le script de tests nommé TESTs.py dans le dossier Tests
# Plots pour (sans Sendrecv)
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend sans interface graphique
import matplotlib.pyplot as plt
def extraire_type_et_nb(nom_fichier):
    """
    Extrait le type et le nombre de processus à partir du nom.
    Formats supportés:
        tempsline_3.txt      -> ('line', 3)
        tempsline2_4.txt     -> ('line2', 4)
        tempscols_5.txt      -> ('cols', 5)
        tempscols2_6.txt     -> ('cols2', 6)
        temps2D_7.txt        -> ('2D', 7)
        temps2D2_8.txt       -> ('2D2', 8)
    """
    base = os.path.basename(nom_fichier)
    parties = base.split('_')
    if len(parties) < 2:
        raise ValueError(f"Nom de fichier mal formé : {base}")
    prefix = parties[0]   # ex: tempsline, tempsline2, tempscols, etc.
    nb_part = parties[1].split('.')[0]

    # Déterminer le type en fonction du préfixe
    if prefix == 'tempsline2':
        typ = 'line2'
    elif prefix == 'tempscols2':
        typ = 'cols2'
    elif prefix == 'temps2D2':
        typ = '2D2'
    else:
        raise ValueError(f"Préfixe inconnu : {prefix}")

    return typ, int(nb_part)

def calculer_moyenne_fichier(nom_fichier):
    with open(nom_fichier, 'r') as f:
        lignes = f.readlines()
    valeurs = []
    for ligne in lignes:
        ligne = ligne.strip()
        if ligne:
            valeurs.append(float(ligne))
    if not valeurs:
        return None
    return np.mean(valeurs)

def main():
    # Définir les motifs pour chaque type
    motifs = [
        ('line2',  'tempsline2_*.txt'),
        ('cols2',  'tempscols2_*.txt'),
        ('2D2',    'temps2D2_*.txt')
    ]

    # Dictionnaire pour stocker les données : clé = type, valeur = (liste_nb, liste_temps)
    data = {typ: ([], []) for typ, _ in motifs}

    for typ, motif in motifs:
        fichiers = glob.glob(motif)
        for f in fichiers:
            try:
                t, nb = extraire_type_et_nb(f)
                if t != typ:
                    print(f"Type inattendu pour {f}: extrait {t}, attendu {typ}")
                    continue
                moy = calculer_moyenne_fichier(f)
                if moy is not None:
                    data[typ][0].append(nb)
                    data[typ][1].append(moy)
                    print(f"{f} : {typ} {nb} processus, temps moyen = {moy:.6f} s")
                else:
                    print(f"Attention : {f} est vide ou illisible.")
            except Exception as e:
                print(f"Erreur lors du traitement de {f} : {e}")

    # Calculer le temps séquentiel
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    seq_path = root_dir / 'tempsseq.txt'
    if not seq_path.exists():
        seq_path = script_dir / 'tempsseq.txt'
    if not seq_path.exists():
        raise FileNotFoundError(f"tempsseq.txt introuvable : {root_dir}/tempsseq.txt et {script_dir}/tempsseq.txt")

    seq_time = calculer_moyenne_fichier(seq_path)
    if seq_time is not None:
        print(f"Sequential : temps moyen = {seq_time:.6f} s")
    else:
        print(f"Attention : {seq_path} est vide ou illisible.")
        seq_time = None

    # Tracer les courbes
    plt.figure(figsize=(12, 8))

    styles = {
        'line':  {'color': 'blue',   'marker': 'o', 'label': 'Line 1'},
        'line2': {'color': 'cyan',   'marker': 'v', 'label': 'Line 2'},
        'cols':  {'color': 'green',  'marker': 's', 'label': 'Cols 1'},
        'cols2': {'color': 'lime',   'marker': 'D', 'label': 'Cols 2'},
        '2D':    {'color': 'red',    'marker': '^', 'label': '2D 1'},
        '2D2':   {'color': 'orange', 'marker': '*', 'label': '2D 2'}
    }

    for typ, (nb_list, temps_list) in data.items():
        if nb_list:
            # Trier par nombre de processus
            tri = sorted(zip(nb_list, temps_list))
            nb_trie, temps_trie = zip(*tri)
            style = styles.get(typ, {'color': 'black', 'marker': 'x', 'label': typ})
            plt.plot(nb_trie, temps_trie,
                     marker=style['marker'],
                     linestyle='-',
                     linewidth=2,
                     markersize=8,
                     label=style['label'],
                     color=style['color'])

    plt.xlabel('Nombre de processus')
    plt.ylabel('Temps moyen (secondes)')
    plt.title('Comparaison des temps moyens selon la méthode de découpage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('2_comparaison_temps_moyens.png', dpi=150)
    print("\nGraphique sans séquentiel sauvegardé sous '2_comparaison_temps_moyens.png'.")

    # Ajouter la ligne séquentielle
    if seq_time is not None:
        plt.axhline(y=seq_time, color='black', linestyle='--', linewidth=2, label='Sequential')
        plt.legend()
        plt.savefig('2_comparaison_temps_moyens_seq.png', dpi=150)
        print("Graphique avec séquentiel sauvegardé sous '2_comparaison_temps_moyens_seq.png'.")

if __name__ == "__main__":
    main() 
