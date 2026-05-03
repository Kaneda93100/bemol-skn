import pathlib as P
import pandas as pd
import numpy as np
import yaml

def rewrite_geom(path:P.Path, attribut:str, y:np.array, lbd:float = 1.):

    """
    Fonction permettant d'écrire dans les fichier blade.dat. 
    La fonction permet de modifier en additionnant/multipliant les valeurs de chaque attrtibuts.
    Les blade.dat doivent être structurés comme suit : 

    """
    
    data_blade = pd.read_csv(path,  sep = '\s+', names = ['radius', 'twist', 'chord', 'airfoil'], header = 0)
    list_attr = list(data_blade.keys())
   
    if y.shape != data_blade[attribut].shape :
        raise ValueError("\nLe vecteur de translation n'est pas de la bonne taille.\n")    
    if attribut not in list_attr:
        raise ValueError(f"L'attribut radius ne se trouve pas dans le fichier {path}, vérifiez la structure de donné du fichier.\n")
    if attribut == 'airfoil' :
        raise ValueError("\nOn ne peut pas écrire sur la liste des profil avec cette fonction.\n")
    
    ## Modification de l'attribut
    data_blade[attribut] *= lbd
    data_blade[attribut] += y
    data_blade.to_csv(path, sep = '\t', header=True)

    print(f"\nLe fichier {path} à bien été modifié. Tâche terminée !\n")

    return 

def rewrite_pitch(path:P.Path, new_pitch):
    with open(path, 'r') as file :
        data_rot = yaml.safe_load(file)

    data_rot['pitchRated'] *= -1

    with open(path, 'w') as file :
        yaml.dump(data_rot, file)

    print(f"\nLe fichier {path} à bien été modifié (l'attribut pitchRated). Tâche terminée !\n")
    return