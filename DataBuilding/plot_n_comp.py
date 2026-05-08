import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pathlib as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def preplot_attr(path_dset1:p.Path, path_dset2:p.Path, abs:str, arg:str):

    if path_dset1.suffix == '.xlsx' :    
        dset1 = pd.read_excel(path_dset1); 
    elif path_dset1.suffix == '.csv':
        dset1 = pd.read_csv(path_dset1, comment = '#', sep = ',')
    else : 
        raise ImportError(f"Chemin {path_dset1} incorrect. Il faut qu'il pointe directement sur le fichier.")
    
    if path_dset2.suffix == '.xlsx' :    
        dset2 = pd.read_excel(path_dset2); 
    elif path_dset2.suffix == '.csv':
        dset2 = pd.read_csv(path_dset2, comment = '#', sep = ',')
    else : 
        raise ImportError(f"Chemin {path_dset2} incorrect. Il faut qu'il pointe directement sur le fichier.")    
        
    d1 = dset1[arg].to_numpy()
    d2 = dset2[arg].to_numpy()

    abs_np = dset1[abs].unique()

    return abs_np,d1,d2
