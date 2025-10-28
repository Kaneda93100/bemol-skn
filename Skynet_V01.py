import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter

PI = np.pi

import os

import pandas as pd

## uncoment following lines if ModuleNotFounError
## add repo folder to PYTHONPATH, if running from the samples folder
import sys
sys.path.append(os.path.abspath(f'{__file__}/../..'))

import bemol


results_folder = 'results/yaw_models'
os.makedirs(results_folder,exist_ok=True)

mexico = bemol.rotor.mexico

wind = 15.06
omega = 44.5163679
rho = 1.191
number_revolutions = 1.0
tStep = 0.1
elements = [24] # node close to the experimental data (r/R = 0.82)
print(type(elements))


preconeAngle = 0.0
tiltAngle = 0.0


#Sans yaw
corrections_unyawed = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
]

#Avec yaw
corrections_yawed = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN,
]


##Solvers 
solver_unyawed = bemol.ning.NingUncoupled(mexico, rho, corrections_unyawed)

solver_yawed = bemol.ning.NingUncoupled(mexico, rho, corrections_yawed)

#######################################################################
###################### RESEAU DE NEURONES SKYNET ###################### 
#######################################################################

"""
    Pour l'instant, j'écris juste un réseau de neurone permettant d'imiter 
    la correction IFPEN (que l'on trouvera dans bemol/secondary.py) en fixant
    le paramétrage de l'éolienne du projet Mexico
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)

################## DATA ##################

##Samples for every parameters
samp_size = 10**1
Yaw = 2*PI*np.random.rand(samp_size, 1)
Skew = Yaw
Azimuths = 2*PI*np.random.rand(samp_size, 1)
#Radius = mexico.tipRadius*np.random.rand(samp_size, 1) + mexico.hubRadius

train_prop = 80/100 #Proportion de l'échantillon de test sur l'échantillon total
train_size = int(train_prop * samp_size)
val_size = samp_size - train_prop

#TODO : peut-être supprimer ces déclarations de arrays, pas forcément utile 
X = np.zeros(samp_size, 3)
X[1,:] = Skew
X[2,:] = Yaw
X[3,:] = Azimuths
#X[4,:] = Radius


X_train = np.zeros(train_size, 3)
X_val = np.zeros(val_size, 3)

##Remplissage de X_train
for i in range(train_size):
        X_train[i,1] = Yaw[i,1]
        X_train[i,2] = Skew[i,1]
        X_train[i,3] = Azimuths[i,1]
##Remplissage de X_val
for i in range(val_size):
        X_val[i,1] = Yaw[train_size + i,1]
        X_val[i,2] = Skew[train_size + i,1]
        X_val[i,3] = Azimuths[train_size + i,1]


##On récupère l'induction axiale
Y_train = np.zeros(180, 2, train_size)
Y_val = np.zeros(180, 2, val_size)

AI = np.zeros(180,2) #variable auxiliaire

for i in range(train_size) :
   _, AI, _ =  solver.cycle(
        mexico.pitchRated,wind,omega,angles=[yawAngle,tiltAngle],tStep=tStep,
        n_phi=180,N=number_revolutions,elements=elements,
    )
    

test_size = int(8/10 * samp_size)


#x_train, x_val, y_train, y_val = map(T.tensor, train_test_split(x, y, test_size=0.2))

##Forcer le typage des tenseurs en float32 (visiblement en float64 par défaut)
x_train = x_train.type(T.float32)
y_train = y_train.type(T.float32)

x_val = x_val.type(T.float32).to(device)
y_val = y_val.type(T.float32).to(device)

train_dataloader = DataLoader(TensorDataset(x_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE, pin_memory=False, shuffle=True)
val_dataloader = DataLoader(TensorDataset(x_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE, pin_memory=False, shuffle=True)

################## ARCHITECTURE DU RESEAU ##################


##Hyperparamètres 
LR = 1 * 1e-4
MAX_EPOCH = 20
BATCH_SIZE = 2000

#Gestion de l'affichage 
step  = 1 #étape à afficher

#Préréglages du réseau 
bool = True
in_size = 2
out_size = 1
layer_size = 2500

act1 = nn.Tanh #Activation pour la 1ère couche
act2 = nn.ReLU #Activation pour la 2ème couche
act3 = nn.ELU #Activation pour la 3ème couche
act4 = nn.Sigmoid #Activation pour la 4ème couche 

a1 = act1.__name__
a2 = act2.__name__
a3 = act3.__name__
a4 = act4.__name__

##Une entrée, act = sigmoid, 1 couche cachée
class N (nn.Module) :
    def __init__(self):
        super(N, self).__init__()

        self.regressor = nn.Sequential(nn.Linear(in_size, layer_size, bool), act1(), #Première couche
                                       nn.Linear(layer_size, layer_size, bool), act2(), #Deuxième couche
                                       nn.Linear(layer_size, layer_size, bool), act3(inplace=True), #Troisième couche
                                       nn.Linear(layer_size,layer_size, bool), act4(), #4ème couche
                                       nn.Linear(layer_size,out_size,bool)) 

    def forward(self,x) : ##Parcourir le réseau
        return self.regressor(x)
