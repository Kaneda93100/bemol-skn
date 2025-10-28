import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from bemol import data_processing
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter

PI = float(np.pi)
tan = np.tan
sin = np.sin 
zeros = np.zeros

import os

import pandas as pd

## uncoment following lines if ModuleNotFounError
## add repo folder to PYTHONPATH, if running from the samples folder
import sys
sys.path.append(os.path.abspath(f'{__file__}/../..'))

import bemol
from bemol import data_processing

wind = data_processing.wind
omega = data_processing.omega
pitch = data_processing.pitch
rho = 1.191
N = 1.0 #number of revolutions
tStep = 0.1

mexico = bemol.rotor.mexico
HR = mexico.hubRadius
TR = mexico.tipRadius

preconeAngle = 0.0
tiltAngle = 0.0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)
device = 'cpu'
#######################################################################
########################### génération  ###############################
########################### des données ###############################
#######################################################################

"""
    On génère des paramètres d'entrée, qui seront considérés comme fixe. On fera d'abord varier 
    A0 comme paramètre, et une fois que ça marche on fera varier (A0, Phi1).
"""

Yaw_corr = data_processing.yaw # Correction de l'AI
Burton = data_processing.Burton # Correction du Skew
AI_vanilla = data_processing.AI_vanilla #Calculateur de l'induction axiale (AI) native
App = data_processing.Apply_corr # Appliquer la correction (à besoins d'être différentiable)

correction = []
solver = bemol.ning.NingUncoupled(mexico, rho, correction)

samp_size = 10**2

# Phi1 & Phi2
phi1 = 2*PI*np.random.rand(samp_size) # phi1 € [0,2pi]
phi2 = 2*PI*np.random.rand(samp_size) # phi2 € [0,2pi]

# Skew, azs, radius
skew = 2*PI*np.random.rand(samp_size) # skew € [0,2pi]
azs = 2*PI*np.random.rand(samp_size) # azs € [0,2pi]
radius = TR*np.random.rand(samp_size) + HR # radius € [HR, TR]

"""
    On calcule les inductions axiales non-corrigées pour des paramètres mu = (phi1, phi2, skew, azs, radius).
    Il y aura sûrement des incohérences dans les combinaisons de mu étant donné que les paramètre 
    sont choisit aléatoirement. 

    On fait plusieurs hypothèses : 
        - preconeAngle = 0.0
        - tiltAngle = 0.0
        - Paramètres physique globalement considéré comme constant (vent, vitesse de rotation, etc)
        - Yaw = Skew
"""

AI_va = np.zeros_like(phi1) # Induction axiale non corrigée

for i in range (AI_va.shape[0]) : 
    AI_va[i], _ = AI_vanilla(solver, azs[i], skew[i], radius[i], tilt = tiltAngle, precone = preconeAngle, pitch = pitch, omega = omega, wind = wind, tStep = tStep)

# Calcul des WSA (Wake Skew Angle)
WSA = np.zeros_like(skew)
for i in range(skew.shape[0]):
    WSA[i] = Burton(AI_va[i], skew[i])

yaw_corrections_A0 = np.zeros_like(phi1)
A0 = 0.35
for i in range(yaw_corrections_A0.shape[0]) :
    yaw_corrections_A0[i] = Yaw_corr(WSA = WSA[i], AA = azs[i],  R = radius[i], Phi1 = phi1[i], Phi2 = phi2[i], A0 = A0, HR=HR, TR=TR)

# correction de l'Induction Axiale avec A0 = 0.35

AI_up_A0 = np.zeros_like(AI_va)
for i in range(AI_up_A0.shape[0]) :
    AI_up_A0[i] = App(AI_va[i], yaw_corrections_A0[i]) 

"""
    On rentre dans le dur du sujet. On va maintenant écrire la procédure 
    d'optimisation permettant de retrouver A0 par descente de gradient.
"""

x0 = 1.0 # Initialisation
step_max = 5000 # Arrêt de la suite
eps = 1e-7 # Seuil de tolérance

input = torch.zeros((samp_size, 6), dtype = torch.float64, device = 'cpu') 
input[:,0] = torch.tensor(phi1, dtype = torch.float64)
input[:,1] = torch.tensor(phi2, dtype = torch.float64)
input[:,2] = torch.tensor(WSA, dtype = torch.float64)
input[:,3] = torch.tensor(azs, dtype = torch.float64)
input[:,4] = torch.tensor(radius, dtype = torch.float64)
input[:,5] = torch.tensor(AI_va, dtype = torch.float64)
input.to(device)

target = torch.tensor(AI_up_A0, device = device, dtype = torch.float64)

# Choix de l'algorithme d'optimisation et de la Loss 
LR = 1e-3
param = torch.tensor(x0, requires_grad=True)
optimiser = optim.Adam([param], lr = LR)
loss_func = nn.MSELoss(reduction = 'mean')

loss_list = []

for step in range(step_max) : 
    optimiser.zero_grad() # Réinitialiser le gradient d'une étape à l'autre
    
    X = Yaw_corr(WSA = input[:,2], AA = input[:,3], R = input[:,4], Phi1 = input[:,0], Phi2 = input[:,1], A0 = param, HR = HR, TR = TR)
    X = App(input[:,5], X)

    loss = loss_func(input = X, target = target)
    loss.backward()

    optimiser.step()
    loss_list.append(loss)

#print(loss_list)
print("Paramètre : ", param.item())
print("Dernière erreur enregistré : ", loss_list[-1])


