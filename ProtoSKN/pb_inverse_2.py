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
n = 3
samp_size = 10**n

print("Calcul des données de références.\n")
# Phi1
phi2 = 2*PI*torch.rand(samp_size, dtype = torch.float64) # phi2 € [0,2pi]

# Skew, azs, radius
skew = 2*PI*torch.rand(samp_size, dtype = torch.float64) # skew € [0,2pi]
azs = 2*PI*torch.rand(samp_size, dtype = torch.float64) # azs € [0,2pi]
radius = TR*torch.rand(samp_size, dtype = torch.float64) + HR # radius € [HR, TR]

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

AI_va = torch.zeros_like(phi2) # Induction axiale non corrigée

for i in range (AI_va.shape[0]) : 
    AI_va[i], _ = AI_vanilla(solver, azs[i], skew[i], radius[i], tilt = tiltAngle, precone = preconeAngle, pitch = pitch, omega = omega, wind = wind, tStep = tStep)

# Calcul des WSA (Wake Skew Angle)
WSA = torch.zeros_like(skew)
for i in range(skew.shape[0]):
    WSA[i] = Burton(AI_va[i], skew[i])

yaw_corrections_A0 = np.zeros_like(phi2)
A0 = 0.35
Phi1 = -PI/9
for i in range(yaw_corrections_A0.shape[0]) :
    yaw_corrections_A0[i] = Yaw_corr(WSA = WSA[i], AA = azs[i],  R = radius[i], Phi1 = Phi1, Phi2 = phi2[i], A0 = A0, HR=HR, TR=TR)

# correction de l'Induction Axiale avec A0 = 0.35

AI_up_A0 = np.zeros_like(AI_va)
for i in range(AI_up_A0.shape[0]) :
    AI_up_A0[i] = App(AI_va[i], yaw_corrections_A0[i]) 

"""
    On rentre dans le dur du sujet. On va maintenant écrire la procédure 
    d'optimisation permettant de retrouver A0 par descente de gradient.
"""

x0 = 1.0 # Initialisation pour A0
theta0 = 1.0 # Initialisation pour phi1
step_max = 500 # Arrêt de la suite
eps = 1e-7 # Seuil de tolérance

input = torch.zeros((samp_size, 5), dtype = torch.float64, device = device) 
input[:,0] = phi2
input[:,1] = WSA
input[:,2] = azs
input[:,3] = radius 
input[:,4] = AI_va 
input.to(device)


target = torch.tensor(AI_up_A0, device = device, dtype = torch.float64)

batch_size = 10**(n-1)
data = DataLoader(TensorDataset(input, target), batch_size=batch_size, pin_memory=False, shuffle = True)

# Choix de l'algorithme d'optimisation et de la Loss 
LR = 1e-2
param = torch.tensor([x0, theta0], requires_grad=True)
optimiser = optim.Adam([param], lr = LR)
loss_func = nn.MSELoss(reduction = 'mean')

loss_list_mean = []

print("Début de l'optimisation\n")
for step in range(step_max) : 
    loss_list_int = []
    if step % 10 == 0 :
        print("Step ", step, "\n")

    for x, y in data : 
        x = x.to(device)
        y = y.to(device)

        optimiser.zero_grad() # Réinitialiser le gradient d'une étape à l'autre
    
        X = Yaw_corr(WSA = x[:,1], AA = x[:,2], R = x[:,3], Phi1 = param[1], Phi2 = x[:,0], A0 = param[0], HR = HR, TR = TR)
        X = App(x[:,4], X)

        loss = loss_func(input = X, target = y)
        loss.backward()

        optimiser.step()
        loss_list_int.append(loss.to('cpu').detach().numpy())
    mean = np.average(loss_list_int)
    loss_list_mean.append(mean)

#print(loss_list)
print("Paramètres calculés : ", param.detach().numpy())
print("Dernière erreur enregistré : ", loss_list_mean[-1])


