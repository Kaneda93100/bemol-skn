import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import brain
import numpy as np
import pandas as pd
import bemol
from bemol import tools
from bemol.data_processing import compute_forces, compute_velocity
import matplotlib.pyplot as plt
from time import perf_counter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)


## Paramètres de l'éolienne
mexico_vortex = bemol.rotor.mexico_vortex
HR = mexico_vortex.hubRadius
TR = mexico_vortex.tipRadius

## Paramètres du modèle
preconeAngle = torch.tensor(0.0)
tiltAngle = torch.tensor(0.0)
pitch = 2.3000244769936637 # Blade pitch
omega = 44.5163679 # Rotation speed
yaw = np.radians(5) # Yaw skew angle
U = 25.04045694375 # Incoming stream's velocity
rho = 1.191 

corr_noyaw = [
    bemol.secondary.HubTipLoss.Dummy,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.Dummy
]
corr_yaw = [
    bemol.secondary.HubTipLoss.Dummy,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]

solver_noyaw = bemol.ning.NingUncoupled(mexico_vortex, rho, corr_noyaw)
solver_yaw = bemol.ning.NingUncoupled(mexico_vortex, rho, corr_yaw)
rotor = solver_yaw.rotor

## Format dans lequel les données sont manipulées
type = torch.float32

print("Début de l'extraction des données...\n")
print("----------------------------------------")
Pre_X, Y = brain.extract('data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv', 
                         type = type)

X = torch.zeros((2592,1,3), dtype=type)
X[:,0,0] = Pre_X[:,0,0] ## Azimuths
X[:,0,1] = Pre_X[:,0,1] ## Radius

elements = np.zeros_like(X[:,0,0], dtype=int)
for i in range(36) : ## Position 
    for j in range(72) :
        elements[72*i + j] = i

for i, position in enumerate(X.shape[0]) : 
    velocities = tools.calculateVelocity(wind = U, omega = omega, rad = rotor.sections[position].radius, azi = X[i,0,0],
                                          yaw = yaw, tilt = tiltAngle, 
                                          precone = preconeAngle)

    X[i,0,2],_,_,_ = solver_yaw.solve(rotor.sections[position], X[i,0,0], pitch = pitch, 
                                    velocity=velocities, angle = [yaw, tiltAngle])
     