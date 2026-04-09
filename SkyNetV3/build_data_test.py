import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import bemol
from bemol import tools
from bemol.data_processing import compute_forces, compute_velocity
import numpy as np
import brain

## Paramètres de l'éolienne
mexico_vortex = bemol.rotor.mexico_vortex
HR = mexico_vortex.hubRadius
TR = mexico_vortex.tipRadius

## Paramètres du modèle
preconeAngle = torch.tensor(0.0)
tiltAngle = torch.tensor(0.0)
pitch = np.radians(2.3000244769936637) # Blade pitch
omega = 44.5163679 # Rotation speed
yaw = np.radians(5) # Yaw skew angle
U = 25.04045694375 # Incoming stream's velocity
rho = 1.191 

## Solver
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


Test = [
    [torch.tensor([1,1,1]), torch.tensor([6])],
    [torch.tensor([1,1,0]), torch.tensor([6])]
]

type = torch.float64
samp_size = 72 ; batch_size = 9

"""
Construire les features : 
    - Un vecteur de taille 34 qui contient les efforts de la bem (corrigées par yaw ou non)
    - Une deuxième qui contient l'azimut (en degré) sur lequel la somme des forces est calculée.  
"""

path = 'data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv'
az,_,_ = brain.extract_line(path, 1)

F = torch.zeros((samp_size, 35))
for i in range(72) :
    forces = torch.zeros(34); 
    for j in range(34) : 
        velocities = tools.calculateVelocity(wind = U, omega = omega, rad = rotor.sections[j].radius, azi = az[i],
                                            yaw = yaw, tilt = tiltAngle, precone = preconeAngle)
        
        forces[j],_,_,_ = solver_yaw.solve(rotor.sections[j], az[i], pitch = pitch,
                                          velocity = velocities, angles = [yaw, tiltAngle])
    F[i, 0 : 34] = forces; F[i,34] = az[i]

L = torch.zeros(samp_size, 34)
for i in range(72) : 
    L[i, :] = brain.extract_column(path, i, dtype = type)

train_dataloader, val_dataloader = brain.data_organisation(F,L, batch_size=batch_size, test_size = 0.8, dtype = type)

k = 0
for f in train_dataloader :
    k+=1
k = 0
for l in val_dataloader :
    k+=1