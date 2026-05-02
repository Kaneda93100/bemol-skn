import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE = os.path.dirname(os.path.abspath(__file__))

import pathlib as p
import numpy as np
import pandas as pd
import bemol
from bemol import data_processing as DP
import matplotlib.pyplot as plt
from time import perf_counter

## Paramètres de l'éolienne
mexico_vortex = bemol.rotor.mexico_vortex
HR = mexico_vortex.hubRadius
TR = mexico_vortex.tipRadius

## Paramètres du modèle
preconeAngle = 0.0
tiltAngle = 0.0
pitch = mexico_vortex.pitchRated # Blade pitch
omega = 44.5163679 # Rotation speed
yaw = np.radians(15) # Yaw skew angle
U = 12.520228472 # Incoming stream's velocity
rho = 1.191 

## Solver
corrections = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]

solver = bemol.ning.NingUncoupled(mexico_vortex, rho, corrections)
rotor = solver.rotor

az = np.linspace(0,350,36) # degré
az = np.radians(az)        # radian
Data = np.zeros((36,36, 10))

for i in range(36) : # Azimuts  
    for j in range(36) : # Section de pale
        velocities = bemol.tools.calculateVelocity(wind = U, omega = omega, rad = rotor.sections[j].radius, azi = az[i],
                                            yaw = yaw, tilt = tiltAngle, precone = preconeAngle)
        


        fn,ft,ai,at = solver.solve(rotor.sections[j], az[i], pitch = pitch,
                                          velocity = velocities, angles = [yaw, tiltAngle])
        
        angle = rotor.sections[j].twist + pitch
        phi, aoa = DP.compute_inflow_aoa(solver, velocities[0], velocities[1], angle)
        V_eff = np.sqrt((U*(1-ai)**2 + (omega*rotor.sections[j].radius*(1+at)**2)))                            


        Data[i,j,0] = rotor.sections[j].radius
        Data[i,j,1] = np.degrees(az[i]) # degré
        Data[i,j,2] = np.degrees(yaw)
        Data[i,j,3] = 8
        Data[i,j,4] = V_eff
        Data[i,j,5] = aoa
        Data[i,j,6] = ai
        Data[i,j,7] = np.degrees(phi)
        Data[i,j,8] = fn
        Data[i,j,9] = ft

xlsx_data = pd.read_excel('DataBuilding/forces_BEM_Yaw (2).xlsx')

az1 = 0        # azimut == 0
az2 = 36*4     # azimut == 40
az3 = 36*100   # azimut == 100

phi_old1 = xlsx_data['phi (angle relatif)'].to_numpy()[az1:(az1+36)]
phi_old2 = xlsx_data['phi (angle relatif)'].to_numpy()[az2:(az2+36)]
phi_old3 = xlsx_data['phi (angle relatif)'].to_numpy()[az3:(az3+36)]

plt.plot(phi_old1, label = 'phi old')
plt.plot(Data[0,:,7], label = 'phi new')
plt.legend()
plt.grid()
plt.show()

x = 0
