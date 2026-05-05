import os
import sys

# Ajouter le chemin du parent dans sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd

import bemol
from bemol import tools

import matplotlib.pyplot as plt
import numpy as np
import pathlib as p
from tools_blade import write_blade as wb

## Paramètres de l'éolienne

## Paramètres du modèle (TSR == 8)
preconeAngle = 0.0
tiltAngle = 0.0
omega = 44.5163679 # Rotation speed
yawAngle = np.radians(5) # Yaw skew angle
U = 12.520228472 # Incoming stream's velocity
rho = 1.191 


## Solver
corrections = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]

path_dir_blade = p.Path('bemol/rotors/mexico_vortex')
blade_file = 'blade.dat'

blade_data = pd.read_csv(path_dir_blade/blade_file, delimiter='\s+', header = 0, names = ['radius', 'twist', 'chord', 'airfoil'])


"""

1.

    Tester pitch > 0 et twists < 0, on plot Fn & Ft sur plusieurs azimuths, on garde tous les 
    modèles de corrections activés 

"""

pitch = 0.040143

rotor1 = bemol.rotor.Rotor(path_dir_blade)
solver1 = bemol.ning.NingUncoupled(rotor1, rho, corrections)

azs = np.linspace(0,360, 36)
Fn1 = np.zeros((36,1,36))
Ft1 = np.zeros((36,1,36))
for i, az in enumerate(azs) :
    for r in range(36):
        print(f"\n{r}\n")
        velocities = tools.calculateVelocity(wind = U, omega = omega, rad = solver1.rotor.sections[r].radius, azi = az,
                                            yaw  = yawAngle, tilt = tiltAngle, precone = preconeAngle)
        Fn1[r,0,i],Ft1[r,0,i],_,_ = solver1.solve(solver1.rotor.sections[r], az, pitch, velocities, angles=[yawAngle, tiltAngle])


    



"""
2.

    Tester pitch < 0 et twists > 0, on plot Fn & Ft sur plusieurs azimuths, on garde tous les 
    modèles de corrections activés 

"""
pitch = 0.040143

pitch *= -1
wb.rewrite_geom(path_dir_blade/blade_file, 'twist', np.zeros((36)), -1)

rotor2 = bemol.rotor.Rotor(path_dir_blade)
solver2 = bemol.ning.NingUncoupled(rotor2, rho, corrections)


Fn2 = np.zeros((36,1,36))
Ft2 = np.zeros((36,1,36))
azs = np.linspace(0,360, 36)
for i,az in enumerate(azs) :
    for r in range(36) :
        print(f"\n{r}\n")
        velocities = tools.calculateVelocity(wind = U, omega = omega, rad = solver2.rotor.sections[r].radius, azi = az,
                                            yaw  = yawAngle, tilt = tiltAngle, precone = preconeAngle)
        Fn2[r,0,i],Ft2[r,0,i],_,_ = solver2.solve(solver2.rotor.sections[r], az, pitch, velocities, angles=[yawAngle, tiltAngle])


wb.rewrite_geom(path_dir_blade/blade_file, 'chord', np.zeros((36)), -1)


rad = blade_data['radius'].to_numpy()
plt.plot(rad, Ft2[:,0,0], label = f'azimuth = {0}°')
plt.plot(rad, Ft2[:,0,4], label = f'azimuth = {40}°')
plt.plot(rad, Ft2[:,0,9], label = f'azimuth = {90}°')
plt.plot(rad, Ft2[:,0,13], label = f'azimuth = {130}°')

#plt.plot(rad, Fn2, label = f'Pitch = {pitch}')

plt.grid()
plt.legend()
plt.xlabel("azimuths")
plt.ylabel("Ft N/m")

plt.show()