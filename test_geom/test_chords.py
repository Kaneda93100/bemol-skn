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
mexico_vortex = bemol.rotor.mexico_vortex
HR = mexico_vortex.hubRadius
TR = mexico_vortex.tipRadius
pitch = mexico_vortex.pitchRated

## Paramètres du modèle
U = 25.04045694375 # Incoming stream's velocity
omega = 44.5163679 # Rotation speed, TSR == 4
rho = 1.191 
azimuthAngle = 0.0
preconeAngle = 0.0
tiltAngle = 0.0
skewAngle = np.radians(5.)
yawAngle = skewAngle
element = [24]


## Solver
corrections = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]

path_blade = p.Path('bemol/rotors/mexico_vortex/blade.dat')
blade_data = pd.read_csv(path_blade, delimiter='\s+', header = 0, names = ['radius', 'twist', 'chord', 'airfoil'])

az = np.radians(10)

"""
Test réalisé sur la section 24 avec la liste des chrods de base

"""
solver = bemol.ning.NingUncoupled(mexico_vortex, rho, corrections)
rad = solver.rotor.sections[element[0]].radius
Fn1 = np.zeros((36))
for r in range(36) :
    velocities = tools.calculateVelocity(wind = U, omega = omega, rad = solver.rotor.sections[r].radius, azi = az,
                                         yaw  = yawAngle, tilt = tiltAngle, precone = preconeAngle)
    Fn1[r],_,_,_ = solver.solve(solver.rotor.sections[r], az, pitch, velocities, angles=[yawAngle, tiltAngle])


"""
Modification des chords : on multiplie les chords par 10

"""

wb.rewrite_geom(path_blade, 'chord', np.zeros((36)), 10.)

new_rotor = bemol.rotor.Rotor('bemol/rotors/mexico_vortex')

mexico_vortex2 = bemol.rotor.mexico_vortex
solver = bemol.ning.NingUncoupled(new_rotor, rho, corrections)

Fn2 = np.zeros((36))
for r in range(36) :
    velocities = tools.calculateVelocity(wind = U, omega = omega, rad = solver.rotor.sections[r].radius, azi = az,
                                         yaw  = yawAngle, tilt = tiltAngle, precone = preconeAngle)
    Fn2[r],_,_,_ = solver.solve(solver.rotor.sections[r], az, pitch, velocities, angles=[yawAngle, tiltAngle])


wb.rewrite_geom(path_blade, 'chord', np.zeros((36)), 1/10.)

rad = blade_data['radius'].to_numpy()
plt.plot(rad, Fn1, label = 'chord de base')
plt.plot(rad, Fn2, label = 'chord modifié')

plt.grid()
plt.legend()
plt.xlabel("azimuths")
plt.ylabel("Fn N/m")

plt.show()