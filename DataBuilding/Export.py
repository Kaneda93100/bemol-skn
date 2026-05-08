"""
Exporter les données BEM pour pouvoir relancer l'entraînement de Skynet. 

Géométrie utilisée :
    Mexico_Vortex avec la distribution affinée des profils. Utilisée pour les simulations CASTOR :
    - Pitch négatif (== -0.040143) 
    - twists positifs

Paramètres de la simulation : 
    - Yaw       == 15°
    - TSR       == 8
    - omega     == 44.5163679
    - U         == 12.520228472 
    - rho       == 1.191
    - precone   == 0.0
    - tilt      == 0.0

Modèle de corrections appliquées : 
    - Hub/Tip loss          : Prandlt
    - SkewAngle             : Burton
    - TurbulentWakeState    : Buhl
    - YawModel              : IFPEN

Données récupérées : 
    - rayon de la section (36)
    - azimuth (36)
    - yaw (un seul)
    - TSR (un seul)
    - V_eff (36*36) --> sqrt([Ux(1-a)]² +[Uy(1+a')]²)
    - aoa (36*36)   --> alpha = phi[r] + pitch + twist[r] 
    - ai (36*36)    --> Calculé avec Ning 
    - phi (36*36)   --> Calculé avec une vitesse 2D : phi = np.arctan2(Ux*(1-a), Uy*(1+a'))
    - fn (36*36)    --> Calculé à partir de Cn et Ct (formules aérodynamiques classique)
    - ft (36*36)    --> Calculé à partir de Cn et Ct (formules aérodynamiques classique)

Retourné en .xlsx (excel)
"""
import sys
import os

project_root = '/home/arthur/Documents/GitHub/bemol-skn'
if project_root not in sys.path:
    sys.path.append(project_root)

import pathlib as p
import numpy as np
import pandas as pd
import openpyxl as op
import bemol
from tools_blade import write_blade as wb
from bemol import data_processing as DP
import matplotlib.pyplot as plt

## Paramètres du modèle
preconeAngle = 0.0
tiltAngle = 0.0
omega = 44.5163679 # Rotation speed
yaw = np.radians(15) # Yaw skew angle
U = 12.520228472 # Incoming stream's velocity
rho = 1.191 

## Charger le rotor
pitch = 0.040143
#wb.rewrite_geom(path = 'bemol/rotors/mexico_vortex/blade.dat', attribut = 'twist', y = np.zeros((36)), lbd = -1.)
rotor = bemol.rotor.Rotor('/home/arthur/Documents/GitHub/bemol-skn/bemol/rotors/mexico_vortex')

#####################################################################################################
###                          DONNÉEs : YAW ON                                                     ###
#####################################################################################################

## Initialiser le solveur BEM
corrections = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]
solver = bemol.ning.NingUncoupled(rotor, rho, corrections)

## Construire les données

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
        V_eff = np.sqrt((U*(1-ai))**2 + (omega*rotor.sections[j].radius*(1+at))**2)                            


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


## Exporter vers .xlsx
sheet = op.Workbook()
write = sheet.active

attr = ['r', 'theta', 'yaw', 'TSR', 'V_eff', 'alpha', 'a', 'phi', 'Fn', 'Ft']
for ind, at in enumerate(attr, start = 1) :
    write.cell(row = 1, column = ind, value = at) 

for i in range(36) : # azimuth
    for j in range(36) : # radius
        write.cell(row = 36*i + (j+2), column = 1, value = Data[i,j,0])  # radius   A
        write.cell(row = 36*i + (j+2), column = 2, value = Data[i,j,1])  # azimuth  B
        write.cell(row = 36*i + (j+2), column = 3, value = Data[i,j,2])  # yaw      C
        write.cell(row = 36*i + (j+2), column = 4, value = Data[i,j,3])  # TSR      D
        write.cell(row = 36*i + (j+2), column = 5, value = Data[i,j,4])  # V_eff    E
        write.cell(row = 36*i + (j+2), column = 6, value = Data[i,j,5])  # aoa      F
        write.cell(row = 36*i + (j+2), column = 7, value = Data[i,j,6])  # ai       G
        write.cell(row = 36*i + (j+2), column = 8, value = Data[i,j,7])  # phi      H
        write.cell(row = 36*i + (j+2), column = 9, value = Data[i,j,8])  # fn       I
        write.cell(row = 36*i + (j+2), column = 10, value = Data[i,j,9]) # ft       J

sheet.save('DataBuilding/BEM_data_YAW_on.xlsx')

solver = None
corrections = None

#####################################################################################################
##                         DONNÉES : YAW OFF                                                       ##
#####################################################################################################
## Initialiser le solveur BEM
yaw = 0.
corrections = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.Dummy
]
solver = bemol.ning.NingUncoupled(rotor, rho, corrections)

## Construire les données
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
wb.rewrite_geom(path = 'bemol/rotors/mexico_vortex/blade.dat', attribut = 'twist', y = np.zeros((36)), lbd = -1.) # Remettre le signe de départ sur les twists


## Exporter vers .xlsx
sheet = op.Workbook()
write = sheet.active

attr = ['r', 'theta', 'yaw', 'TSR', 'V_eff', 'alpha', 'a', 'phi', 'Fn', 'Ft']
for ind, at in enumerate(attr, start = 1) :
    write.cell(row = 1, column = ind, value = at) 

for i in range(36,) : # azimuth
    for j in range(36) : # radius
        write.cell(row = 36*i + (j+2), column = 1, value = Data[i,j,0])  # radius   A
        write.cell(row = 36*i + (j+2), column = 2, value = Data[i,j,1])  # azimuth  B
        write.cell(row = 36*i + (j+2), column = 3, value = Data[i,j,2])  # yaw      C
        write.cell(row = 36*i + (j+2), column = 4, value = Data[i,j,3])  # TSR      D
        write.cell(row = 36*i + (j+2), column = 5, value = Data[i,j,4])  # V_eff    E
        write.cell(row = 36*i + (j+2), column = 6, value = Data[i,j,5])  # aoa      F
        write.cell(row = 36*i + (j+2), column = 7, value = Data[i,j,6])  # ai       G
        write.cell(row = 36*i + (j+2), column = 8, value = Data[i,j,7])  # phi      H
        write.cell(row = 36*i + (j+2), column = 9, value = Data[i,j,8])  # fn       I
        write.cell(row = 36*i + (j+2), column = 10, value = Data[i,j,9]) # ft       J

sheet.save('DataBuilding/BEM_data_YAW_off.xlsx')


print("\nJob done\n")