import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pathlib as p
import numpy as np
import pandas as pd
import bemol as bem
from bemol import data_processing as DP
import matplotlib.pyplot as plt

## Paramètres de l'éolienne
mexico_vortex = bem.rotor.mexico_vortex
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

## Init le solver
corrections = [
    bem.secondary.HubTipLoss.Prandtl,
    bem.secondary.SkewAngle.Burton,
    bem.secondary.TurbulentWakeState.Buhl,
    bem.secondary.YawModel.IFPEN
]
solver = bem.ning.NingUncoupled(mexico_vortex, rho, corrections)



def phi_vortex(aoa, twist) :
    return aoa + twist + pitch

def a_vor(phi, v_eff) :
    return 1 - np.sin(phi)*(v_eff/U)

def test_vor_bem(section:int):
    """
    Recalculer à azimut fixé l'induction axiale sur une certaine section.
    Les paramètres fixés sont :
        - Le yaw
        - Le tilt 
        - la section de pale
        - pitch
        - precone
        - tilt
        - omega
    """

    ## Init le solver
    corrections = [
        bem.secondary.HubTipLoss.Prandtl,
        bem.secondary.SkewAngle.Burton,
        bem.secondary.TurbulentWakeState.Buhl,
        bem.secondary.YawModel.IFPEN
    ]
    solver = bem.ning.NingUncoupled(mexico_vortex, rho, corrections)    
    
    ## Donnée BEM (BEM_ref)
    BEM_data = pd.read_excel('DataBuilding/forces_BEM_Yaw.xlsx')
    a_ref    = BEM_data['a'].to_numpy()
    phi_ref  = BEM_data['phi (angle relatif)'].to_numpy()

    ## récupérer les azimuths de data_vor + conversion en radians, récupérer les données vortex
    az = np.linspace(0,360,36, endpoint=False)
    az_rad = np.radians(az)
    data_vor = pd.read_excel('DataBuilding/dataset_forces_mexico.xlsx')

    ## récupérer (selon un rayon) les v_eff et les phi 
    twists       = data_vor['twist'].to_numpy()
    aoa          = data_vor['alpha'].to_numpy()
    V_eff        = data_vor['V_eff'].to_numpy()

    ## Partie Vortex
    phi_vor = np.zeros((36))
    ai_vor = np.zeros((36))
    for i in range(36) :
        phi_vor[i]   = phi_vortex(aoa[section + i*36], twists[section + i*36])
        ai_vor[i]   = a_vor(phi_vor[i], V_eff[section + i*36])

    ## Partie BEM
    ai_bem  = np.zeros((36))
    phi_bem = np.zeros((36))
    for i in range(36):
        ai_bem[i], phi_bem[i] = DP.AI_ning_alg(solver, section, U, omega, pitch, preconeAngle, tiltAngle, yaw, az_rad[i])

    ## Partie BEM ref
    ai_bem_ref  = np.zeros((36))
    phi_bem_ref = np.zeros((36))
    for i in range(36) :
        ai_bem_ref[i]  = a_ref[section + i*36]
        phi_bem_ref[i] = phi_ref[section + i*36]

    return ai_vor, phi_vor, ai_bem, phi_bem, ai_bem_ref, phi_bem_ref

R = [5, 15, 25, 35]
az = np.linspace(0,360,36, endpoint=False)


for r in R :
    ai_vor, phi_vor, ai_bem, phi_bem, ai_bem_ref, phi_bem_ref = test_vor_bem(r)

    fig = plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.plot(az, ai_vor, label = f'Vortex, section : {r}')
    plt.plot(az, ai_bem, label = f'BEM, section : {r}')
    plt.plot(az, ai_bem_ref, label = f'BEM_ref, section : {r}')
    plt.legend()
    plt.grid()
    plt.xlabel('azimuths')
    plt.ylabel('induction axiale')
    plt.title("Induction axiale Vortex vs BEM")

    plt.subplot(1,2,2)
    plt.plot(az, phi_vor, label = f'Vortex, section : {r}')
    plt.plot(az, phi_bem, label = f'BEM, section : {r}')
    plt.plot(az, phi_bem_ref, label =f'BEM_ref, section : {r}')
    plt.legend()
    plt.grid()
    plt.xlabel('azimuths')
    plt.ylabel('Angle de déviation relatif')
    plt.title("Angle de déviation relatif Vortex vs BEM")

    plt.savefig('DataBuilding/fig/'+str(r)+'.png')
    plt.close()
