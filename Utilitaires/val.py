import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter

PI = np.pi
tan = np.tan
sin = np.sin 

import os

import pandas as pd

## uncoment following lines if ModuleNotFounError
## add repo folder to PYTHONPATH, if running from the samples folder
import sys
sys.path.append(os.path.abspath(f'{__file__}/../..'))

import bemol
from bemol import data_processing
from bemol import tools

results_folder = 'results/yaw_models'
os.makedirs(results_folder,exist_ok=True)

##Déclaration préliminaire
AIProcess = data_processing.AxialInducProcessing
Yaw_corr = data_processing.yaw
Burton = data_processing.Burton
DefRad = data_processing.default_radius

##Paramètres du modèle
mexico = bemol.rotor.mexico
HR = mexico.hubRadius
TR = mexico.tipRadius
wind = mexico.windRated     #15.06 
omega = mexico.omegaRated   #44.5163679
pitch = mexico.pitchRated   #-0.040143

rho = 1.191                 #Air Density
N = 1.0                     #Nombre de révolution (float)
tStep = 0.1                 #Probablement un paramètre temporel (je croyais que la BEM était un modèle stationnaire)
elements = 24 # node close to the experimental data (r/R = 0.82)


azimuthAngle = 0.0
preconeAngle = 0.0
tiltAngle = 0.0
skewAngle = np.radians(30.)
yawAngle = skewAngle

samp = 180 
delta_phi = 2.0*PI*N/(samp) #Discrétisation du cercle parcouru par l'élément
azs = np.arange(0, 2*N*PI, delta_phi)

with open("azimuths.txt", "a", encoding='utf-8') as azs_fic:
    for val in azs:
        azs_fic.write(f"{val}\n")

corrections_yaw = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN,
]
solver_yawed = bemol.ning.NingUncoupled(mexico, rho, corrections_yaw)
section = solver_yawed.rotor.sections[elements]

corrections_unyaw = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.Dummy
]
solver_unyawed = bemol.ning.NingUncoupled(mexico, rho, corrections_unyaw)



az_val = azs[2]
angles =  [yawAngle, tiltAngle]
velo = tools.calculateVelocity(
    wind, omega, section.radius, az_val, yawAngle, tiltAngle, preconeAngle

)

fn_ref, _,  _ , _ = solver_yawed.solve(section, az_val, pitch, velocity=velo, angles=angles, tStep = tStep)

AI, treat = data_processing.AxialInducProcessing(solver_unyawed, az_val, angles[0], pitch=pitch)
wakeSkewAngle = data_processing.Burton(AI, angles[0])

corr = data_processing.yaw(wakeSkewAngle, az_val)

solver_unyawed._axial_induction *= corr
fn_test, _, _ ,_ = data_processing.AxialTreatment(solver_unyawed, **treat)

print(fn_test, "\t\t", fn_ref)

"""
####Solver avec le correctif de l'IFPEN
_, _, azs = solver_yawed.cycle(
        mexico.pitchRated, wind, omega, angles=[yawAngle, tiltAngle], tStep=tStep,
        n_phi=samp, N=N, elements=elements,
)
####Test de la correction que j'ai réécrite
forces_CopyCat = np.zeros((180,1))
solver_validation = bemol.ning.NingUncoupled(mexico, rho, corrections_unyaw)

for i, az in enumerate(azs) : 
    AI, treat = data_processing.AxialInducProcessing(solver_validation, az, yawAngle)
    
    wakeSkewAngle = Burton(AI, yawAngle)
    corr = Yaw_corr(wakeSkewAngle, az)

    solver_validation._axial_induction *= corr
    fn, _, _, _ = data_processing.AxialTreatment(solver_validation, **treat)
    forces_CopyCat[i,0] = fn
"""