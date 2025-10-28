import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## uncoment following lines if ModuleNotFounError
## add repo folder to PYTHONPATH, if running from the samples folder
import sys
sys.path.append(os.path.abspath(f'{__file__}/../..'))

import bemol


results_folder = 'results/yaw_models'
os.makedirs(results_folder,exist_ok=True)
mexico = bemol.rotor.mexico

wind = mexico.windRated  # 15.06
omega = 44.5163679
rho = 1.191
number_revolutions = 1.0
tStep = 0.1
elements = [24] # node close to the experimental data (r/R = 0.82)


azimuthAngle = 0.0
preconeAngle = 0.0
tiltAngle = 0.0

#preconeAngle = np.radians(preconeAngle)
#tiltAngle = np.radians(tiltAngle)
skewAngle = np.radians(30.)
yawAngle = skewAngle #np.radians(40.)

mexico = bemol.rotor.mexico

base_corrections = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN,
]


solver = bemol.ning.NingUncoupled(mexico,rho,base_corrections)
    
forces, axialInduction, azimuths = solver.cycle(
        mexico.pitchRated,wind,omega,angles=[yawAngle,skewAngle],tStep=tStep,
        n_phi=180,N=number_revolutions,elements=elements,
    )

azimuth_deg = np.degrees(azimuths)
    ## plot values
#plt.plot(axialInduction[:,0,0],axialInduction[:,0,1], label='IFPEN')
    
    ## export data
data = pd.DataFrame(
        {'azi':azimuth_deg,'fn':forces[:,0,0],'ft':forces[:,0,1]}
        )
data.to_csv(f'{results_folder}/results_yaw_model_{'IFPEN'}.csv')



## plot experimental data
# r/R = 0.82
# Final report of IEA-29 Phase 3, page 37, Figure 4.6 (d)
# scanned with webPlotDigitilizer
# data-points every 10 deg from 0

plt.ion()  # mode interactif pour l'animation
fig, ax = plt.subplots()
angles_deg = np.linspace(0, 85, 100)
fps = 1e-300  


for a in angles_deg:
    r = np.radians(a)
    forces, axialInduction, azimuths = solver.cycle(
        mexico.pitchRated, wind, omega, angles=[r, skewAngle, preconeAngle], tStep=tStep,
        n_phi=180, N=number_revolutions, elements=elements,
    )
    azimuth_deg = np.degrees(azimuths)
    ax.clear()
    ax.plot(axialInduction[:,0,0], axialInduction[:,0,1])
    ax.set_xlabel('axialInduction[:,0,0]')
    ax.set_ylabel('axialInduction[:,0,1]')
    ax.set_title(f'Yaw Angle = {a :.1f}°')
    ax.grid()
    plt.pause(fps)  # pause pour voir l'animation

plt.ioff()

plt.ion()
fig, ax = plt.subplots()
Yaw = 0.
angles_deg = np.linspace(0, 85, 100)

for a in angles_deg:
    r = np.radians(a)
    forces, axialInduction, azimuths = solver.cycle(
        mexico.pitchRated, wind, omega, angles=[r, r], tStep=tStep,
        n_phi=180, N=number_revolutions, elements=elements,
    )
    azimuth_deg = np.degrees(azimuths)
    ax.clear()
    ax.plot(axialInduction[:,0,0], axialInduction[:,0,1])
    ax.set_xlabel('axialInduction[:,0,0]')
    ax.set_ylabel('axialInduction[:,0,1]')
    ax.set_title(f'Skew Angle = {a :.1f}°')
    ax.grid()
    plt.pause(fps)

plt.ioff()

plt.show()
