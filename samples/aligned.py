"""

Sample code for aligned flow.

"""

import os

import pandas as pd
import matplotlib.pyplot as plt

## uncoment this to if ModuleNotFounError
## add repo folder to PYTHONPATH, if running from the samples folder
import sys
sys.path.append("../bemol-v0.0.1")

#print(sys.path)

import bemol

results_folder = 'results/aligned'
os.makedirs(results_folder,exist_ok=True)


##Paramètres du modèle Mexico
turbine = bemol.rotor.mexico #Extension du fichier
wind = turbine.windRated  #vitesse du vent (m/s)
omega = turbine.omegaRated #vitesse de rotation de l'hélice (rad/s)
pitch = turbine.pitchRated #orientation des pales de l'hélice
rho = 1.191 #densité de l'air (kg/m3) 

corrections = ( ##Etudier les différentes corrections apportées
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.YawModel.PittAndPeters,
    bemol.secondary.DynamicInflow.Dummy,
    bemol.secondary.TurbulentWakeState.Buhl,
    )

# all angles are null, flow completly aligned
skewAngle = 0.0 ##Je ne vois pas trop à quoi cela correspond 
yawAngle = 0.0  ##Yaw angle null --> le yaw system de l'éolienne est immobile
azimuthAngle = 0.0 ##Azimut par rapport au nord

##Chercher une méthode dans le script ning.py
solver_uncoupled = bemol.ning.NingUncoupled(turbine,rho,corrections)

forces, _ = solver_uncoupled.steady(
    azimuthAngle,pitch,wind,omega
)


data = pd.DataFrame({'fn':forces[:,0],'ft': forces[:,1]})
data.to_csv(f'{results_folder}/results_aligned_uncoupled.csv',index=False)


#Deux solvers sont appelés
solver_coupled = bemol.ning.NingCoupled(turbine,rho,corrections)

forces_coupled, _ = solver_coupled.steady(
    azimuthAngle,pitch,wind,omega,
    angles=[yawAngle,0.0],
    skew=0.0,
)

data = pd.DataFrame({'fn':forces_coupled[:,0],'ft':forces_coupled[:,1]})
data.to_csv(f'{results_folder}/results_aligned_coupled.csv',index=False)


lib_folder = os.path.abspath(os.path.dirname(bemol.__file__))

# get reference data for plot
##Solvers précalculés avec les modèles AeroDeep et CASTOR 

ref = {}
for solver in ('AeroDeeP','CASTOR'):
    ref[solver] = pd.read_csv(
        f'{lib_folder}/rotors/mexico/ref/data_{solver}.csv',
        index_col=None,comment='#',sep=',',
        )

for i, name in enumerate(['Fn','Ft']):

    # change orientation for tangent
    factor = -1 if name == 'Ft' else 1

    fig, ax = plt.subplots(1,1,constrained_layout=True)

    ax.plot(ref['AeroDeeP']['radius'],factor*ref['AeroDeeP'][name],
            '-ob',linewidth=1.0,markersize=3,label='AeroDeeP (BEM)')
    ax.plot(ref['CASTOR']['radius'],factor*ref['CASTOR'][name],
            '-sk',linewidth=1.0,markersize=3,label='CASTOR (FVW)')

    ax.plot(turbine.radius,factor*forces[:,i],'-b',label='uncoupled BEM')
    ax.plot(turbine.radius,factor*forces_coupled[:,i],'--r',label='coupled BEM')
    ax.legend()
    ax.grid()
    ax.set_xlabel('radius, m')
    ax.set_ylabel(f'{name}, N/m')
    fig.savefig(f'{results_folder}/graph_aligned_force_{name}.png')
    plt.show() # uncomment if you want to show the figure


