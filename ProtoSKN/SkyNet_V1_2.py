import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter

import sys
import os
sys.path.append(os.path.abspath(f'{__file__}/../..'))

import bemol
from bemol import data_processing
from bemol import tools
from bemol import secondary

#######################################################################
#######################################################################
#######################################################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)

################## Hyperparamètres et NN ##################

LR = 1 * 1e-5
MAX_EPOCH = 50

in_size = 5
out_size = 1
layer_size = 3000
deepness = 3
bias = True

act = nn.ReLU #Activation 
a = act.__name__

class SkyNet(nn.Module) : 
    def __init__(self, deepness, dtype, bias:bool):
        super(SkyNet, self).__init__()
        self.deepness = deepness

        self.HL = nn.ModuleList()
        self.HL.append(nn.Linear(in_size, layer_size, bias = bool, dtype = dtype))

        for i in range(self.deepness) : 
            self.HL.append(nn.Linear(layer_size, layer_size, bool, dtype = dtype))
            self.HL.append(act(inplace=True))
            self.HL.append(nn.Linear(layer_size, layer_size, bool, dtype = dtype))

        self.HL.append(nn.Linear(layer_size, out_size, bool, dtype = dtype))

    def forward(self,x) : ##Parcourir le réseau
        for Layer in self.HL :
            x = Layer(x)
        return x

#######################################################################
#######################################################################
#######################################################################


#######################################################################
########################### DATA ######################################
#######################################################################

"""
    Cette version de SkyNet doit être capable de s'entraîner sur les rayons. Une fois que ça fonctionnera correctement, 
    on l'adaptera à SkyNet V2 puis on s'en servira pour écrire SkyNet V3.

    De plus, je ne tire plus aléatoirement les données, je crois que ce n'est pas utile, et c'est surement plus pertinent d'entraîner sur les mêmes données à chaque fois.
"""

pi = torch.pi 
type = torch.float32 #type des données
N = 1.0 #nombre de révolution
rho = 1.191 #densité de l'air
omega = 44.5163679 #vitesse de rotation, radian par seconde
mexico =  bemol.rotor.mexico #rotor
HR = bemol.rotor.mexico.hubRadius
TR = bemol.rotor.mexico.tipRadius

###Solver BEM

corrections_noyaw = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.Dummy
]
solver_noyaw = bemol.ning.NingUncoupled(mexico, rho, corrections_noyaw)


###Récupération des samples
n = 3
samp_size = 10**n
batch_size = 10**(n-1)

#Azimuths € [0, 2Npi]
delta_az  = 2*N*pi/samp_size
Azimuths = torch.arange(0.0, 2*N*pi + delta_az, delta_az, dtype = type)

#Yaw € [-pi/3, pi/3]
delta_yaw = 2*pi/(3*samp_size) 
Yaw = torch.arange(-pi/3, pi/3 + delta_yaw, delta_yaw, dtype = type)

#Wind € [10,20]
delta_wind = (15.06-10)/samp_size
Wind = torch.arange(10 , 15.06 + delta_wind, step = delta_wind, dtype = type) 

"""
    Pour chaque élément issu de la discrétisation de la pale, on calcule l'induction axiale sans correction pour chaque sample
"""
elements = torch.arange(10, 34, 1)

X = torch.zeros(len(elements) * samp_size, in_size)
Y = torch.zeros(len(elements) * samp_size)

#Pour calculer les labels
Burton = secondary.SkewAngle.Burton.__call__
Bur = secondary.SkewAngle.Burton()
Yaw_correction = secondary.YawModel.IFPEN.__call__
If = secondary.YawModel.IFPEN()

for i, el in enumerate(elements) : 

    ##Récupération de l'élément
    section = solver_noyaw.rotor.sections[el]
    r = solver_noyaw.rotor.radius[el]
    print("Element " + str(el) + " : " + str(r))

    for j in range(samp_size) : 
        psi = Azimuths[j]
        gamma = Yaw[j] 
        w = Wind[j]
        velocity = tools.calculateVelocity(
            w, omega, r, psi, gamma, 
            tilt = 0., #fixé à 0. par défaut
            precone = 0. #fixé à 0. par défaut
        )

        _, _, AI_nat, _ = solver_noyaw.solve(
            section, #éléments
            psi,  #azimuth
            -0.040143, #pitch
            velocity, #velocité
            [gamma, 0.0],  #[0] -> yaw , [1] -> tilt
            0.0 #tstep -> inertie entre l'arrivée du flux d'air et son action sur la turbine
        )

        #stockage des features calculées
        X[i*samp_size + j, 0] = el        
        X[i*samp_size + j, 1] = AI_nat
        X[i*samp_size + j, 2] = Yaw[j]
        X[i*samp_size + j, 3] = Azimuths[j]
        X[i*samp_size + j, 4] = Wind[j]

        #Calcul et stockage des labels 
        WakeSkewAngle = Burton(Bur,axialInduction = AI_nat, yawAngle = gamma)
        Y[i*samp_size + j] = Yaw_correction(If, 1., WakeSkewAngle, Azimuths[j], r, HR, TR)


        