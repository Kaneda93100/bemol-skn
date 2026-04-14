import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE = os.path.dirname(os.path.abspath(__file__))

import torch
from torch import nn, optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import brain
import numpy as np
import pandas as pd
import bemol
from bemol import tools
from bemol.data_processing import compute_forces, compute_velocity
import matplotlib.pyplot as plt
from time import perf_counter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)

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

type = torch.float64
samp_size = 72 ; batch_size = 9

"""
Construire les features : 
    - Un vecteur de taille 34 qui contient les efforts de la bem (corrigées par yaw ou non)
    - Une deuxième qui contient l'azimut (en radian) sur lequel la somme des forces est calculée.  
"""

path = 'data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv'
az,_,_ = brain.extract_line(path, 1)

F = torch.zeros((samp_size, 37))
for i in range(72) :
    forces = torch.zeros(36); 
    for j in range(36) : 
        velocities = tools.calculateVelocity(wind = U, omega = omega, rad = rotor.sections[j].radius, azi = az[i],
                                            yaw = yaw, tilt = tiltAngle, precone = preconeAngle)
        
        forces[j],_,_,_ = solver_yaw.solve(rotor.sections[j], az[i], pitch = pitch,
                                          velocity = velocities, angles = [yaw, tiltAngle])
    F[i, 0 : 36] = forces; F[i,36] = az[i]

"""
Construire les labels :
    - La force calculées par le vortex (issu de la feuille ..\\ data_vortex_yaw_mexico\\data_vortex_mexico_tsr004_yaw005_fn.csv)
"""

L = torch.zeros(samp_size, 36)
for i in range(72) : 
    L[i, :] = brain.extract_column(path, i, dtype = type)

train_dataloader, val_dataloader = brain.data_organisation(F,L, batch_size=batch_size, test_size = 0.8, dtype = type)

val_azimuth = []
for l in val_dataloader : 
    for j in range(len(l[0])) :
        val_azimuth.append(l[0][j][-1])

## Paramètres du réseaux de neurones
bias = True
in_size = 37
out_size = 36
layer_size = 3000
deepness = 10
ReLU = nn.ReLU

T800 = brain.SkyNet(ReLU, in_size = in_size, out_size = out_size, 
              layer_size = layer_size,
              bias = bias, deepness = deepness, type = type, device = device
              )


#######################################################################
########################### ENTRAINEMENT ##############################
#######################################################################

LR = 1e-4
MAX_EPOCH = 50
#Erreur absolue
loss_train= []
loss_val = []

#Erreur relative
rel_error_train = []
rel_error_val = []

#Norme du gradient
grad_norm = []

optimiser = optim.Adam(T800.parameters(), lr = LR, weight_decay=1)
loss_func = nn.MSELoss(reduction = 'mean')

#Gestion de l'affichage 
step  = 1 #étape à afficher dans le terminal
tronc = 5 #ordre de la troncature pour err_min_train et err_min_val
tol = 1e-7 #seuil de tolérance pour l'erreur faite sur les données d'entraînement ET de validation


print("\n***Début de l'entraînement***\n")
start = perf_counter()
for ep in range(MAX_EPOCH):
    T800.train()    

    if ep % step == 0 :
        print("Etape "+str(ep + 1)+" sur " + str(MAX_EPOCH) + "\n")
    
    loss_train_int = list()
    rel_err_int = []
    
    start_train = perf_counter()
    for features, label in train_dataloader :
        features = features.to(device)
        label = label.to(device)
        optimiser.zero_grad()
        
        FN_NN = T800(features) ## Forces normale BEM 
        loss = loss_func(input=FN_NN, target=label)
        loss.backward()
        optimiser.step()
        loss_train_int.append(loss.detach().cpu().numpy())

        ## Relative error
        numerator = np.linalg.norm((FN_NN - label).detach().cpu().numpy())
        denominator = np.linalg.norm(label.detach().cpu().numpy())
        rel_error = numerator/denominator
        rel_err_int.append(rel_error)
    stop_train = perf_counter()

    loss_train.append(np.average(loss_train_int))
    rel_error_train.append(np.average(rel_err_int))

    if ep % step == 0 :
        print("ERREUR SUR ENTRAINEMENT")
        print("Temps de calcul : ", stop_train-start_train)
        print("Erreur absolue : " + str(format(loss_train[-1], ".1e")))
        print("Erreur relative : " + str(format(rel_error_train[-1], ".1e")), "\n")

    T800.eval()

    loss_val_int = list()
    rel_err_int = []
    start_val = perf_counter()
    for features, label in val_dataloader :

        features = features.to(device)
        label = label.to(device)
        
        FN_NN = T800(features)
        loss = loss_func(input = FN_NN, target = label)

        loss_val_int.append(loss.detach().cpu().numpy())

        #Relative error
        numerator = np.linalg.norm((FN_NN - label).detach().cpu().numpy())
        denominator = np.linalg.norm(label.detach().cpu())
        rel_error = numerator/denominator
        rel_err_int.append(rel_error)
    stop_val = perf_counter()

    loss_val.append(np.average(loss_val_int))
    rel_error_val.append(np.average(rel_err_int))


    if ep % step == 0 :
        print("ERREUR SUR VALDIATION")
        print("Temps de calcul : ", stop_val - start_val)
        print("Erreur absolue : "+ str(format(loss_val[-1], ".1e")))
        print("Erreur relative : " + str(format(rel_error_val[-1], ".1e")), "\n")



corr = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]
solver_yawed = bemol.ning.NingUncoupled(mexico_vortex, rho, corr)

azimuthAngle = 0.0
preconeAngle = 0.0
tiltAngle = 0.0
skewAngle = np.radians(5.)
yawAngle = skewAngle
element = [17]
rad = solver_yawed.rotor.sections[element[0]].radius

"""
Premier test : Tester la capacité du réseau à calculer la distribution des efforts normaux 
               sur la pale pour un azimut donné (sur lequel il ne s'est pas entraîné)
"""

az1 = val_azimuth[3].detach().numpy()  
az2 = val_azimuth[24].detach().numpy()
az3 = val_azimuth[40].detach().numpy()
az4 = val_azimuth[57].detach().numpy()
AZ_test = [az1, az2, az3, az4]

Fn_Vortex = torch.zeros((36, 4))

Fn_Vortex[:, 0] = brain.extract_column(path, 22) ## 110°
Fn_Vortex[:, 1] = brain.extract_column(path, 61) ## 305° 
Fn_Vortex[:, 2] = brain.extract_column(path, 36) ## 180°
Fn_Vortex[:, 3] = brain.extract_column(path, 46) ## 230°

T800.eval()
Fn_SKN = np.zeros((36,4))
food_NN = torch.zeros(37, device = device, dtype = type)
for i in range(4) :
    for j in range(36) : 
            velocities = tools.calculateVelocity(wind = U, omega = omega, rad = rotor.sections[j].radius, azi = AZ_test[i],
                                                yaw = yaw, tilt = tiltAngle, precone = preconeAngle)
            
            food_NN[j],_,_,_ = solver_yaw.solve(rotor.sections[j], AZ_test[i], pitch = pitch,
                                            velocity = velocities, angles = [yaw, tiltAngle])
    food_NN[36] = torch.tensor(AZ_test[i], device = device, dtype = type)
    Fn_SKN[:,i] = T800(food_NN).to('cpu').detach().numpy()
    food_NN = torch.zeros(37, device = device, dtype = type)




radius = brain.extract_rad("data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv")

fig = plt.figure(figsize=(10, 10))
fig.suptitle("Distribution des efforts normaux à azimut fixé (SkyNet vs Vortex)")

plt.subplot(2,2,1)
plt.plot(radius, Fn_SKN[:,0], color = 'red', label = 'Fn-SkyNet')
plt.plot(radius, Fn_Vortex[:,0], color = 'blue', label = 'Fn-Vortex')
plt.title('Azimut à 110°')
plt.xlabel('rayon')
plt.ylabel('Force normale')
plt.legend()
plt.grid()

plt.annotate(f"Erreur (relative) sur la distribution : {np.linalg.norm(Fn_SKN[:,0] - Fn_Vortex[:,0].numpy())/np.linalg.norm(Fn_Vortex[:,0].numpy()):.2e}",
             xy=(0.5, -0.25), xycoords='axes fraction',
             ha='center', fontsize=12)


plt.subplot(2,2,2)
plt.plot(radius, Fn_SKN[:,1], color = 'red', label = 'Fn-SkyNet')
plt.plot(radius, Fn_Vortex[:,1], color = 'blue', label = 'Fn-Vortex')
plt.title('Azimut à 305°')
plt.xlabel('rayon')
plt.ylabel('Force normale')
plt.legend()
plt.grid()
plt.annotate(f"Erreur (relative) sur la distribution : {np.linalg.norm(Fn_SKN[:,1] - Fn_Vortex[:,1].numpy())/np.linalg.norm(Fn_Vortex[:,1].numpy()):.2e}",
             xy=(0.5, -0.25), xycoords='axes fraction',
             ha='center', fontsize=12)

plt.subplot(2,2,3)
plt.plot(radius, Fn_SKN[:,2], color = 'red', label = 'Fn-SkyNet')
plt.plot(radius, Fn_Vortex[:,2], color = 'blue', label = 'Fn-Vortex')
plt.title('Azimut à 180°')
plt.xlabel('rayon')
plt.ylabel('Force normale')
plt.legend()
plt.grid()
plt.annotate(f"Erreur (relative) sur la distribution : {np.linalg.norm(Fn_SKN[:,2] - Fn_Vortex[:,2].numpy())/np.linalg.norm(Fn_Vortex[:,2].numpy()):.2e}",
             xy=(0.5, -0.25), xycoords='axes fraction',
             ha='center', fontsize=12)

plt.subplot(2,2,4)
plt.plot(radius, Fn_SKN[:,3], color = 'red', label = 'Fn-SkyNet')
plt.plot(radius, Fn_Vortex[:,3], color = 'blue', label = 'Fn-Vortex')
plt.title('Azimut à 230°')
plt.xlabel('rayon')
plt.ylabel('Force normale')
plt.legend()
plt.grid()
plt.annotate(f"Erreur (relative) sur la distribution : {np.linalg.norm(Fn_SKN[:,3] - Fn_Vortex[:,3].numpy())/np.linalg.norm(Fn_Vortex[:,3].numpy()):.2e}",
             xy=(0.5, -0.25), xycoords='axes fraction',
             ha='center', fontsize=12)

plt.subplots_adjust(hspace=0.4)
plt.show()


"""
Deuxième test :  Avec ce modèle, tester son calcul des distributions des forces 
                 à élément fixé sur une révolution complète.
                 C'est l'occasion de tester sa perf sur la BEM.
"""

azimuthAngle = 0.0
preconeAngle = 0.0
tiltAngle = 0.0
skewAngle = np.radians(5.)
yawAngle = skewAngle
element = [24]

## Forces calculées par la BEM native
forces, AI_yawed , azs = solver_yawed.cycle(
        mexico_vortex.pitchRated, U, omega, angles=[yawAngle, tiltAngle], tStep=0,
        n_phi=72, N=1, elements=element,
)
Fn_BEM = forces[:,0,0]

## Forces Vortex pour l'éléments 24 ~ ?
_,Fn_Vortex,_ = brain.extract_line(path, 24)

T800.eval()
Fn_SKN = np.zeros((72,))
NN_FN_out = []
for i in range(len(azs)) : 
    food_NN = torch.zeros(37, device = device, dtype = type)
    for j in range(36) :
        velocities = tools.calculateVelocity(wind = U, omega = omega, rad = solver_yawed.rotor.sections[j].radius,
                                            azi = azs[i],
                                            yaw = yaw, tilt = tiltAngle, 
                                            precone = preconeAngle)
        
        food_NN[j],_,_,_ = solver_yaw.solve(solver_yawed.rotor.sections[j],
                                        azs[i], pitch = pitch, velocity=velocities,
                                        angles = [yaw, tiltAngle])
    food_NN[36] = azs[i]
    FN_dis = T800(food_NN).to('cpu').detach().numpy()
    Fn_SKN[i] = FN_dis[24]

fig_test2 = plt.figure(figsize = (10,10))
plt.suptitle("Effort sur un élément à travers une révolution")

plt.subplot(1,2,1)
plt.plot(azs, Fn_BEM, color = 'green', label = 'Fn-BEM')
plt.plot(azs, Fn_SKN, color = 'red', label = 'Fn-SkyNet')
plt.plot(azs, Fn_Vortex, color = 'blue', label = 'Fn-Vortex')
plt.title("Distribution des efforts sur un élément au cour d'une révolution")
plt.legend()
plt.grid() 

err_SKN_Vortex = np.abs(Fn_Vortex - Fn_SKN)
plt.subplot(1,2,2)
plt.plot(azs, err_SKN_Vortex)
plt.title("Erreur sur le calcul de la distribution")
plt.xlabel("azimut (en degré)")
plt.legend()
plt.grid()

plt.show()
