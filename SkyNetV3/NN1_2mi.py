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

type = torch.float32
samp_size = 72 ; batch_size = 9
index_element = 24
X,Y, element = brain.extract_line('data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv', index_element, type = type)


## Création des features et des labels
features = torch.zeros(samp_size, 2)
features[:,0] = X
for i in range(samp_size):
    velocities = tools.calculateVelocity(wind = U, omega = omega, rad = rotor.sections[element['indice']].radius, azi = X[i],
                                         yaw  = yaw, tilt = tiltAngle, precone = preconeAngle)
    
    features[i,1], _, _ ,_ = solver_yaw.solve(rotor.sections[element['indice']], X[i], pitch = pitch, 
                                       velocity=velocities, angles= [yaw, tiltAngle])

labels = torch.zeros((72,1), dtype = type)
labels[:,0] = Y

print("Azimuths : \n", features[0,:], "\n")
print("Efforts BEM : \n", features[1,:], "\n")
print("Efforts Vortex : \n", labels,'\n')

train_dataloader, val_dataloader = brain.data_organisation(features,labels, batch_size=batch_size, dtype = type)
print("Affichage du train_dataloader : \n")
for t in train_dataloader : 
    print(t,"\n")
print("\nAffichage du val_dataloader : \n")
for v in val_dataloader :
    print(v,"\n")

## Paramètres du réseaux de neurones
bias = True
in_size = 2
out_size = 1
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
MAX_EPOCH = 2
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

print("Début de l'entraînement\n")
start = perf_counter()
for ep in range(MAX_EPOCH):
    
    T800.train()    
    if ep % step == 0 :
        print("Etape "+str(ep + 1)+" sur " + str(MAX_EPOCH) + "\n")
    loss_train_int = list()
    rel_err_int = []
    start_train = perf_counter()
    for features, label in train_dataloader :
        features = features.unsqueeze(1).to(device)
        label = label.unsqueeze(1).to(device)
        optimiser.zero_grad()
        
        FN_NN = T800(features) ## Forces normale BEM 
        loss = loss_func(input=FN_NN.type(torch.float32), target=label)
        loss.backward()
        optimiser.step()
        loss_train_int.append(loss.detach().cpu().numpy())

        ## Relative error
        numerator = np.linalg.norm((FN_NN - label).detach().cpu().numpy())
        denominator = np.linalg.norm(label.detach().cpu().numpy())
        rel_error = numerator/denominator
        rel_err_int.append(rel_error)
    stop_train = perf_counter()
    
    #Gradient du réseau
    param = list(T800.named_parameters())
    Grad = torch.zeros((0,1), device = device)
    for i in range(len(param)) : 
        P = param[i][1].grad.view(-1, 1)
        Grad = torch.cat((Grad, P), dim = 0)    
    Norm_grad = torch.norm(Grad, dim = 0).item()    
    grad_norm.append(Norm_grad)

    loss_train.append(np.average(loss_train_int))
    rel_error_train.append(np.average(rel_err_int))

    if ep % step == 0 :
        print("ERREUR SUR ENTRAINEMENT")
        print("Temps de calcul : ", stop_train-start_train)
        print("Erreur absolue : " + str(loss_train[-1]))
        print("Erreur relative : " + str(rel_error_train[-1]), "\n")

    T800.eval()

    loss_val_int = list()
    rel_err_int = []
    start_val = perf_counter()
    for features, label in val_dataloader :

        features = features.to(device)
        label = label.to(device)

        features = features.unsqueeze(1)
        label = label.unsqueeze(1)
        
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
        print("ERREUR DE VALDIATION")
        print("Temps de calcul : ", stop_val - start_val)
        print("Erreur absolue : "+ str(loss_val[-1]))
        print("Erreur relative : " + str(rel_error_val[-1]), "\n")


fig1 = plt.figure(figsize=(30, 30))

plt.subplot(1,2,1)
plt.loglog(loss_train, color = 'b', label = 'Entraînement')
plt.loglog(loss_val, color = 'r', label = 'Validation')
plt.title('Erreur absolue')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.loglog(rel_error_train, color = 'b', label = 'Entraînement')
plt.loglog(rel_error_val, color = 'r', label = 'Validation')
plt.title('Erreur relative')
plt.legend()
plt.grid(True)

"""
Test ultime.
"""

fig2 = plt.figure(figsize=(30,30))

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
element = [index_element]
rad = solver_yawed.rotor.sections[element[0]].radius

forces, AI_yawed , azs = solver_yawed.cycle(
        mexico_vortex.pitchRated, U, omega, angles=[yawAngle, tiltAngle], tStep=0,
        n_phi=72, N=1, elements=element,
)
Fn_BEM = forces[:,0,0]


T800.eval()
Fn_SKN = np.zeros((72,))
NN_FN_out = []
for i in range(len(azs)) : 
    velocities = tools.calculateVelocity(wind = U, omega = omega, rad = solver_yawed.rotor.sections[element[0]].radius,
                                          azi = azs[i],
                                          yaw = yaw, tilt = tiltAngle, 
                                          precone = preconeAngle)
    FN_BEM,_,_,_ = solver_yaw.solve(solver_yawed.rotor.sections[element[0]],
                                     azs[i], pitch = pitch, velocity=velocities,
                                       angles = [yaw, tiltAngle])
    T = torch.tensor([azs[i], FN_BEM], dtype = torch.float32, device = device)
    Fn_SKN[i] = T800(T).to('cpu').detach().numpy()[0]
Fn_Vortex = Y



azs_vortex = np.linspace(0, 360, 72, endpoint=False)
azs = np.degrees(azs)


plt.subplot(1,1,1)
plt.plot(azs, Fn_BEM, color = 'green', label = 'Fn-BEM')
plt.plot(azs, Fn_SKN, color = 'red', label = 'Fn-SkyNet')
plt.plot(azs_vortex, Fn_Vortex, color = 'blue', label = 'Fn-Vortex')
plt.legend()
plt.grid()

plt.show()