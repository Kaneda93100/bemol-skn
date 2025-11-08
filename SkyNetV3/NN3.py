import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

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
pitch = 2.3000244769936637 # Blade pitch
omega = 44.5163679 # Rotation speed
yaw = np.radians(5) # Yaw skew angle
U = 25.04045694375 # Incoming stream's velocity
rho = 1.191 

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

## Format dans lequel les données sont manipulées
type = torch.float32

print("Début de l'extraction des données...\n")
print("----------------------------------------")
Pre_X, Y = brain.extract('data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv', 
                         type = type)

X = torch.zeros((2592,1,4), dtype=type)
X[:,0,0] = Pre_X[:,0,0] ## Azimuths
X[:,0,1] = Pre_X[:,0,1] ## Radius

elements = np.zeros_like(X[:,0,0], dtype=int)
for i in range(36) : ## Position 
    for j in range(72) :
        elements[72*i + j] = i

for i, position in enumerate(elements) : 
    velocities = tools.calculateVelocity(wind = U, omega = omega, rad = rotor.sections[position].radius, azi = X[i,0,0],
                                          yaw = yaw, tilt = tiltAngle, 
                                          precone = preconeAngle)

    _,_,X[i,0,2],X[i,0,3] = solver_yaw.solve(rotor.sections[position], X[i,0,0], pitch = pitch, ## Axial Induction --> X[:,0,2] | TangentialInduction --> X[:,0,3]
                                    velocity=velocities, angles = [yaw, tiltAngle])

train_dataloader, val_dataloader = brain.data_organisation(X,Y, batch_size=72, dtype = type)

print("Données extraite avec succès et prête à l'utilisation !\n")
print("----------------------------------------")

## Paramètres du réseaux de neurones
bias = True
in_size = 4
out_size = 1
layer_size = 3000
deepness = 10
ReLU = nn.ReLU

T800 = brain.SkyNet(ReLU, in_size = in_size, out_size = out_size, 
              layer_size = layer_size,
              bias = bias, deepness = deepness, type = type, device = device
              )

path = "SkyNetV3/NN1_results/Parameters"
path_to_save = T800.init_weight(path = path, transfert = False, random_init = False ,samp_size = 2592)

#######################################################################
########################### ENTRAINEMENT ##############################
#######################################################################

LR = 1e-4
MAX_EPOCH = 100
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
param_saved = [T800.state_dict(), 0]


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
        features = features.to(device)
        label = label.to(device)
        optimiser.zero_grad()
        label = label.squeeze(2).squeeze(1).type(torch.float32)
        
        AI_NN = T800(features).squeeze(2).squeeze(1) ## Induction axiale par réseau de neurones
        Ux, Uy = compute_velocity(wind = U, omega = omega,
                                  rad = features[:,0,1], azimuth = features[:,0,0],
                                  yaw=yaw, tilt=tiltAngle, precone=preconeAngle
                                  )
        fn_NN_list = []
        for i, data in enumerate(features) : 
            fn, _ = compute_forces(solver = solver_noyaw, Ux = Ux[i], Uy = Uy[i], 
                                    x = AI_NN[i], y = features[i,0,3], 
                                    index = features[i,0,2]
                                    )
            fn_NN_list.append(fn)
        FN_NN = torch.stack(fn_NN_list)
        
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

    #Gradient du réseau
    param = list(T800.named_parameters())
    Grad = torch.zeros((0,1), device = device)
    for i in range(len(param)) : 
        P = param[i][1].grad.view(-1, 1)
        Grad = torch.cat((Grad, P), dim = 0)    
    Norm_grad = torch.norm(Grad, dim = 0).detach().cpu().numpy()       
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
    for features_val, label_val in val_dataloader :
        
        features_val = features_val.to(device)
        label_val = label_val.to(device)

        features_val = features_val.squeeze(1)
        label_val = label_val.squeeze(2).squeeze(1).type(torch.float32)
        AI_NN = T800(features_val).squeeze(2).squeeze(1)
        Ux, Uy = compute_velocity(wind = U, omega = omega,
                                  rad = features_val[:,0,1], azimuth = features_val[:,0,0],
                                  yaw=yaw, tilt=tiltAngle, precone=preconeAngle
                                  )
        fn_NN_list_val = []
        for i, data in enumerate(features) : 
            fn, _ = compute_forces(solver = solver_noyaw, Ux = Ux[i], Uy = Uy[i], 
                                    x = AI_NN[i], y = features_val[i,0,3], 
                                    index = features_val[i,0,2]
                                    )
            fn_NN_list_val.append(fn)
        FN_NN_val = torch.stack(fn_NN_list_val)
        
        loss = loss_func(input = FN_NN_val, target = label_val)

        loss_val_int.append(loss.detach().cpu().numpy())

        #Relative error
        numerator = np.linalg.norm((FN_NN - label).detach().cpu().numpy())
        denominator = np.linalg.norm(label.detach().cpu().numpy())
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

    if loss_val[-1] <= tol and loss_train[-1] <= tol : 
        ep_reach = ep
        print("L'erreur d'entrainement et de validation sont passée sous le seuil de 9e-7 en "+str(ep_reach))
        break
    
    ##Récupération des meilleurs paramètres obtenus
    if ep == 0 :
        param_saved[1] = loss_train[-1]
    else : 
        if param_saved[1] >= loss_train[-1] :
            param_saved[0] = T800.state_dict()
            param_saved[1] = loss_train[-1]
            print("\n\n**************************************")
            print("Nouveau set de paramètres enregistré.")
            print("**************************************\n\n")
        else : 
            continue



##Export des meilleurs paramètres trouvés
T800.save_param("", False)

fig = plt.figure(figsize=(20, 14))

#Perte sur entraînement & sur validation
plt.loglog(loss_train, color = 'r', label = 'Entraînement')
plt.loglog(loss_val, color = 'b', label = 'Validation')
plt.title('Evolution de la perte')
plt.legend()
plt.grid(True)

"""
Test ultime.
"""
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
element = [24]
rad = solver_yawed.rotor.sections[element[0]].radius

forces, AI_yawed , azs = solver_yawed.cycle(
        mexico_vortex.pitchRated, U, omega, angles=[yawAngle, tiltAngle], tStep=0,
        n_phi=72, N=1, elements=element,
)
Fn_BEM = forces[:,0,0]

T800.state_dict(param_saved[0])
T800.eval()
Fn_SKN = np.zeros((72,))
NN_FN_out = []
for i in range(len(azs)) : 
    velocities = tools.calculateVelocity(wind = U, omega = omega, rad = solver_yawed.rotor.sections[element[0]].radius,
                                          azi = X[i,0,0],
                                          yaw = yaw, tilt = tiltAngle, 
                                          precone = preconeAngle)
    _,_,AI_BEM,TI_BEM = solver_yaw.solve(solver_yawed.rotor.sections[elements[0]],
                                     azs[i], pitch = pitch, velocity=velocities,
                                       angles = [yaw, tiltAngle])
    T = torch.tensor([azs[i],rad, AI_BEM, TI_BEM], dtype = torch.float32, device = device)
    Fn_SKN[i] = T800(T).to('cpu').detach().numpy()[0]
Fn_Vortex = Y[25*72:26*72]



azs_vortex = np.linspace(0, 360, 72, endpoint=False)
fig_fn = plt.figure(figsize = (20,15))
azs = np.degrees(azs)
plt.plot(azs, Fn_BEM, color = 'green', label = 'Fn-BEM')
plt.plot(azs, Fn_SKN, color = 'red', label = 'Fn-SkyNet')
plt.plot(azs_vortex, Fn_Vortex[:,0,0], color = 'blue', label = 'Fn-Vortex')
plt.legend()
plt.grid()

plt.show()