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
samp_size = 2592 ; batch_size = 72

print("Début de l'extraction des données...\n")
print("----------------------------------------")
Pre_X, Y = brain.extract('data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv', 
                         type = type)

X = torch.zeros((samp_size,1,3), dtype=type)
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

    X[i,0,2],_,_,_ = solver_yaw.solve(rotor.sections[position], X[i,0,0], pitch = pitch, 
                                    velocity=velocities, angles = [yaw, tiltAngle])

train_dataloader, val_dataloader = brain.data_organisation(X,Y, batch_size=batch_size, dtype = type)

print("Données extraite avec succès et prête à l'utilisation !")
print("----------------------------------------")

## Paramètres du réseaux de neurones
bias = True
in_size = 3
out_size = 1
layer_size = 3000
deepness = 10
ReLU = nn.ReLU

T800 = brain.SkyNet(ReLU, in_size = in_size, out_size = out_size, 
              layer_size = layer_size,
              bias = bias, deepness = deepness, type = type, device = device
              )
transfert = True
path = "SkyNetV3/NN1_results/Parameters"
path_to_save, init_random = T800.init_weight(path = path, transfert = transfert, random_init = False, samp_size = samp_size)

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
param_saved = [T800.state_dict(), [0,0]]

loss_fic = os.path.join(BASE, "NN1_results/loss.txt")
with open(loss_fic, 'r', encoding='utf-8') as f : 
    res = f.read().split(',')
    param_saved[1][0] = float(res[0]) ; param_saved[1][1] = float(res[1]) 

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
        
        FN_NN = T800(features).squeeze(2).squeeze(1) ## Forces normale BEM 
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
    for features_val, label_val in val_dataloader :

        features_val = features_val.to(device)
        label_val = label_val.to(device)

        features_val = features_val.squeeze(1)
        label_val = label_val.squeeze(3).squeeze(2).squeeze(1).type(torch.float32)
        
        FN_NN_val = T800(features_val).squeeze(2).squeeze(1)
        loss = loss_func(input = FN_NN_val, target = label_val)

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

    if loss_val[-1] <= tol and loss_train[-1] <= tol : 
        ep_reach = ep
        print("L'erreur d'entrainement et de validation sont passée sous le seuil de 9e-7 en "+str(ep_reach))
        break
    
    ##Récupération des meilleurs paramètres obtenus
    if init_random == False :
        if param_saved[1][0] >= loss_train[-1] and param_saved[1][1] >= loss_val[-1]:
            param_saved[0] = T800.state_dict()
            param_saved[1][0] = loss_train[-1]; param_saved[1][1] = loss_val[-1]
            print("\n\n**************************************")
            print("Nouveau set de paramètres enregistré.")
            print("**************************************\n\n")
        else :
            """
            T800.load_state_dict(param_saved[0])
            print("L'epoch à été ignoré, faute de performance satisfaisante. Les anciens paramètres ont été rechargé.")
            """
            continue
    else : 
        if param_saved[1][0] >= loss_train[-1] and param_saved[1][1] >= loss_val[-1]:
            param_saved[0] = T800.state_dict()
            param_saved[1][0] = loss_train[-1]; param_saved[1][1] = loss_val[-1]
            print("\n\n**************************************")
            print("Nouveau set de paramètres enregistré.")
            print("**************************************\n\n")
        else : 
            continue

##Export des meilleurs paramètres trouvés
T800.save_param(path_to_save, True)
err_max_train = max(loss_train)
err_max_val = max(loss_train)
err_min_train = min(loss_train)
err_min_val = min(loss_val)
err_min_grad = min(grad_norm)

#######################################################################
#######################################################################
#######################################################################




#######################################################################
########################### PLOT ET LOG ############################### 
#######################################################################
 

compteur_fic = os.path.join(BASE, "NN1_results/compteur_NN1.txt")
compteur = 0
with open(compteur_fic,"r", encoding='utf-8') as f :
    compteur = int(f.read())
compteur += 1
with open(compteur_fic, "w", encoding='utf-8') as f :
    f.write(str(compteur))

with open(loss_fic, "w", encoding='utf-8') as f :
    f.write(f"{param_saved[1][0]+1000}, {param_saved[1][1]+1000}")

fig = plt.figure(figsize=(30, 30))

plt.subplot(1,3,1)
plt.loglog(loss_train, color = 'b', label = 'Entraînement')
plt.loglog(loss_val, color = 'r', label = 'Validation')
plt.title('Erreur absolue')
plt.legend()
plt.grid(True)

plt.subplot(1,3,2)
plt.loglog(rel_error_train, color = 'b', label = 'Entraînement')
plt.loglog(rel_error_val, color = 'r', label = 'Validation')
plt.title('Erreur relative')
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

T800.load_state_dict(param_saved[0])
T800.eval()
Fn_SKN = np.zeros((72,))
NN_FN_out = []
for i in range(len(azs)) : 
    velocities = tools.calculateVelocity(wind = U, omega = omega, rad = solver_yawed.rotor.sections[element[0]].radius,
                                          azi = X[i,0,0],
                                          yaw = yaw, tilt = tiltAngle, 
                                          precone = preconeAngle)
    FN_BEM,_,_,_ = solver_yaw.solve(solver_yawed.rotor.sections[elements[0]],
                                     azs[i], pitch = pitch, velocity=velocities,
                                       angles = [yaw, tiltAngle])
    T = torch.tensor([azs[i],rad, FN_BEM], dtype = torch.float32, device = device)
    Fn_SKN[i] = T800(T).to('cpu').detach().numpy()[0]
Fn_Vortex = Y[25*72:26*72]



azs_vortex = np.linspace(0, 360, 72, endpoint=False)
azs = np.degrees(azs)
plt.subplot(1,3,3)
plt.plot(azs, Fn_BEM, color = 'green', label = 'Fn-BEM')
plt.plot(azs, Fn_SKN, color = 'red', label = 'Fn-SkyNet')
plt.plot(azs_vortex, Fn_Vortex[:,0,0], color = 'blue', label = 'Fn-Vortex')
plt.legend()
plt.grid()

path_fig= os.path.join(BASE,"NN1_results/Graphiques/simulation" + str(compteur))
plt.savefig(path_fig)
plt.show()


##Ecriture des performances et autres informations relatives à différents tests 
file = os.path.join(BASE,"NN1_results/log_NN1.txt")
fic = open(file, "a", encoding='utf-8') 

fic.write("*******************************************************************************************************\n")
fic.write("Simulation numéro " + str(compteur) + "\n")
fic.write("Informations relatives à la simulation : \n")
fic.write("Fonction régressée : correction de l'IFPEN \n")
fic.write("Taille de l'échantillon prévelé : " + str(samp_size)+"\n")
fic.write("Learning rate : " + str(LR) + "\n")
fic.write("Nombre d'epochs parcouru : "+str(MAX_EPOCH)+"\n")
fic.write("Taille des batchs : " + str(batch_size) + "\n")

fic.write("\n\n")

fic.write("Informations relatives au réseau de neurone : \n")
fic.write("Biais ? " + str(bias)+"\n")
fic.write("Taille de l'entrée : " + str(in_size) + "\n")
fic.write("Taille de la sortie : " + str(out_size) + "\n")
fic.write("Nombre de neurone par couche : "+str(layer_size)+"\n")
fic.write("Nombre de couche : " + str(deepness)+"\n")
fic.write("Fonctions d'activation d'activation :" + str(nn.ReLU.__name__) + "\n")
fic.write("Fonction de perte : " + str(loss_func.__class__.__name__) + "\n")
fic.write("Méthode d'optimisation : " + str(optimiser.__class__.__name__) + "\n")
fic.write("Transfert ?" + str(transfert) + "\n")

fic.write("\n\n")

fic.write("Performances et résultat du modèle : \n")
fic.write("GPU ou CPU ?  " + str(device) + "\n")
fic.write(f"Plus grande perte sur l'entraînement : {err_max_train : .5e}\n")
fic.write(f"Plus petite perte sur l'entraînement : {err_min_train : .5e}\n")
fic.write(f"Plus grande perte sur la validation : {err_max_val : .5e}\n")
fic.write(f"Plus petite perte sur la validation : {err_min_val : .5e} \n")
fic.write(f"Norme de gradient minimale : {err_min_grad : .5e}\n")
fic.write("*******************************************************************************************************\n\n\n\n\n")
##############################################################################################################