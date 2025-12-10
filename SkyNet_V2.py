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

import os

import pandas as pd

## uncoment following lines if ModuleNotFounError
## add repo folder to PYTHONPATH, if running from the samples folder
import sys
sys.path.append(os.path.abspath(f'{__file__}/../..'))

import bemol
from bemol import data_processing

plt.rcParams['lines.linewidth'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.style.use('seaborn-v0_8-poster')

sin = data_processing.sin
tan = data_processing.tan
PI = torch.pi

##################################################################################################
################ Paramètres intrasèques au modèle, donc considérés comme constant ################
##################################################################################################

wind = data_processing.wind
omega = data_processing.omega
pitch = data_processing.pitch
rho = 1.191
N = 1.0 #number of revolutions
tStep = 0.1
elements = [24]

mexico = bemol.rotor.mexico
HR = mexico.hubRadius
TR = mexico.tipRadius

preconeAngle = 0.0
tiltAngle = 0.0
#######################################################################
#######################################################################
#######################################################################




#######################################################################
###################### RESEAU DE NEURONES SKYNET ###################### 
#######################################################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)

################## Hyperparamètres et NN ##################

LR = 1 * 1e-5
MAX_EPOCH = 50

in_size = 3
out_size = 1
layer_size = 3000
deepness = 3

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
############################## DATA ################################### 
#######################################################################

##Déclaration préliminaire
Yaw_corr = data_processing.yaw # Correction de l'AI
Burton = data_processing.Burton # Correction du Skew
AI_vanilla = data_processing.compute_AI #Calculateur de l'induction axiale (AI) native
App = data_processing.Apply_corr # Appliquer la correction (à besoins d'être différentiable)

correction_none = []
solver = bemol.ning.NingUncoupled(mexico, rho, correction_none)

type = torch.float32
n = 4
samp_size = 10**n
batch_size = 10**(n-1)

##Azimuths & Yaw
delta_phi = 2.0*PI*N/(samp_size) #Discrétisation du cercle parcouru par l'élément
Azimuths = torch.arange(0, 2*N*PI + delta_phi , delta_phi) #On récupère samp_size + 1 éléments
Yaw = PI/2 *torch.rand(Azimuths.shape, dtype = type)  #yaw € [0, pi/2]
WSA = torch.zeros_like(Yaw)

## Calculer l'induction axiale native
print("Calcul des inductions axiales. \n")
AI_va = torch.zeros_like(Yaw, dtype = type)
for i in range(Yaw.shape[0]) :
    AI_va[i], _ = AI_vanilla(solver, Azimuths[i], Yaw[i])
    WSA[i] = Burton(AI_va[i], Yaw[i])

print("Stockage des données avant entraînement.\n")
X = torch.zeros((Azimuths.shape[0], 3), dtype = type)
X[:,0] = Azimuths
X[:,1] = AI_va
X[:,2] = Yaw

data_used = ["Azimuths", "Induction Axiale", "Yaw"]
## Construire l'induction axiale corrigée
print("Calcul des données de références.\n")
Z = torch.zeros((Azimuths.shape[0],1), dtype = type)
for i in range(AI_va.shape[0]) : 
    coeff = data_processing.yaw(WSA = AI_va[i], AA = Azimuths[i])
    Z[i,0] = App(AI_va[i], coeff)

x_train, x_val, z_train, z_val = train_test_split(X, Z, test_size=0.2)

train_dataloader = DataLoader(TensorDataset(x_train.unsqueeze(1), z_train.unsqueeze(1)), batch_size = batch_size, pin_memory = True, shuffle = True)
val_dataloader = DataLoader(TensorDataset(x_val.unsqueeze(1),z_val.unsqueeze(1)), batch_size=batch_size, pin_memory=True, shuffle=True)

#######################################################################
#######################################################################
#######################################################################




#######################################################################
########################### ENTRAINEMENT ##############################
#######################################################################

#Gestion de l'affichage 
step  = 1 #étape à afficher dans le terminal
tronc = 5 #ordre de la troncature pour err_min_train et err_min_val
tol = 1e-6 #seuil de tolérance pour l'erreur faite sur les données d'entraînement ET de validation

##Instanciation d'un NN 
T800 = SkyNet(deepness, type, True).to(device)
optimiser = optim.Adam(T800.parameters(), lr = LR)
loss_func = nn.MSELoss(reduction = 'mean')

err_abs_train = []
err_rel_train = []

err_abs_val = []
err_rel_val = []

perf_av = []
ep_reach = MAX_EPOCH

print("Début de l'entraînement.\n")
start_loop = perf_counter()
for ep in range(MAX_EPOCH) :
    
    
    if ep % step == 0 :
        print("Etape "+str(ep + 1)+" sur " + str(MAX_EPOCH) + "\n")

    """
        Boucle d'entraînement du modèle.
    """

    loss_train_int = []
    rel_err_int = []
    perf_loc = []

    T800.train()
    for data, target in train_dataloader : 
        start_train = perf_counter()

        data = data.to(device)
        target = target.to(device)

        optimiser.zero_grad()

        coeff = T800(data)
        coeff = coeff.reshape(data[:,0,1].shape)
        AI_update = torch.zeros_like(coeff)
        AI_update = App(data[:,0,1], coeff)

        loss = loss_func(input = AI_update, target = target[:,0,0])
        loss.backward()
        optimiser.step()

        #Absolute error
        loss_train_int.append(loss.detach().cpu().numpy())

        #Relative error
        numerator = np.linalg.norm((AI_update - target[:,0,0]).detach().cpu().numpy())
        denominator = np.linalg.norm(target[:,0,0].detach().cpu().numpy())
        rel_error = numerator/denominator
        rel_err_int.append(rel_error)

        stop_train = perf_counter()
        perf_loc.append(stop_train-start_train)

    err_abs_train.append(np.average(loss_train_int))
    err_rel_train.append(np.average(rel_err_int))
    perf_av.append(np.average(perf_loc))

    if ep % step == 0 :
        print("Erreur absolue (entrainement) : " + str(err_abs_train[-1]))
        print("Erreur relative (entrainement) : " + str(err_rel_train[-1]))
        if ep >= 1 : 
            if err_abs_train[-1] - err_abs_train[-2] < 0 :
                print("Loss_train décroissante\n")
            else : 
                print("Loss_train croissante\n")             

    """
        Validation sur les échantillons de test.
    """
    loss_val_int = []
    rel_err_int = []

    T800.eval()
    for data, target in val_dataloader : 

        data = data.to(device)
        target = target.to(device)

        coeff = T800(data)
        coeff = coeff.reshape(data[:,0,1].shape)
        AI_update = torch.zeros_like(coeff)
        AI_update = App(data[:,0,1], coeff)

        loss = loss_func(input = AI_update, target = target[:,0,0])

        #Absolute error
        loss_val_int.append(loss.detach().cpu().numpy())

        #Relative error
        numerator = np.linalg.norm((AI_update - target[:,0,0]).detach().cpu().numpy())
        denominator = np.linalg.norm(target[:,0,0].detach().cpu().numpy())
        rel_error = numerator/denominator
        rel_err_int.append(rel_error)

    err_abs_val.append(np.average(loss_val_int))   
    err_rel_val.append(np.average(rel_err_int))

    if ep % step == 0 :
        print("Erreur absolue (validation) : "+ str(err_abs_val[-1]))
        print("Erreur relative (validation) : " + str(err_rel_val[-1]))
        if ep >= 1 :
            if err_abs_val[-1] - err_abs_val[-2] < 0 :
                print("Loss_val décroissante\n")
            else : 
                print("Loss_val croissante\n")   

    if err_abs_train[-1] <= tol and err_abs_val[-1] <= tol : 
        ep_reach = ep
        print("L'erreur d'entrainement et de validation sont passée sous le seuil de " + f'{tol:.2e}' + " en "+str(ep_reach) + "epochs.")
        break

MAX_EPOCH = ep_reach
#Mesure des performances temporelles de l'entraînement
stop_loop = perf_counter()
time_av = np.average(perf_av)
time_loop = stop_loop - start_loop

#Erreur minimal sur l'entrainement et la validation
err_min_train_abs = float(min(err_abs_train))
err_min_train_rel = float(min(err_rel_train))

err_min_val_abs = float(min(err_abs_val))
err_min_val_rel = float(min(err_rel_val))

#######################################################################
#######################################################################
#######################################################################




#######################################################################
########################### PLOT ET LOG ############################### 
#######################################################################
"""
##Compter les simulations pour s'y retrouver plus facilement
compteur_fic = "SkyNetV2_results/compteur.txt"
compteur = 0
with open(compteur_fic,"r", encoding='utf-8') as f :
    compteur = int(f.read())
compteur += 1
with open(compteur_fic, "w", encoding='utf-8') as f :
    f.write(str(compteur))




##Ecriture des performances et autres informations relatives à différents tests 

file = "SkyNetV2_results/log_SkyNetV2.txt"
fic = open(file, "a", encoding='utf-8')

fic.write("*******************************************************************************************************\n")
fic.write("Simulation numéro " + str(compteur) + "\n")
fic.write("Informations relatives à la simulation : \n")
fic.write("Taille de l'échantillon prévelé : " + str(samp_size)+"\n")
fic.write("Learning rate : " + str(LR) + "\n")
fic.write("Nombre d'epochs parcourues : "+str(MAX_EPOCH)+"\n")
fic.write("Taille des batchs : " + str(batch_size) + "\n")
fic.write("Format des float : " + str(type) + "\n")

fic.write("\n\n")

fic.write("Informations relatives au réseau de neurone : \n")
fic.write("Biais ? " + str(bool)+"\n")
fic.write("Taille de l'entrée : " + str(in_size) + "\n")
fic.write("Taille de la sortie : " + str(out_size) + "\n")
fic.write("Nombre de neurone par couche (le même pour toutes les couches pour l'instant) : "+str(layer_size)+"\n")
fic.write("Nombre de couche : " + str(deepness)+"\n")
fic.write("Fonction d'activation  :" + str(a) + "\n")
fic.write("Fonction de perte : " + str(loss_func.__class__.__name__) + "\n")
fic.write("Méthode d'optimisation : " + str(optimiser.__class__.__name__) + "\n")
fic.write("Initialisation pour le gradient : random\n")
fic.write("Données utilisées pour l'entraînement : \n")
for string in data_used : 
    fic.write("\t\t" + string + "\n")

fic.write("\n\n")

fic.write("Performances et erreurs du modèle : \n")
fic.write("GPU ou CPU ?  " + str(device) + "\n")
fic.write(f"Temps d'exécution de la boucle : {time_loop :.2e} secondes.\n")
fic.write(f"Temps d'entraînement moyen par epoch : {time_av: .2e} secondes.\n")
fic.write("Erreur minimale sur l'entraînement : \n")
fic.write(f"\t\t Erreur absolue : {err_min_train_abs:.2e}\n")
fic.write(f"\t\t Erreur relative :  {err_min_train_rel:.2e}\n")
fic.write("Erreur minimale sur la validation : \n")
fic.write(f"\t\t Erreur absolue : {err_min_val_abs:.2e}\n")
fic.write(f"\t\t Erreur relative : {err_min_val_rel:.2e}\n")
fic.write("*******************************************************************************************************\n\n\n\n\n")
##############################################################################################################
"""
#Perte sur entraînement & sur validation
fig1 = plt.figure(figsize=(20, 14))

plt.subplot(1,2,1)
plt.loglog(err_abs_train, color = 'r', label = 'Perte sur entraînement')
plt.loglog(err_abs_val, color = 'b', label = 'Perte sur validation')
plt.title('Erreur absolue')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.loglog(err_rel_train, color = 'r', label = 'Perte sur entraînement')
plt.loglog(err_rel_val, color = 'b', label = 'Perte sur validation')
plt.title('Erreur relative')
plt.legend()
plt.grid(True)

"""
    Test ultime : vérifcation de la qualité de la prédiction sur 
    l'exemple de yaw_models.py. On reprend donc les mêmes paramètres pour voir
    si, au moins dans ce cas, le correctif SkyNet fonctionne bien.
"""
azimuthAngle = 0.0
preconeAngle = 0.0
tiltAngle = 0.0
skewAngle = np.radians(30.)
yawAngle = skewAngle
DefRad = data_processing.default_radius

####Solver avec le correctif de l'IFPEN
corr_yaw_ifpen = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Dummy,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]
solver_yawed_ifpen = bemol.ning.NingUncoupled(mexico, rho, corr_yaw_ifpen)
forces_ifpen, AI_yawed_ifpen , azs = solver_yawed_ifpen.cycle(
        mexico.pitchRated, wind, omega, angles=[yawAngle, tiltAngle], tStep=tStep,
        n_phi=180, N=N, elements=elements,
)

corr_yaw_PP = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Dummy,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.PittAndPeters
]

solver_yawed_PP = bemol.ning.NingUncoupled(mexico, rho, corr_yaw_PP)
forces_PP, AI_PP, _ = solver_yawed_PP.cycle(
        mexico.pitchRated, wind, omega, angles=[yawAngle, tiltAngle], tStep=tStep,
        n_phi=180, N=N, elements=elements,
)

####Test de SkyNet
T800.eval()

forces_SKN = np.zeros((180,1)) ##Normal Forces corrected with SkyNet
solver_validation = bemol.ning.NingUncoupled(mexico, rho, corr_yaw_ifpen)

Corr_skynet = []
Corr_ifpen = []
Corr_PP = []

for i, az in enumerate(azs):
    AI, treat = data_processing.compute_AI(solver_validation, az, yawAngle)
    #wakeSkewAngle = Burton(AI, yawAngle)
    
    X = torch.tensor([az, AI, yawAngle], dtype=type, device=device) 
    corr_T800 = T800(X).to("cpu").item() ; corr_ifpen = Yaw_corr(AI, az) ; corr_PP = data_processing.PP(AI, az)
    Corr_skynet.append(corr_T800) ; Corr_ifpen.append(corr_ifpen) ; Corr_PP.append(corr_PP)    
    solver_validation._axial_induction = App(corr_T800, AI)

    fn, _ = data_processing.AIProcess(solver_validation, **treat)
    forces_SKN[i,0] = fn  

azs_deg = np.degrees(azs)

fig2 =plt.figure(figsize=(20, 14))
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
plt.subplot(1,2,1)
plt.plot(azs_deg, forces_ifpen[:,0,0], color  = 'green', label = 'IFPEN')
plt.plot(azs_deg, forces_PP[:,0,0], color = 'orange', label = 'Pitt&Pitter')
plt.plot(azs_deg, forces_SKN[:,0], color = 'purple', label = 'SkyNet')
plt.xlabel('Degrees')
plt.ylabel('Normal Forces')
plt.title('Comparaison sur les forces normales')
plt.legend()
plt.grid(True)


plt.subplot(1,2,2)
plt.plot(Corr_ifpen, color = 'green', label = 'Correction IFPEN')
plt.plot(Corr_PP, color = 'orange', label = 'Correction Pitt&Peter')
plt.plot(Corr_skynet, color = 'purple', label = 'Correction SkyNet')
plt.title('Comparaison numérique des correctifs')
plt.legend()
plt.grid(True)

"""
## Enregistrer le graphique
saved_fig = "SkyNetV2_results/graph_SkyNetV2_run_" + str(compteur)
plt.savefig(saved_fig)
"""
plt.show()
