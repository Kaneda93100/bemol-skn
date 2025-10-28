"""
    Dans ce code, j'expérimente une deuxième approche : au lieu d'imiter la fonction qui apporte le correctif, 
    j'imite directement la sortie de ning à laquelle on applique le correctif.
"""

import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from bemol import data_processing
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter

PI = float(np.pi)
tan = np.tan
sin = np.sin 
zeros = np.zeros

import os

import pandas as pd

## uncoment following lines if ModuleNotFounError
## add repo folder to PYTHONPATH, if running from the samples folder
import sys
sys.path.append(os.path.abspath(f'{__file__}/../..'))

import bemol
from bemol import data_processing

results_folder = 'results/yaw_models'
os.makedirs(results_folder,exist_ok=True)


#######################################################################
################ Paramètres intrasèques au modèle, ####################
################ donc considérés comme constant    ####################
#######################################################################

##On retrouvera le détail des données de l'éolienne dans mexico/rotors/rotor.yml
mexico = bemol.rotor.mexico
HR = mexico.hubRadius
TR = mexico.tipRadius
wind = mexico.windRated     #15.06 
omega = mexico.omegaRated   #44.5163679
pitch = mexico.pitchRated   #-0.040143

rho = 1.191                 #Air Density
N = 1.0                     #Nombre de révolution (float)
tStep = 0.1                 #Probablement un paramètre temporel (je croyais que la BEM était un modèle stationnaire)
elements = [24] # node close to the experimental data (r/R = 0.82)

#Angles
tiltAngle = 0.0
preconeAngle = 0.0

#######################################################################
#######################################################################
#######################################################################




#######################################################################
###################### RESEAU DE NEURONES SKYNET ###################### 
#######################################################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)

################## Hyperparamètres ##################

LR = 1 * 1e-4
MAX_EPOCH = 50
BATCH_SIZE = 200
deepness = 5

####################  NN  ###########################

bias = True
in_size = 3
out_size = 1
layer_size = 3000

act1 = nn.ReLU#Activation pour la 1ère couche
#act2 = nn.ReLU #Activation pour la 2ème couche
#act3 = nn.ReLU #Activation pour la 3ème couche
#act4 = nn.ReLU #Activation pour la 4ème couche 
#act5 = nn.ReLU #Activation pour la 5ème couche

a1 = act1.__name__
#a2 = act2.__name__
#a3 = act3.__name__
#a4 = act4.__name__
#a5 = act5.__name__

class SkyNet(nn.Module) : 
    def __init__(self, deepness):
        super(SkyNet, self).__init__()
        self.deepness = deepness

        self.HL = nn.ModuleList()
        self.HL.append(nn.Linear(in_size, layer_size, bool))

        for i in range(self.deepness) : 
            self.HL.append(nn.Linear(layer_size, layer_size, bool))
            self.HL.append(act1())
            self.HL.append(nn.Linear(layer_size, layer_size, bool))

        self.HL.append(nn.Linear(layer_size, out_size, bool))

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

"""
    De même que pour SkyNet V1, il faut prétraiter les données pour entraîner 
    correctement le réseau. Ainsi, on applique la fonction data.processing à 
    la liste contenant les skew pour le corrigé. 
"""

##Déclaration préliminaire
AIProcess = data_processing.AxialInducProcessing
FProcess = data_processing.AxialTreatment
Yaw_corr = data_processing.yaw
Burton = data_processing.Burton

##Correctifs sans yaw
corrections_unyawed = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
]

##Correctifs avec yaw
corrections_yawed = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN,
]

solver_yawed = bemol.ning.NingUncoupled(mexico, rho, corrections_yawed)
solver_unyawed = bemol.ning.NingUncoupled(mexico, rho, corrections_unyawed)

##Samples for every parameters
samp_size = 2*10**2

delta_phi = 2.0*PI*N/(samp_size)
Azimuths = np.arange(0, 2*N*PI + delta_phi, delta_phi)

Yaw = PI/2 *(np.random.rand(samp_size + 1, 1))
WSA = np.zeros((samp_size + 1, 1))

for i in range(len(Yaw)-1) : 
    temp, _ = AIProcess(solver_unyawed, Azimuths[i], Yaw[i,0]) 
    WSA[i,0] = temp

X = np.zeros((samp_size + 1, 3))
X[:,0] = WSA[:,0]
X[:,1] = Azimuths
X[:,2] = Yaw[:,0]

Y = np.zeros((samp_size + 1, 1))

for i, az in enumerate(Azimuths) : 
    Forces,_ = solver_yawed.steady(az, pitch, wind, omega, angles=[Yaw[i,0], tiltAngle], precone  = preconeAngle, elements = elements)
    Y[i,0] = Forces[0,0]

x_train, x_val, y_train, y_val = map(torch.tensor, train_test_split(X,Y, test_size=0.2))

x_train = x_train.type(torch.float32)
y_train = y_train.type(torch.float32)

x_val = x_val.type(torch.float32).to(device)
y_val = y_val.type(torch.float32).to(device)

train_dataloader = DataLoader(TensorDataset(x_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE, pin_memory=False, shuffle=True)
val_dataloader = DataLoader(TensorDataset(x_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE, pin_memory=False, shuffle=True)

"""
   Précalcul pour chaque donnée de X_train et X_val de la sortie par le solver .cycle sans le correctif yaw de l'IFPEN
   Stockage des données résolues d'entraînement --> Y_train_unyaw
   Stockage des données résolues de validation --> Y_val_unyaw
"""

#######################################################################
#######################################################################
#######################################################################




#######################################################################
########################### ENTRAINEMENT ##############################
#######################################################################

##Instanciation d'un NN 
T800 = SkyNet(deepness).to(device)
optimiser = optim.Adam(T800.parameters(), lr = LR)
loss_func = nn.MSELoss(reduction = 'mean')

#Gestion de l'affichage 
step  = 3 #étape à afficher dans le terminal
tronc = 7 #ordre de la troncature pour err_min_train et err_min_val

loss_train= list()
loss_val = list()

perf_av = []
print("#######################################################################")
print("Début de l'entraînement.\n")

start = perf_counter()
for ep in range(MAX_EPOCH) :
    T800.train()
    
    if ep % step == 0 :
        print("Etape "+str(ep + 1)+" sur " + str(MAX_EPOCH) + "\n")
    
    loss_train_int = list()

    perf_loc = []
    
    for Xdata_train, Ydata_train in train_dataloader :
        start_train = perf_counter()

        #Transfert sur GPU/CPU
        Xdata_train = Xdata_train.to(device)
        Ydata_train = Ydata_train.to(device)

        optimiser.zero_grad()

        #Calcul du correctif
        output = T800(Xdata_train) 
        corr_yaw = output[:,0,0]

        #Calcul de l'induction axiale avant l'application de toutes corrections
        AI, treat = AIProcess(solver_unyawed, Xdata_train[:,0,1], Xdata_train[:,0,2])
        AI *= corr_yaw
        solver_unyawed._axial_induction = AI

        output = FProcess(solver_unyawed, treat) #Forces normales corrigées

        #Calcul de la loss
        loss = loss_func(input=output, target=Ydata_train)
        loss.backward()

        optimiser.step()
        loss_train_int.append(loss.detach().cpu().numpy())

        #Mesure de performance
        stop_train = perf_counter()
        perf_loc.append(stop_train-start_train)

    loss_train.append(np.average(loss_train_int))
    perf_av.append(np.average(perf_loc))

    if ep % step == 0 :
        print("Erreur (entrainement) : " + str(loss_train[-1]))

    #Avoir un apperçu des variations de loss_train
    if ep > 1 and ep % step == 0 : 
        if loss_train[-1] - loss_train[-2] < 0 :
            print("Loss_train décroissante\n")
        else : 
            print("Loss_train croissante\n")

    T800.eval()

    loss_val_int = list()
    for xdata_val, zdata_val in val_dataloader :

        #Transfert sur GPU/CPU
        Xdata_val = Xdata_val.to(device)
        zdata_val = zdata_val.to(device)


        
        loss = loss_func(input = output, target = Ydata_val_yaw)
        loss_val_int.append(loss.detach().cpu().numpy())

    loss_val.append(np.average(loss_val_int))

    if ep % step == 0 :
        print("Erreur (validation) : "+ str(loss_val[-1]))

    #Avoir un apperçu des variations de la loss_val
    if ep > 1 and ep % step == 0:
        if loss_val[-1] - loss_val[-2] < 0 :
            print("Loss_val décroissante\n")
        else : 
            print("Loss_val croissante\n")
stop =  perf_counter()
print("\nFin de l'entraînement.")   
print("#######################################################################\n\n")

#Mesure des performances temporelles de l'entraînement
time_exe = stop - start
time_av = np.average(perf_av)

#Erreur minimal sur l'entrainement et la validation
err_min_train = float(min(loss_train))
err_min_val = float(min(loss_val))

#######################################################################
########################### PLOT ET LOG ############################### 
#######################################################################

##Compter les simulations pour s'y retrouver plus facilement
compteur_fic = "SkyNetV2_results/compteur.txt"
compteur = 0
with open(compteur_fic,"r", encoding='utf-8') as f :
    compteur = int(f.read())
compteur += 1
with open(compteur_fic, "w", encoding='utf-8') as f :
    f.write(str(compteur))

fig = plt.figure(figsize=(20,14))

#Loss sur l'entrainement
plt.subplot(2,2,1)
plt.plot(loss_train, color = 'r')
plt.title('Perte sur entrainement')
plt.grid(True)

#Loss sur la validation
plt.subplot(2,2,2)
plt.plot(loss_val, color = 'b')
plt.title('Perte sur validation')
plt.grid(True)

################## Test ultime : comparer la courbe du correctif IFPEN et celle de Skynet ##################

azimuthAngle = 0.0
preconeAngle = 0.0
tiltAngle = 0.0
skewAngle = np.radians(30.)
yawAngle = skewAngle
DefRad = data_processing.default_radius

## Courbe du correctif IFPEN
forces_IFPEN, axialInduction_yawed, azs = solver_yawed.cycle(
        mexico.pitchRated,wind,omega,angles=[yawAngle,tiltAngle],tStep=tStep,
        n_phi=180,N=N,elements=elements,
)


##Courbe de SkyNet
T800.eval()

forces_SKN = np.zeros((180,1)) ##Normal Forces corrected with SkyNet
solver_validation = bemol.ning.NingUncoupled(mexico, rho, corrections_unyawed)

for i, az in enumerate(azs):
    AI, treat = data_processing.AxialInducProcessing(solver_validation, az, yawAngle)
    wakeSkewAngle = Burton(AI, yawAngle)
    
    X = torch.tensor([wakeSkewAngle, az], dtype=torch.float32, device=device)
    Corr_T800 = T800(X).to("cpu").item()
    solver_validation._axial_induction *= Corr_T800

    fn, _, _, _ = data_processing.AxialTreatment(solver_validation, **treat)
    forces_SKN[i,0] = fn  

plt.subplot(2,2,4)
plt.plot(axialInduction_unyawed[:,0,0], axialInduction_unyawed[:,0,1], label='SkyNet')
plt.title('SkyNet')
plt.xlabel('[:,0,0]')
plt.ylabel('[:,0,1]')
plt.grid(True)


# Affichage d'informations complémentaires
plt.figtext(0.02, 0.04, 
            "Fonction approchée : Correctif de l'IFPEN",
            wrap=True, horizontalalignment='left', fontsize=10, 
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 5})

plt.figtext(0.43, 0.04, 
            "Learning rate = {} | Epochs = {} | Batch size = {} | Echantillon = {}".format(LR, MAX_EPOCH, BATCH_SIZE, samp_size),
            wrap=True, horizontalalignment='center', fontsize=10, 
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 5})

plt.figtext(0.95, 0.04, 
            "Erreur minimale sur train = {:.5f} | Erreur minimale sur val = {:.5f}".format(err_min_train, err_min_val),
            wrap=True, horizontalalignment='right', fontsize=10, 
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 5})

plt.tight_layout(rect=[0, 0.09, 1, 0.91]) 
save_fig = "SkyNetV2_results/graph_SkyNetV2_run_" + str(compteur)
plt.savefig(save_fig, dpi = 600, bbox_inches='tight')
plt.show()

##Ecriture des performances et autres informations relatives à différents tests 
file = "SkyNetV2_results/log_SkyNetV2.txt"
fic = open(file, "a", encoding='utf-8')

fic.write("*******************************************************************************************************\n")
fic.write("Simulation numéro " + str(compteur) + "\n")
fic.write("Informations relatives à la simulation : \n")
fic.write("Fonction régressée : correction de l'IFPEN \n")
fic.write("Taille de l'échantillon prévelé : " + str(samp_size)+"\n")
fic.write("Learning rate : " + str(LR) + "\n")
fic.write("Nombre d'epochs parcouru : "+str(MAX_EPOCH)+"\n")
fic.write("Taille des batchs : " + str(BATCH_SIZE) + "\n")
fic.write("Temps de calcul de Y_train : " + str(comp_Ytrain) + " secondes\n")
fic.write("Temps de calcul de Y_val : " + str(comp_Yval) + " secondes\n")

fic.write("\n\n")

fic.write("Informations relatives au réseau de neurone : \n")
fic.write("Biais ? " + str(bool)+"\n")
fic.write("Taille de l'entrée : " + str(in_size) + "\n")
fic.write("Taille de la sortie : " + str(out_size) + "\n")
fic.write("Nombre de neurone par couche (le même pour toutes les couches pour l'instant) : "+str(layer_size)+"\n")
fic.write("Nombre de couche : " + str(3)+"\n")
fic.write("Fonctions d'activations (dans l'ordre du forward pass): \n")
fic.write("Couche 1 : " + str(a1) + "\n")
fic.write("Couche 2 : " + str(a2) + "\n")
fic.write("Couche 3 : " + str(a3) + "\n")
fic.write("Couche 4 : " + str(a4) + "\n")
fic.write("Couche 5 : " + str(a5) + "\n")
fic.write("Fonction de perte : " + str(loss_func.__class__.__name__) + "\n")
fic.write("Méthode d'optimisation : " + str(optimiser.__class__.__name__) + "\n")
fic.write("Initialisation pour le gradient : vecteur nul\n")

fic.write("\n\n")

fic.write("Performances du modèle : \n")
fic.write("GPU ou CPU ?  " + str(device) + "\n")
fic.write("Temps d'exécution de la boucle : " + str(time_exe)+"\n")
fic.write("Temps d'entraînement moyen par epoch : " + str(time_av)+"\n")
fic.write("Plus petite perte sur l'entraînement : " + str(err_min_train)+"\n")
fic.write("Plus petite perte sur la validation : " + str(err_min_val)+"\n")
fic.write("*******************************************************************************************************\n\n\n\n\n")
##############################################################################################################
