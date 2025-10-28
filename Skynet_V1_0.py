"""
    Dans ce code, on entraîne SkyNet à imiter le correctif multiplicatif du modèle BEM proposé par l'IFPEN
    (Blondel et al. 2017, Improving a BEM yaw model based on NewMexico experimental data and Vortex/CFD simulations.
    CFM 2017 - 23ème Congrès Français de Mécanique ifp.hal.science/hal-01663643). 
    
    Ce correctif est vu comme une fonction de 3 paramètres : le Skew, l'azimut et le radius
    Les autres paramètres sont considérés comme intrasèque au modèle (donc fixé).  

	Microsoft vous êtes des gros fils de pute, je vous déteste, je vais passer sous linux 
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
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter

PI = np.pi
tan = np.tan
sin = np.sin 

import os
def is_directory_empty(path) : 
    with os.scandir(path) as it :
        return not any(it)
    
import pandas as pd

## uncoment following lines if ModuleNotFounError
## add repo folder to PYTHONPATH, if running from the samples folder
import sys
sys.path.append(os.path.abspath(f'{__file__}/../..'))

import bemol
from bemol import data_processing

results_folder = 'results/yaw_models'
os.makedirs(results_folder,exist_ok=True)

##################################################################################################
################ Paramètres intrasèques au modèle, donc considérés comme constant ################
##################################################################################################

wind = 15.06
omega = 44.5163679
rho = 1.191
N = 1.0 #number of revolutions
tStep = 0.1
elements = [24] # node close to the experimental data (r/R = 0.82)

##Attributs de la classe YawModel (bemol-v0.0.1\bemol\secondary.py)
A0 = 0.35
Phi1 = - PI/9
Phi2 = PI

mexico = bemol.rotor.mexico
HR = mexico.hubRadius
TR = mexico.tipRadius

#######################################################################
#######################################################################
#######################################################################




#######################################################################
###################### RESEAU DE NEURONES SKYNET ###################### 
#######################################################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)

################## Hyperparamètres et NN ##################

LR = 1e-4
MAX_EPOCH = 10
n = 2
BATCH_SIZE = 10**(n-1)

bias = True
in_size = 4
out_size = 1
layer_size = 3000
deepness = 5

act1 = nn.ReLU 

a1 = act1.__name__

class SkyNet(nn.Module) : 
    def __init__(self, deepness):
        super(SkyNet, self).__init__()
        self.deepness = deepness

        self.HL = nn.ModuleList()
        self.HL.append(nn.Linear(in_size, layer_size, bias = bias))

        for i in range(self.deepness) : 
            self.HL.append(nn.Linear(layer_size, layer_size, bias = bias))
            self.HL.append(act1(inplace = True))
            self.HL.append(nn.Linear(layer_size, layer_size, bias = bias))

        self.HL.append(nn.Linear(layer_size, out_size, bias = bias))

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
    Avant l'application du correctif sur le Yaw, la méthode solve de NingUncoupled applique une correction sur le SkewAngle
    Donc Avant de donner le skewAngle à SkyNet, on applique la correction de Burton. On donne le Yaw qu'on prend 
    sur le même intervalle qu'avant, on applique Burton à la liste Skew. 
"""

##Déclaration préliminaire
AIProcess = data_processing.compute_AI
Yaw_corr = data_processing.yaw
Burton = data_processing.Burton

corrections_unyawed = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.Dummy
]

corrections_yawed = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]

solver_unyawed = bemol.ning.NingUncoupled(mexico, rho, corrections_unyawed)
solver_yawed = bemol.ning.NingUncoupled(mexico, rho, corrections_yawed)


##Samples for every parameters
samp_size = 10**(n)

"""
    samp_size remplace la variable n_phi. En fait, pour chaque angle possible,
    SkyNet corrige pour un angle d'azimut donné un yaw par rapport à un skew
    corrigé.
"""

##Azimuths
delta_phi = 2.0*PI*N/(samp_size) #Discrétisation du cercle parcouru par l'élément
Azimuths = np.arange(0, 2*N*PI + delta_phi, delta_phi)

"""
    Construction des données :
        On prend un angle de yaw aléatoire sur [0,pi/2] et on regarde
        l'induction axiale d'un angle de yaw sur un angle azimuthale.
"""

##Yaw & Skew
Yaw = PI/3 *(2*np.random.rand(samp_size + 1) - 1) # Yaw € [-pi/3, pi/3]
AI_va = np.zeros((samp_size + 1))
Wind = 10*np.random.rand(samp_size + 1) + 10 # Wind € [10,20]

for i in range(len(Yaw)) : 
    AI_va[i], _ = AIProcess(solver_unyawed, Azimuths[i], Yaw[i], wind = Wind[i]) 
    #temp = Burton(temp, Yaw[i,0])
    #WSA[i] = temp

X = np.zeros((samp_size + 1, in_size))
X[:,0] = AI_va
X[:,1] = Azimuths
X[:,2] = Yaw
X[:,3] = Wind 

Y = Yaw_corr(X[:,0], X[:,1])

x_train, x_val, y_train, y_val = map(torch.tensor, train_test_split(X, Y, test_size=0.2))

plot = x_val[:, 1].size(0)

##Forcer le typage des tenseurs en float32 (visiblement en float64 par défaut)
x_train = x_train.type(torch.float32)
y_train = y_train.type(torch.float32)

x_val = x_val.type(torch.float32).to(device)
y_val = y_val.type(torch.float32).to(device)

train_dataloader = DataLoader(TensorDataset(x_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE, pin_memory=False, shuffle=True)
val_dataloader = DataLoader(TensorDataset(x_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE, pin_memory=False, shuffle=True)

#######################################################################
#######################################################################
#######################################################################



#######################################################################
########################### ENTRAINEMENT ##############################
#######################################################################

#Gestion de l'affichage 
step  = 1 #étape à afficher dans le terminal
tronc = 5 #ordre de la troncature pour err_min_train et err_min_val
tol = 1e-7 #seuil de tolérance pour l'erreur faite sur les données d'entraînement ET de validation

#Erreur absolue
loss_train= []
loss_val = []

#Erreur relative
rel_error_train = []
rel_error_val = []

#Norme du gradient
grad_norm = []

##Instanciation d'un NN
T800 = SkyNet(deepness).to(device) ##Indiquer si on entraîne sur CPU ou GPU

# Initialisation des paramètres du réseau
dir_param = "SkyNetv1_results/Parameters/Bias_" + str(bias) + "/in_size_" + str(in_size) + "/samp_size_" + str(samp_size) + "/layer_size_" + str(layer_size) + "/deepness_" + str(deepness)
file_param = dir_param + "/best_param.pt2"
prec_test = {'train' : 1, 'val' : 1}
param_saved = [T800.state_dict(), prec_test]
if os.path.exists(dir_param) : 
    T800.load_state_dict(torch.load(file_param))
    param_saved[0] = T800.state_dict()
    init_SKN = "Paramètre initiale issus d'une simulation précédente."
    print("Des paramètres d'une simulation précédente ont été trouvé.\n")
else : 
    os.makedirs(dir_param)
    init_SKN = "Initialisation des paramètres aléatoire."
    print("Initialisation aléatoire.\n")

#Optimiser & Loss func
optimiser = optim.Adam(T800.parameters(), lr = LR, weight_decay=1)
loss_func = nn.MSELoss(reduction = 'mean')

ep_reach = MAX_EPOCH
perf_av = []

print("Début de l'entraînement.\n")
start = perf_counter()
for ep in range(MAX_EPOCH) :
    T800.train()
    
    if ep % step == 0 :
        print("Etape "+str(ep + 1)+" sur " + str(MAX_EPOCH) + "\n")
    
    loss_train_int = list()
    rel_err_int = []
    
    perf_loc = []
    
    for Xdata_train, Ydata_train in train_dataloader :
        start_train = perf_counter()

        Xdata_train = Xdata_train.to(device)
        Ydata_train = Ydata_train.to(device)

        optimiser.zero_grad()

        score = T800(Xdata_train)
        score = score.reshape(Ydata_train.shape)
        loss = loss_func(input=score, target=Ydata_train)
        loss.backward()

        optimiser.step()
        loss_train_int.append(loss.detach().cpu().numpy())

        #Relative error
        numerator = np.linalg.norm((score - Ydata_train).detach().cpu().numpy())
        denominator = np.linalg.norm(Ydata_train.detach().cpu().numpy())
        rel_error = numerator/denominator
        rel_err_int.append(rel_error)

        stop_train = perf_counter()
        perf_loc.append(stop_train-start_train)
    
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
    perf_av.append(np.average(perf_loc))

    if ep % step == 0 :
        print("Erreur absolue (entrainement) : " + str(loss_train[-1]))
        print("Erreur relative (entrainement) : " + str(rel_error_train[-1]))

    #Avoir un apperçu des variations de loss_train
    if ep > 1 and ep % step == 0 : 
        if loss_train[-1] - loss_train[-2] < 0 :
            print("Loss_train décroissante\n")
        else : 
            print("Loss_train croissante\n")

    T800.eval()

    loss_val_int = list()
    rel_err_int = []
    for Xdata_val, Ydata_val in val_dataloader :
        Xdata_val = Xdata_val.to(device)
        Ydata_val = Ydata_val.to(device)

        score = T800(Xdata_val)
        score = score.reshape(Ydata_val.shape)
        loss = loss_func(input = score, target = Ydata_val)

        loss_val_int.append(loss.detach().cpu().numpy())

        #Relative error
        numerator = np.linalg.norm((score - Ydata_val).detach().cpu().numpy())
        denominator = np.linalg.norm(Ydata_val.detach().cpu().numpy())
        rel_error = numerator/denominator
        rel_err_int.append(rel_error)

    loss_val.append(np.average(loss_val_int))
    rel_error_val.append(np.average(rel_err_int))


    if ep % step == 0 :
        print("Erreur absolue (validation) : "+ str(loss_val[-1]))
        print("Erreur relative (validation) : " + str(rel_error_val[-1]))

    #Avoir un apperçu des variations de la loss_val
    if ep > 0 and ep % step == 0:
        if loss_val[-1] - loss_val[-2] < 0 :
            print("Loss_val décroissante\n")
        else : 
            print("Loss_val croissante\n")
    
    if loss_val[-1] <= tol and loss_train[-1] <= tol : 
        ep_reach = ep
        print("L'erreur d'entrainement et de validation sont passée sous le seuil de 9e-7 en "+str(ep_reach))
        break
    
    ##Récupération des meilleurs paramètres obtenus
    if ep == 0 : #Pour initialiser les paramètres
        param_saved[1]['train'] = loss_train[-1]
        param_saved[1]['val'] = loss_val[-1]
    else : 
        if param_saved[1]['train'] >= loss_train[-1] and param_saved[1]['val'] >= loss_val[-1] : #Tester les performances du nouveau set
            param_saved[0] = T800.state_dict()
            param_saved[1]['train'] = loss_train[-1]
            param_saved[1]['val'] = loss_val[-1]   
            print("\n\n**************************************")
            print("Nouveau set de paramètres enregistré.")
            print("**************************************\n\n")
        else : 
            continue

##Export des meilleurs paramètres trouvés
torch.save(param_saved[0], file_param)

T800.state_dict(param_saved[0]) #Chargement des meilleurs paramètres trouvés pendant l'entraînement

stop =  perf_counter()
MAX_EPOCH = ep_reach
#Mesure des performances temporelles de l'entraînement
time_exe = stop - start
time_av = np.average(perf_av)


#Erreur minimal sur l'entrainement, la validation et le gradient
err_min_train = float(min(loss_train))
err_min_val = float(min(loss_val))
err_min_grad = float(min(grad_norm))


#######################################################################
#######################################################################
#######################################################################




#######################################################################
########################### PLOT ET LOG ############################### 
#######################################################################
 
##Compter les simulations pour s'y retrouver plus facilement
compteur_fic = "SkyNetV1_results/compteur.txt"
compteur = 0
with open(compteur_fic,"r", encoding='utf-8') as f :
    compteur = int(f.read())
compteur += 1
with open(compteur_fic, "w", encoding='utf-8') as f :
    f.write(str(compteur))

fig = plt.figure(figsize=(20, 14))

#Perte sur entraînement & sur validation
plt.subplot(2,2,1)
plt.loglog(loss_train, color = 'r', label = 'Entraînement')
plt.loglog(loss_val, color = 'b', label = 'Validation')
plt.title('Evolution de la perte')
plt.legend()
plt.grid(True)

#Gradient sur entraînement et validation
plt.subplot(2,2,2)
plt.loglog(grad_norm, color = 'r')
plt.title('Gradient du réseau au cours des epochs')
plt.grid(True)

###############################################################################
###############################################################################

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
corr = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]
solver_yawed = bemol.ning.NingUncoupled(mexico, rho, corr)
forces, AI_yawed , azs = solver_yawed.cycle(
        mexico.pitchRated, wind, omega, angles=[yawAngle, tiltAngle], tStep=tStep,
        n_phi=180, N=N, elements=elements,
)

####Test de la correction que j'ai réécrite
forces_CopyCat = np.zeros((180,1))
solver_validation = bemol.ning.NingUncoupled(mexico, rho, corrections_unyawed)

for i, az in enumerate(azs) : 
    AI, treat = data_processing.AI_vanilla(solver_validation, az, yawAngle)
    
    wakeSkewAngle = Burton(AI, yawAngle)
    corr = Yaw_corr(wakeSkewAngle, az)

    solver_validation._axial_induction *= corr
    fn, _, _, _ = data_processing.AxialTreatment(solver_validation, **treat)
    forces_CopyCat[i,0] = fn

####Test de SkyNet
T800.eval()

corr = []
forces_SKN = np.zeros((180,1)) ##Normal Forces corrected with SkyNet
solver_validation = bemol.ning.NingUncoupled(mexico, rho, corr)

Corr_skynet = []
Corr_alg = []
for i, az in enumerate(azs):
    AI, treat = data_processing.AI_vanilla(solver_validation, az, yawAngle)
    wakeSkewAngle = Burton(AI, yawAngle)
    
    X = torch.tensor([wakeSkewAngle, az, yawAngle, wind], dtype=torch.float32, device=device)
    Corr_T800 = T800(X).to("cpu").item(); corr_alg = Yaw_corr(wakeSkewAngle, az)
    Corr_skynet.append(Corr_T800) ; Corr_alg.append(corr_alg)
    solver_validation._axial_induction *= Corr_T800

    fn, _, _, _ = data_processing.AxialTreatment(solver_validation, **treat)
    forces_SKN[i,0] = fn  

plt.subplot(2,2,3)
plt.plot(Corr_alg, color = 'green', label = 'Correction algébrique')
plt.plot(Corr_skynet, color = 'purple', label = 'Correction neuronale ')
plt.title('Comparaison numérique des correctifs')
plt.legend()
plt.grid(True)

##Test d'écart entre les différentes forces normales obtenues
err_IF_CC = np.zeros((len(azs),1))
err_IF_SKN = np.zeros((len(azs),1))
err_CC_SKN = np.zeros((len(azs),1))

validation = open("validation.txt", "w", encoding='utf-8')

for i in range(len(azs)) : 
    err_IF_CC[i,0] = abs(forces[i,0,0] - forces_CopyCat[i,0])
    err_IF_SKN[i,0] = abs(forces[i,0,0] - forces_SKN[i,0])
    err_CC_SKN[i,0] = abs(forces_CopyCat[i,0] - forces_SKN[i,0])

    if err_IF_CC[i,0] >= 1 :
        validation.write("Différence à l'indice " + str(i) + "\n")

err_norm_IF_CC = np.linalg.norm(forces[:,0,0] - forces_CopyCat[:,0])
err_norm_IF_SKN = np.linalg.norm(forces[:,0,0] - forces_SKN[:,0])
err_norm_CC_SKN = np.linalg.norm(forces_SKN[:,0] - forces_CopyCat[:,0])

print("Erreur en norme euclidienne (CopyCat et IFPEN) : ",err_norm_IF_CC)
print("Erreur en norme euclidienne (SkyNet et IFPEN) : ",err_norm_IF_SKN)
print("Erreur en norme euclidienne (CopyCat et SkyNet) : ",err_norm_CC_SKN)
azs_deg = np.degrees(azs)

plt.subplot(2,2,4)
plt.title('Axial induction - Yawed solver ')
plt.plot(azs_deg, forces[:,0,0], color='green', label = 'IFPEN')
plt.plot(azs_deg, forces_CopyCat[:,0], color='orange', label = 'Rewritten correctionn')
plt.plot(azs_deg, forces_SKN[:,0], color='purple', label='SkyNet')
plt.xlabel('Degrees')
plt.ylabel('Normal Forces')
plt.legend()
plt.grid(True)

"""
plt.subplot(2,3,4)
plt.title('|(IFPEN correction) - (Rewritten correction)| (error in absolute value)')
plt.plot(azs_deg, err_IF_CC)
plt.xlabel('Degrees')
plt.ylabel('Error')
plt.grid(True)

plt.subplot(2,3,5)
plt.title('|(IFPEN correction) - (SkyNet correction)|(error in absolute value)')
plt.plot(azs_deg, err_IF_SKN)
plt.xlabel('Degrees')
plt.ylabel('Error')
plt.grid(True)

plt.subplot(2,3,6)
plt.title('|(Rewritten correction) - (SkyNet correction)|(error in absolute value)')
plt.plot(azs_deg, err_CC_SKN)
plt.xlabel('Degrees')
plt.ylabel('Error')
plt.grid(True)
"""

###############################################################################
############################################################################### 
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
            "Erreur minimale sur train = {:.2e} | Erreur minimale sur val = {:.2e}".format(err_min_train, err_min_val),
            wrap=True, horizontalalignment='right', fontsize=10, 
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 5})

plt.tight_layout(rect=[0, 0.09, 1, 0.91])

saved_fig = "SkyNetV1_results/graph_SkyNetV1_run_" + str(compteur)
plt.savefig(saved_fig)
plt.show()


##Ecriture des performances et autres informations relatives à différents tests 
file = "SkyNetV1_results/log_SkyNetV1.txt"
fic = open(file, "a", encoding='utf-8')

fic.write("*******************************************************************************************************\n")
fic.write("Simulation numéro " + str(compteur) + "\n")
fic.write("Informations relatives à la simulation : \n")
fic.write("Fonction régressée : correction de l'IFPEN \n")
fic.write("Taille de l'échantillon prévelé : " + str(samp_size)+"\n")
fic.write("Learning rate : " + str(LR) + "\n")
fic.write("Nombre d'epochs parcouru : "+str(MAX_EPOCH)+"\n")
fic.write("Taille des batchs : " + str(BATCH_SIZE) + "\n")

fic.write("\n\n")

fic.write("Informations relatives au réseau de neurone : \n")
fic.write("Biais ? " + str(bias)+"\n")
fic.write("Taille de l'entrée : " + str(in_size) + "\n")
fic.write("Taille de la sortie : " + str(out_size) + "\n")
fic.write("Nombre de neurone par couche (le même pour toutes les couches pour l'instant) : "+str(layer_size)+"\n")
fic.write("Nombre de couche : " + str(3)+"\n")
fic.write("Fonctions d'activation d'activation :" + str(a1) + "\n")
#fic.write("Couche 1 : " + str(a1) + "\n")
#fic.write("Couche 2 : " + str(a2) + "\n")
#fic.write("Couche 3 : " + str(a3) + "\n")
#fic.write("Couche 4 : " + str(a4) + "\n")
#fic.write("Couche 5 : " + str(a5) + "\n")
fic.write("Fonction de perte : " + str(loss_func.__class__.__name__) + "\n")
fic.write("Méthode d'optimisation : " + str(optimiser.__class__.__name__) + "\n")
fic.write(init_SKN + "\n")

fic.write("\n\n")

fic.write("Performances et résultat du modèle : \n")
fic.write("GPU ou CPU ?  " + str(device) + "\n")
fic.write("Temps d'exécution de la boucle : " + str(time_exe)+"\n")
fic.write("Temps d'entraînement moyen par epoch : " + str(time_av)+"\n")
fic.write("Plus petite perte sur l'entraînement : " + str(err_min_train)+"\n")
fic.write("Plus petite perte sur la validation : " + str(err_min_val)+"\n")
fic.write(f"Norme de gradient minimale : {err_min_grad : .2e}\n")
fic.write("Erreur euclidienne IFPEN/CopyCat (sur Fn) : "+ str(err_norm_IF_CC) + "\n")
fic.write("Erreur euclidienne IFPEN/SkyNet (sur Fn) : " + str(err_norm_IF_SKN) + "\n")
fic.write("Erreur euclidienne CopyCat/SkyNet (sur Fn) : " + str(err_norm_CC_SKN) + "\n")
fic.write("*******************************************************************************************************\n\n\n\n\n")
##############################################################################################################