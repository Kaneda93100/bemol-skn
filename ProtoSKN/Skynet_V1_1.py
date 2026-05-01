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



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CPU or GPU (GPU --> cuda) ? --> ", device)

################## Hyperparamètres et NN ##################

LR = 1 * 1e-5
MAX_EPOCH = 10

in_size = 4
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
###################### RESEAU DE NEURONES SKYNET ###################### 
#######################################################################

"""
    Cette version de SkyNet doit être capable de s'entraîner sur les rayons. Une fois que ça fonctionnera correctement, 
    on l'adaptera à SkyNet V2 puis on s'en servira pour écrire SkyNet V3.

    De plus, je ne tire plus aléatoirement les données, je crois que ce n'est pas utile, et c'est surement plus pertinent d'entraîner sur les mêmes données à chaque fois.
"""

##Déclaration préliminaire
AIProcess = data_processing.compute_AI
Yaw_corr = data_processing.yaw
Burton = data_processing.Burton

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

pi = torch.pi
type = torch.float32


#Azimuths € [0, 2Npi]
delta_az  = 2*N*pi/samp_size
Azimuths = torch.arange(0.0, 2*N*pi + delta_az, delta_az, dtype = type)

#Yaw € [-pi/3, pi/3]
delta_yaw = 2*PI/(3*samp_size) 
Yaw = torch.arange(-pi/3, pi/3 + delta_yaw, delta_yaw, dtype = type)

#Wind € [10,20]
delta_wind = (15.06-10)/samp_size
Wind = torch.arange(10 , 15.06 + delta_wind, step = delta_wind, dtype = type) 

"""
    Pour chaque élément issu de la discrétisation de la pale, on calcule l'induction axiale sans correction pour chaque sample
"""

X = torch.zeros(24 * samp_size, in_size)
Y = torch.zeros(24 * samp_size)

"""
    Cette boucle risque d'être lourde en calcul, on pourrait tout à fait étudier la parallélisation du processus.
"""
index_element = 6
elements = torch.arange(10, 34, 1)

for i, el  in enumerate(elements) :
    print("Element ")
    print("Element " + str(el) + ":" + str(solver_noyaw.rotor.sections[el].radius/TR))
    for j in range(samp_size) :
        #Axial induction
        AI, _ = AIProcess(solver_noyaw,
                        az = Azimuths[j], yaw = Yaw[j], wind = 15.06, #Paramètres bouclant sur j
                        element = [el] #Paramètres bouclant sur i
                        )
        
        #Features
        X[i*samp_size + j, 0] = el
        X[i*samp_size + j, 1] = AI
        X[i*samp_size + j, 2] = Yaw[j]
        X[i*samp_size + j, 3] = Azimuths[j]
       # X[i*samp_size + j, 4] = Wind[j]

        #Labels
        Y[i*samp_size + j] = Yaw_corr(WSA = AI, AA = Azimuths[j], R = solver_noyaw.rotor.sections[el].radius)


x_train, x_val, y_train, y_val = map(torch.tensor, train_test_split(X, Y, test_size=0.2))

plot = x_val[:, 1].size(0)

##Forcer le typage des tenseurs en float32 (visiblement en float64 par défaut)
x_train = x_train.type(type)
y_train = y_train.type(type)

x_val = x_val.type(type).to(device)
y_val = y_val.type(type).to(device)

train_dataloader = DataLoader(TensorDataset(x_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size = batch_size, pin_memory = False, shuffle = True)
val_dataloader = DataLoader(TensorDataset(x_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size = batch_size, pin_memory = False, shuffle = True)


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
T800 = SkyNet(deepness, type, True).to(device) ##Indiquer si on entraîne sur CPU ou GPU

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
corrections_yaw = [
    bemol.secondary.HubTipLoss.Prandtl,
    bemol.secondary.SkewAngle.Burton,
    bemol.secondary.TurbulentWakeState.Buhl,
    bemol.secondary.YawModel.IFPEN
]
solver_yaw = bemol.ning.NingUncoupled(mexico, rho, corrections_yaw)

forces, AI_yawed , azs = solver_yaw.cycle(
        mexico.pitchRated, wind, omega, angles=[yawAngle, tiltAngle], tStep=tStep,
        n_phi=180, N=N, elements=[24],
)

####Test de la correction que j'ai réécrite
forces_CopyCat = np.zeros((180,1))
solver_validation = bemol.ning.NingUncoupled(mexico, rho, corrections_noyaw)

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
    
    X = torch.tensor([24, wakeSkewAngle, az, yawAngle], dtype=torch.float32, device=device)
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

plt.show()

