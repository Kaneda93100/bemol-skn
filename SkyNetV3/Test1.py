"""
Le premier test va consister à entrainer
SkyNet sur une feuille de donnée. Chaque
feuille contient 72*36 = 2592 donnée de sortie.
Donc une feuille devrai suffir à faire des trucs sympas.

Le régresseur a pour objectif de reproduire une induction axiale en apprenant
sur les forces normales
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import brain
import numpy as np
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

## Format dans lequel les données sont manipulées
type = torch.float32

print("Début de l'extraction des données...\n")
print("----------------------------------------")
Pre_X, Y = brain.extract('data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv', 
                         type = type)

X = torch.zeros((2592,1,5), dtype=type)
X[:,0,0] = Pre_X[:,0,0] ## Azimuths
X[:,0,1] = Pre_X[:,0,1] ## Radius

elements = np.zeros_like(X[:,0,0], dtype=int)
for i in range(36) : ## Position 
    for j in range(72) :
        elements[72*i + j] = i
X[:,0,2] = torch.as_tensor(elements, dtype=type)

for j, posi in enumerate(elements) :
    velocity = tools.calculateVelocity(wind=U, 
                                       omega=omega,
                                       rad = solver_noyaw.rotor.sections[posi].radius,
                                       azi = X[j,0,0],
                                       yaw=yaw,
                                       tilt=tiltAngle,
                                       precone=preconeAngle,
                                       )
    """
    solver_noyaw.update(
                        Ux = velocity[0], Uy = velocity[1],
                        chord = solver_noyaw.rotor.sections[posi].chord,
                        radius = solver_noyaw.rotor.sections[posi].radius,
                        angle = solver_noyaw.rotor.sections[posi].twist + pitch,
                        funLift = solver_noyaw.rotor.sections[posi].airfoil.cd,
                        funDrag = solver_noyaw.rotor.sections[posi].airfoil.cl,
                        )
    """
    """
    f = solver_noyaw.residuals

    I = np.linspace(np.pi/2, np.pi, 200)
    F = np.zeros_like(I)
    for i in range(len(I)):
        F[i] = f(I[i])

    plt.plot(I, F)
    plt.show()
    """
    _,_,axial_induction,tangential_induction = solver_noyaw.solve(section = solver_noyaw.rotor.sections[posi], ## Section de la pale sur laquelle on calcule le facteur
                                               azimuth=X[j,0,0], ## Usage de l'azimuth calculé avant
                                               pitch=pitch, ## Paramètre par défaut de la simulation
                                               velocity=velocity, 
                                               angles = [yaw,tiltAngle]
                                               )
    X[j,0,3] = axial_induction
    X[j,0,4] = tangential_induction
train_dataloader, val_dataloader = brain.data_organisation(X,Y, batch_size=200, dtype = type)

print("Données extraite avec succès et prête à l'utilisation !\n")
print("----------------------------------------")


## Paramètres du réseaux de neurones
bias = True
in_size = 5
out_size = 1
layer_size = 2500
deepness = 10
ReLU = nn.ReLU

T800 = brain.SkyNet(ReLU, in_size = in_size, out_size = out_size, 
              layer_size = layer_size,
              bias = bias, deepness = deepness, type = type, device = device
              )

path = "SkyNetV3/SKN_V3_V1_results/Parameters"
path_to_save = T800.init_weight(path = path, transfert = True, random_init = False ,samp_size = 2592)
#######################################################################
#######################################################################
#######################################################################



#######################################################################
########################### ENTRAINEMENT ##############################
#######################################################################

LR = 1e-4
MAX_EPOCH = 200
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
perf_av = []
start = perf_counter()
for ep in range(MAX_EPOCH):
    T800.train()
    
    if ep % step == 0 :
        print("Etape "+str(ep + 1)+" sur " + str(MAX_EPOCH) + "\n")
    
    loss_train_int = list()
    rel_err_int = []
    
    perf_loc = []
    
    for features, label in train_dataloader :
        start_train = perf_counter()

        features = features.to(device)
        label = label.to(device)

        optimiser.zero_grad()
    
        c = T800(features).squeeze(2).squeeze(1)
        AI_NN = c*features[:,0,3]
        Ux, Uy = compute_velocity(wind = U, omega = omega,
                                  rad = features[:,0,1], azimuth = features[:,0,0],
                                  yaw=yaw, tilt=tiltAngle, precone=preconeAngle
                                  )
        """
        fn_NN, _ = compute_forces(solver=solver_noyaw, Ux=Ux, Uy=Uy,
                                  x=AI_NN, y=features[:,0,4], index = features[:,0,2]
                                    )
        """
        """
        T = torch.zeros((Ux.shape[0],4))
        T[:,0] = Ux; T[:,1] = Uy; T[:,2] = AI_NN; T[:,3] = features[:,0,4]; index = features[:,0,2].to('cpu').type(torch.int32).unsqueeze(1)
        E = torch.gather(T, dim = 1, index=index)
        #fn_NN = brain.compute_forces(solver_noyaw, E[])
        
        """
        label = label.squeeze(2).squeeze(1).type(torch.float32)
        fn_NN_list = []
        for i, data in enumerate(features) : 
            fn, _ = compute_forces(solver = solver_noyaw, Ux = Ux[i], Uy = Uy[i], 
                                    x = AI_NN[i], y = features[i,0,4], 
                                    index = features[i,0,2]
                                    )
            fn_NN_list.append(fn)
        fn_NN = torch.stack(fn_NN_list)
        

        loss = loss_func(input=fn_NN.type(torch.float32), target=label)
        loss.backward()

        with open("SkyNetV3/grad_log.txt", "w") as f:
            brain.print_graph(loss.grad_fn, f)
        
        optimiser.step()
        loss_train_int.append(loss.detach().cpu().numpy())

        #Relative error
        numerator = np.linalg.norm((fn_NN - label).detach().cpu().numpy())
        denominator = np.linalg.norm(label.detach().cpu().numpy())
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
if path_to_save != None :
    torch.save(param_saved[0], path_to_save)
    print("Meilleur paramétrage trouvé enregistré à l'adresse : ", path_to_save)

T800.state_dict(param_saved[0]) #Chargement des meilleurs paramètres trouvés pendant l'entraînement

stop =  perf_counter()
#Mesure des performances temporelles de l'entraînement
time_exe = stop - start
time_av = np.average(perf_av)

fig = plt.figure(figsize=(20, 14))

#Perte sur entraînement & sur validation
plt.loglog(loss_train, color = 'r', label = 'Entraînement')
plt.loglog(loss_val, color = 'b', label = 'Validation')
plt.title('Evolution de la perte')
plt.legend()
plt.grid(True)


plt.show()