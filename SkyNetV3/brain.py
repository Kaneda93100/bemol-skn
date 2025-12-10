
"""
On trouvera sur ce fichier toutes les fonctions nécéssaires à la création d'un NN ainsi qu'à son entraînement.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn, optim
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from time import perf_counter


data_type = torch.float64



class SkyNet(nn.Module) : 
    bias : bool #Bias
    in_size : int #Taille des vecteurs d'entré
    out_size : int #Taille du vecteur retourné
    layer_size : int #Nombre de neurone par couche
    deepness : int #Profondeur du réseau
    device : str #CPU ou GPU
    type : str #format des données manipulées
    """    
    def __init__(self,in_size:int, out_size:int, layer_size:int, bias:bool, deepness:int, type = data_type) : 
        
        Constructeur par défaut d'un objet sans architecture.

        
        self.bias = bias
        self.in_size = in_size
        self.out_size = out_size
        self.layer_size = layer_size
        self.deepness = deepness
        
    """
    def __init__(self,act, in_size:int, out_size:int, layer_size:int, bias:bool, deepness:int, device:str, type:str):
        """
        Constructeur de l'architecture du réseau.
        """

        super(SkyNet, self).__init__()

        self.bias = bias
        self.in_size = in_size
        self.out_size = out_size
        self.layer_size = layer_size
        self.deepness = deepness
        self.device = device
        self.type = type
        
        self.HL = nn.ModuleList()
        self.HL.append(nn.Linear(in_size, layer_size, bias = bias, dtype = type, device=device))

        for i in range(self.deepness) : 
            self.HL.append(nn.Linear(layer_size, layer_size, bias = bias, dtype = type, device=device))
            self.HL.append(act(inplace = True))
            self.HL.append(nn.Linear(layer_size, layer_size, bias = bias,dtype = type, device=device))

        self.HL.append(nn.Linear(layer_size, out_size, bias = bias, dtype = type, device=device))


    def init_weight(self, path:str, samp_size:int, transfert = True, random_init = False) :
        """
        Initialisation des paramètres (poids et biais) d'un réseau de neurones. 
        Si un entraînement à eu lieu précédemment, alors ce sont ces paramètres qui sont prit, 
        sinon ils sont initialisé aléatoirement.
        """

        if transfert == True and random_init == False:
            """
            Dans ce cas, path n'a besoins d'être SKNV3_results/ 
            """
            in_size = self.in_size
            layer_size = self.layer_size
            deepness = self.deepness
            bias = self.bias

            dir_param = path + "/Bias_" + str(bias) + "/in_size_" + str(in_size) + "/samp_size_" + str(samp_size) + "/layer_size_" + str(layer_size) + "/deepness_" + str(deepness)
            file_param = dir_param + "/best_param.pt2"

            prec_test = {'train' : 1, 'val' : 1}
            param_saved = [self.state_dict(), prec_test]
            if os.path.exists(dir_param) : 
                self.load_state_dict(torch.load(file_param))
                param_saved[0] = self.state_dict()
                print("Des paramètres d'une simulation précédente ont été trouvé.\n")
                return file_param, False
            else : 
                os.makedirs(dir_param)
                print("Initialisation aléatoire.\n")
                return file_param, True
                

        if random_init == True : 
            print("Initialisation aléatoire.\n")
            return None, True
        else : 
            """
            Ici, path est le chemin du jeu de paramètre que l'on veut associer au NN.
            """
            if os.path.exists(path) :
                self.load_state_dict(torch.load(path))
                print("Paramétrage effectué : ", path, "\n")
                return None, False
            else : 
                print("Aucun jeu de paramètre n'a été trouvé. Le réseau est initialisé aléatoirement.\n")
                return None, True
        
    """
    def train_NN1(self, MAX_EPOCH = 10, LR = 1e-3,) : 
        loss_train= []
        loss_val = []
        rel_error_train = []
        rel_error_val = []
        grad_norm = []
        optimiser = optim.Adam(self.parameters(), lr = LR, weight_decay=1)
        loss_func = nn.MSELoss(reduction = 'mean')
        param_saved = [self.state_dict(), 0]

        #Gestion de l'affichage 
        step  = 1 #étape à afficher dans le terminal
        tronc = 5 #ordre de la troncature pour err_min_train et err_min_val
        tol = 1e-7 #seuil de tolérance pour l'erreur faite sur les données d'entraînement ET de validation


        print("Début de l'entraînement\n")
        perf_av = []
        start = perf_counter()
        for ep in range(MAX_EPOCH):
            self.train()
            
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

                label = label.squeeze(2).squeeze(1).type(torch.float32)
                AI_NN = T800(features).squeeze(2).squeeze(1)
                Ux, Uy = compute_velocity(wind = U, omega = omega,
                                        rad = features[:,0,1], azimuth = features[:,0,0],
                                        yaw=yaw, tilt=tiltAngle, precone=preconeAngle
                                        )
        return 0          
    """ 
    
    def save_param(self, path:str, save:bool) : 
        """
        Enregistre les paramètres d'un NN à une adresse donnée en paramètre (path)
        """
        if save == True : 
            torch.save(self.state_dict(), path)
            print("Paramétrage enregistré à l'adresse " , path)
        else : 
            print("Aucun paramètre n'a été enregistré.")
        return 0
    
    def forward(self,x) :
        """
        Évaluation du modèle en un point.
        """
        for Layer in self.HL :
            x = Layer(x)
        return x
    
def extract(path:str, type = data_type) : 
    """
    Cette fonction a pour but d'extraire des données se trouvant dans un fichier .csv (celle fournies par l'IFPEN à propos du vortex).
    Elle est spécialement adapté à la mise en page des .csv fournis (pas abstraite du tout).

    Arguments : 
    - path --> chemin menant à la feuille de données, str
    - type --> type que l'on souhaite attribuer aux données extraites. !!! Attention !!! Il faut que le format soit issu de la librairie PyTorch.
    
    Return : 
    - X --> Tenseur PyTorch contenant les covariables d'entraînement du modèle.
    - Y --> Tenseur PyTorch contenant les étiquettes de chaque covariable.

    """
    
    
    X = torch.zeros((2592,1,2), dtype=type) # Features
    Y = torch.zeros((2592,1,1), dtype=type) # Target

    ## Extraction des données depuis un .csv
    with open(path) as data :
        data = csv.reader(data,delimiter=',', quotechar='|' )
    
        for i in range(24) : 
            next(data)
        k = 0
        for s in data : 
            lbd = float(s[0])
        
            for i in range(1,len(s)): 

                features = torch.zeros((1,2), dtype=type)
                features[0][0] = np.radians((i-1)*5) # azimuth
                features[0][1] = lbd*2.26  # r/R
            
                X[(i-1) + k*72,:,:] = features # Features
                Y[(i-1) + k*72,:,:] = float(s[i]) # Target
        
            k+=1
    print("Extraction des données du .csv réussie !")
    print("----------------------------------------")

    return X,Y

def extract_line(path:str, num_line:int, type = data_type) :
    """
    Fonction qui permet d'extraire une ligne particulière d'une feuille .csv du fichier data_vortex_mexico 
    (donc fixer un élément puis faire varier les azimuths). (À tester).

    Arguments : 
    path     --> chemin menant à la feuille 
    num_line --> numéro de la ligne que l'on souhaite extraire
    type     --> format dans lequel les données sont extraites

    Return : 
    X --> liste d'azimuth
    Y --> liste d'effort associé
    """
    with open(path) as data : 
        line =  csv.reader(data, delimiter=',', quotechar='|')

        for i in range(24 + num_line) :
            next(line)
        row = next(line)
        radius = float(row[0])
        X = torch.tensor([np.radians(i*5) for i in range(len(row)-1)], dtype = data_type)
        Y = torch.tensor([float(row[i]) for i in range(1, len(row))])
        
        element = {'indice' : num_line, 'rayon' : radius}
        print("Indice de l'élément :  ", element['indice'],'\n', "Rayon normalisé : ", element["rayon"])
        return X,Y,element

def extract_rad(path:str, type = data_type):
    R = []
    with open(path) as data : 
        
        rad = csv.reader(data, delimiter = ',', quotechar = '|')
        
        for i in range(24) :
            next(rad)
        
        for s in rad : 
            R.append(float(s[0]))
    R_r = np.array(R, dtype = np.float64)

    return R_r

def data_organisation(X,Y, batch_size:int, dtype = data_type,test_size = 0.5, pin_memory=False, shuffle=True) : 

    if type(X) == torch.Tensor and type(Y) == torch.Tensor :
        x_train, x_val , y_train, y_val = train_test_split(X,Y, random_state = 3, shuffle = True, test_size = test_size)
    else : 
        x_train, x_val , y_train, y_val = map(torch.tensor, train_test_split(X,Y, random_state = 0, shuffle = True, test_size = test_size))
    
    x_train = x_train.type(dtype); x_val = x_val.type(dtype)
    y_train = y_train.type(dtype); y_val = y_val.type(dtype)

    """
    print("x_train : \n", x_train, "y_train : \n", y_train)
    print("x_val : \n", x_val, "y_val : \n", y_val)
    """
    
    train_dataloader = DataLoader(TensorDataset(x_train, y_train),
                                                 batch_size = batch_size,
                                                 pin_memory = pin_memory,
                                                 shuffle = shuffle
                                                )
    k = 0
    for t in train_dataloader : 
        k+=1
    print("Nombre d'éléments dans le train_dataloader : ", k)

    val_dataloader = DataLoader(TensorDataset(x_val, y_val),
                                                 batch_size = batch_size,
                                                 pin_memory = pin_memory,
                                                 shuffle = shuffle
                                                )
    k = 0 
    for t in val_dataloader : 
        k+=1
    print("Nombre d'éléments dans le val_dataloader : ", k)
    return train_dataloader, val_dataloader

class autodiff_interp :
    def __init__(self, x, y, device='cpu'):
        self.x = torch.as_tensor(x, dtype = torch.float64, device = device)
        self.y = torch.as_tensor(y, dtype = torch.float64, device = device)
        self.device = device

    def __call__(self, x_new):
        x_new = torch.as_tensor(x_new, dtype = torch.float64, device = self.device)
        
        id_x = torch.searchsorted(self.x,x_new) - 1
        id_x = torch.clamp(id_x, 0, len(self.x) - 2)

        p = (self.y[id_x + 1] - self.y[id_x])/(self.x[id_x+1] - self.x[id_x])

        return p*(x_new - self.x[id_x+1]) + self.y[id_x+1]

def dicho_ad(f,init,eps = 1e-12, iter_max = 500, device ='cpu') :
    a = init[0]
    b = init[1]
    for i in range(iter_max) :
        temp = (a+b)/2
        if f(a)*f(temp) <= 0 :
            b = temp
        else : 
            a = temp
        
        if abs(a-b) < eps :
            break
    root = torch.tensor((a+b)/2, device = device, requires_grad = True)
    return root

def brent_dekker_ad(f, a,b, type = torch.float64, eps = 1e-8, iter_max = 1000, device = 'cpu') : 
    a = torch.as_tensor(a, dtype= type, device = device); b = torch.as_tensor(b, dtype = type, device = device)
    prec = a.clone()

    for i in range(iter_max) : 
        M = b.clone() if torch.abs(f(b)) >= torch.abs(f(a)) else a.clone()
        s = M - f(M)*(M - prec )/(f(M)-f(prec))
        m = (a+b)/2
        CP = s if (s >= m and s <= b) else m
        if(f(a)*f(CP) <= 0):
            b = CP
            prec = M
        else : 
            a = CP
            prec = M
        if torch.abs(a-b) < eps :
            break

    ## Gradient en f
    root = ((a+b)/2).to(device).requires_grad_(True) 
    image = f(root)
    image.backward()
    return root

def Newton_autodiff(f, x0:float, device = 'cpu',eps = 1e-12, iter_max = 500):
    x = torch.tensor(x0, dtype = torch.float32, device=device, requires_grad=True).unsqueeze(0)
    x.retain_grad()
    i=0

    while i < iter_max :
        if x.grad is not None:
            x.grad.zero_()

        y = f(x)
        y.backward(retain_graph = True)

        if x.grad is None :
            raise ValueError("Le gradient de x n'a pas été correctement calculé.")
        x_new = x - y/x.grad
        
        if abs(x_new.item() - x.item()) < eps :
            break
        x = x_new.detach().requires_grad_()
        x.retain_grad()
        i+=1

    x_final = torch.tensor(x_new.item(), device = device ,dtype = torch.float32, requires_grad = True).unsqueeze(0)
    x_final.retain_grad()

    y = f(x_final)
    y.backward()
    return x_final, 1/x_final.grad

def print_graph(fn, f):
    if fn is None:
        return
    f.write(str(fn) + "\n")
    for next_fn, _ in fn.next_functions:
        print_graph(next_fn, f)


print("Module brain.py fonctionnel.\n")
