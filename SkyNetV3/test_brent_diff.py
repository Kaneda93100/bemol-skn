"""
    Maintenant, on essaie de généraliser cette méthode avec torch.autograd. L'idée est de pouvoir afficher
    le graphique de phi' sans utiliser la formule.
"""

import torch
import numpy as np
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
import brain


##Fonction dont on veut calculer l'inverse
def h2(z):
    if isinstance(z, np.ndarray):
        return  z*np.cosh(z)
    if isinstance(z, torch.Tensor):
        return  z*torch.cosh(z)

def h(z) : 
    if isinstance(z, np.ndarray) :
        return z**3
    if isinstance(z, torch.Tensor):
        return z**3
def id (z) : 
    return z
def h_rec (x) :
    return np.cbrt(x)
def h_rec_prime(x) : 
    if isinstance(x , torch.Tensor) :
        return 1/(3*torch.pow(torch.abs(x), 2/3))
    if isinstance(x, np.ndarray) : 
        y = (3*np.pow(np.abs(x), 2/3))
        return 1/y

N = 1000 #Pas de discrétisation
eps = 1e-12 #tolérance
iter_max = 5000 #nombre maximal d'itération
bord = [-1,1] 
x0 = 1/2 #init pour newton

def Id(x) : 
    return x

print("point d'annulation : ", brain.brent_dekker_ad(torch.sin, torch.pi/2, torch.pi, torch.float32, device='cpu'))

I0 = np.linspace(bord[0], bord[1], N)
I1 = np.arctan(I0)
I = I0
I_plot = I0
H = h(I)
Identity = id(I0)
phi = np.zeros((len(I1), 1))
grad_phi = np.zeros((len(I0), 1))   

#phi, grad_phi = auxiliar.compute_inverse(h, I, 0.5) 


for i, d in enumerate(I):
    d_tens = torch.tensor(d, dtype = torch.float32, requires_grad = True)
    f = lambda z : h(z) - d_tens
    phi0 = brain.brent_dekker_ad(f,-1,1)
    phi[i,0] = phi0.item(); grad_phi[i,0] = phi0.grad.item()

crbt = h_rec(I)
crbt_prime = h_rec_prime(I)
print(np.max(phi[:,0] - crbt))

if len(phi) != len(I) :
    raise ValueError("Les listes phi et I ne sont pas de la même taille.\n")

plt.subplot(3,1,1)
plt.plot(I_plot,phi[:,0], label = 'Phi', color = 'blue')
plt.plot(I_plot, H, label = 'h', color = 'red')
plt.legend()
plt.grid()

if len(H) != len(I) : 
    raise ValueError("Les liste h  et I ne sont pas de la même taille.\n")

"""
plt.subplot(4,1,3)
plt.plot(I_plot, Identity, label = 'arctan', color = 'purple')
plt.legend()
plt.grid()
"""

plt.subplot(3,1,2)
plt.plot(I_plot, H, label = 'h', color = 'red')
plt.plot(I_plot, phi[:,0], label ='phi', color = 'blue')
plt.plot(I_plot, Identity, label = 'Id', color = 'green')
plt.legend()
plt.grid()

plt.subplot(3,1,3)
plt.plot(I_plot, grad_phi[:,0], label = 'auto diff')
plt.plot(I_plot, crbt_prime, label ='symbolic')
plt.legend()
plt.title('Dérivée de phi')
plt.grid()

plt.show()



