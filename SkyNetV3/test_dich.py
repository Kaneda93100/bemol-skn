import torch
import numpy as np
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
import brain

def f(x) : 
    return x**2

I = torch.linspace(0,1, 1000)
arcsin = torch.zeros_like(I)

print("zéro de la fonction : ", brain.brent_dekker_ad(f, -0,3, type=torch.float64, device='cpu').item())

for i, y in enumerate(I) : 
    inv = lambda x : f(x) - y
    arcsin[i] = brain.brent_dekker_ad(inv, 0, 5, type = torch.float64, device='cpu').item()

plt.plot(I.numpy(),arcsin.detach().numpy(), label = 'approximation', color = 'red')
plt.plot(I.numpy(), f(I.numpy()), label = 'fonction exacte', color = 'blue')
plt.legend()
plt.grid()

plt.show()
