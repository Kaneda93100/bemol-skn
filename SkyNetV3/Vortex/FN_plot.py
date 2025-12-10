import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import brain
import matplotlib.pyplot as plt
import torch 

plt.rcParams['lines.linewidth'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18

plt.style.use('seaborn-v0_8-poster')

P = "element"
X = torch.arange(0,360, 5, dtype = torch.float64)
for i in range (36) : 
    Path = "/home/arthur/Bureau/BEM IFPEN/BEM IFPEN/bemol-v0.0.1/SkyNetV3/Vortex/plot" + P + str(i)
    _, Fn, _ = brain.extract_line('data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv', i, type = torch.float64)
    fig = plt.figure(figsize = (30,30))
    plt.plot(X, Fn.numpy(), label = 'efforts normaux', color = 'blue')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    title = "Efforts normaux sur l'élément " + str(i) 
    plt.title(title)

    plt.xlabel("azimuths")
    plt.ylabel("Efforts normaux")
    
    plt.legend()
    plt.grid()   
    
    plt.savefig(Path)