import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pathlib as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

bem_Yon_file    = p.Path('/home/arthur/Documents/GitHub/bemol-skn/DataBuilding/BEM_data_YAW_on.xlsx')
bem_Yoff_file   = p.Path('/home/arthur/Documents/GitHub/bemol-skn/DataBuilding/BEM_data_YAW_off.xlsx')

data_BEM_Yon    = pd.read_excel(bem_Yon_file)
data_BEM_Yoff   = pd.read_excel(bem_Yoff_file)


Ft_Yon  = np.zeros((36,36))
Ft_Yoff = np.zeros((36,36))

for i in range(36) : 
    Ft_Yon[:,i]  = data_BEM_Yon['Ft'].to_numpy()[i*36:(i+1)*36]
    Ft_Yoff[:,i] = data_BEM_Yoff['Ft'].to_numpy()[i*36:(i+1)*36]

rad = data_BEM_Yoff['r'].to_numpy()[0:36]
azs = np.linspace(0,360,36, endpoint=False)

save_plot = True
if save_plot == True :
    for i in range(36) :
        fig = plt.figure(figsize=(18,15))

        plt.plot(rad,Ft_Yon[:,i], label = 'Ft, yaw on')
        plt.plot(rad,Ft_Yoff[:,i], label = 'Ft, yaw off')
        plt.title(f"Azimuth : {azs[i]}")
        plt.xlabel("rayon")
        plt.ylabel("Ft N/m")

        plt.legend()
        plt.grid()
        plt.savefig('/home/arthur/Documents/GitHub/bemol-skn/DataBuilding/fig_ft/ft_az'+f'{azs[i]}.png')
        plt.close()

# Initialisation du graphique
fig, ax = plt.subplots()
line1, = ax.plot([], [], lw=2, label = 'Yaw off')
line2, = ax.plot([], [], lw=2, label = 'Yaw on')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1,line2,

def animate(i):
    x_i = rad
    y_i = Ft_Yoff[36*i:36*(i+1)]
    z_i = Ft_Yon[36*i:36(i+1)]
    line1.set_data(x_i, y_i)
    line2.set_data(x_i, z_i)
    return line1,line2,

# Création de l'animation
ani = animation.FuncAnimation(fig, animate, frames=36, init_func=init, blit=True)

plt.show()
x = 0