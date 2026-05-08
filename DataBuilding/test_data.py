import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pathlib as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plot_n_comp as pnc

path_bem    = p.Path('/home/arthur/Documents/GitHub/bemol-skn/DataBuilding/BEM_data_YAW_on.xlsx')
path_sven   = p.Path('/home/arthur/Documents/GitHub/bemol-skn/DataBuilding/results_yaw_15.0deg.csv')

arg = 'Ft'
abs = 'r'

rad, d1,d2 = pnc.preplot_attr(path_bem, path_sven, abs, arg)

B20 = d1[2*36:3*36] ; B90 = d1[9*36:10*36]; B140 = d1[14*36:15*36]
S20 = d2[2*36:3*36] ; S90 = d2[9*36:10*36]; S140 = d2[14*36:15*36]

fig = plt.figure(figsize=(15,12))

plt.subplot(1,3,1)
plt.plot(rad, B20, lw = 2, label = 'BEM')
plt.plot(rad, S20, lw = 2, label = 'SVEN')
plt.title('azimuth : 20°')
plt.xlabel(f'{abs}')
plt.ylabel(f'{arg}')
plt.grid()
plt.legend()

plt.subplot(1,3,2)
plt.plot(rad, B90, lw = 2, label = 'BEM')
plt.plot(rad, S90, lw = 2, label = 'SVEN')
plt.title('azimuth : 90°')
plt.xlabel(f'{abs}')
plt.ylabel(f'{arg}')
plt.grid()
plt.legend()

plt.subplot(1,3,3)
plt.plot(rad, B140, lw = 2, label = 'BEM')
plt.plot(rad, S140, lw = 2, label = 'SVEN')
plt.title('azimuth : 20°')
plt.xlabel(f'{abs}')
plt.ylabel(f'{arg}')
plt.grid()
plt.legend()


plt.show()