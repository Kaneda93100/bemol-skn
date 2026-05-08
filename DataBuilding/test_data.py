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

rad, d1,d2 = pnc.preplot_attr(path_bem, path_sven, 'r', 'V_eff')

B_V_eff20 = d1[2*36:3*36] ; B_V_eff90 = d1[9*36:10*36]; B_V_eff140 = d1[14*36:15*36]
S_V_eff20 = d2[2*36:3*36] ; S_V_eff90 = d2[9*36:10*36]; S_V_eff140 = d2[14*36:15*36]

fig = plt.figure(figsize=(15,12))

plt.subplot(1,3,1)
plt.plot(rad, B_V_eff20, lw = 2, label = 'BEM')
plt.plot(rad, S_V_eff20, lw = 2, label = 'SVEN')
plt.title('azimuth : 20°')
plt.xlabel('r')
plt.ylabel('V_eff')
plt.grid()
plt.legend()

plt.subplot(1,3,2)
plt.plot(rad, B_V_eff90, lw = 2, label = 'BEM')
plt.plot(rad, S_V_eff90, lw = 2, label = 'SVEN')
plt.title('azimuth : 90°')
plt.xlabel('r')
plt.ylabel('V_eff')
plt.grid()
plt.legend()

plt.subplot(1,3,3)
plt.plot(rad, B_V_eff140, lw = 2, label = 'BEM')
plt.plot(rad, S_V_eff140, lw = 2, label = 'SVEN')
plt.title('azimuth : 20°')
plt.xlabel('r')
plt.ylabel('V_eff')
plt.grid()
plt.legend()


plt.show()