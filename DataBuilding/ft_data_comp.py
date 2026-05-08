import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tools_blade.write_blade as wb 

sven_data   = pd.read_excel('/home/arthur/Documents/GitHub/bemol-skn/DataBuilding/dataset_forces_mexico.xlsx')
bem_data    = pd.read_excel('/home/arthur/Documents/GitHub/bemol-skn/DataBuilding/BEM_data_YAW_off.xlsx')
castor_data = pd.read_csv('/home/arthur/Documents/GitHub/bemol-skn/data_vortex_yaw_mexico/data_vortex_mexico_tsr008_yaw015_ft.csv', comment = '#', delimiter = ',', quotechar = '"')

rad      = castor_data['span'].to_numpy()*2.25

## Tester l'azimuth 0.0°
castor0 = castor_data['0.0'].to_numpy()
bem0    = bem_data['Ft'][0:36].to_numpy()
sven0   = -sven_data['Ft'][0:36].to_numpy()

## Tester l'azimuth 90.0°
castor90 = castor_data['90.0'].to_numpy()
bem90    = bem_data['Ft'][36*9:36*10].to_numpy()
sven90   = -sven_data['Ft'][36*9:36*10].to_numpy()

## Tester l'azimuth 180°
castor180 = castor_data['180.0'].to_numpy()
bem180    = bem_data['Ft'][36*18:36*19].to_numpy()
sven180   = -sven_data['Ft'][36*18:36*19].to_numpy()

fig = plt.figure(figsize=(15,10))

plt.suptitle("Test des Ft sur les azimuths 0, 90 et 180")

plt.subplot(3,1,1)
plt.plot(rad, castor0, label = 'castor')
plt.plot(rad, bem0, label = 'bem')
plt.plot(rad, sven0, label = 'sven')
plt.xlabel('rayon')
plt.ylabel('Ft N/m')
plt.title("Azimuth : 0.0°")
plt.grid()
plt.legend()

plt.subplot(3,1,2)
plt.plot(rad, castor90, label = 'castor')
plt.plot(rad, bem90, label = 'bem')
plt.plot(rad, sven90, label = 'sven')
plt.xlabel('rayon')
plt.ylabel('Ft N/m')
plt.grid()
plt.legend()
plt.title("Azimut : 90.0°")

plt.subplot(3,1,3)
plt.plot(rad, castor180, label = 'castor')
plt.plot(rad, bem180, label = 'bem')
plt.plot(rad, sven180, label = 'sven')
plt.xlabel('rayon')
plt.ylabel('Ft N/m')
plt.grid()
plt.legend()
plt.title("Azimut : 180.0°")

plt.show()
x = 0