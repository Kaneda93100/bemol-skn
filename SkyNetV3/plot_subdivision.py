import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import brain as brain

radius_input = [
    0.0009513254770636137, 0.004749387224011609, 0.012316605174680945,
      0.02359538823125587, 0.03849989804436044, 0.05691670229472784, 
      0.07870563798164243, 0.10370087814800763, 0.13171219392361866, 0.1625264022817283, 0.19590898849060362, 0.2316058909122384, 0.2693454345648251,
        0.308840398733408, 0.3497902028929527, 0.3918831943076373, 0.43479901989635283, 0.47821106431308547, 0.5217889356869145, 0.5652009801036472, 
        0.6081168056923628, 0.6502097971070474, 0.6911596012665921, 0.7306545654351748, 0.7683941090877615, 0.8040910115093964, 0.8374735977182717, 
        0.8682878060763813, 0.8962991218519925, 0.9212943620183576, 0.9430832977052722, 0.9615001019556396, 0.9764046117687442, 0.9876833948253191,
          0.9952506127759884, 0.9990486745229363
        ]

radius_vortex = brain.extract_rad("data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv")

forces = brain.extract_column("data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv", 5)
fig1 = plt.figure(figsize=(10,10))
plt.plot(radius_vortex, forces, color = 'blue', label = 'forces at az 35deg')
plt.legend()
plt.grid()

"""
fig2 = plt.figure(figsize=(10,10))
plt.plot(radius_vortex, color = 'red')
plt.title("Rayon issus des feuille .csv")
plt.grid()
"""

plt.show()


