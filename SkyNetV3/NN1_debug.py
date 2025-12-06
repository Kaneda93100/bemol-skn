import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE = os.path.dirname(os.path.abspath(__file__))

import torch
from torch import nn, optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import brain
import numpy as np
import pandas as pd
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

type = torch.float32
samp_size = 72 ; batch_size = 9

X,Y, element = brain.extract_line('data_vortex_yaw_mexico/data_vortex_mexico_tsr004_yaw005_fn.csv', 7, type = type)

print(X,'\n', Y, '\n')