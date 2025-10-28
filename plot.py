import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter

list = [
    0.21410600,
    0.22639200,
    0.246758000,
    0.275040000,
    0.311012000,
    0.35438200,
    0.404803000,
    0.461867000,
    0.525116000,
    0.594040000,
    0.668085000,
    0.746654000,
    0.829114000,
    0.914803000,
    1.003029000,
    1.093082000,
    1.184238000,
    1.275762000,
    1.366918000,
    1.456971000,
    1.545197000,
    1.630886000,
    1.713346000,
    1.791915000,
    1.865960000,
    1.934884000,
    1.998133000,
    2.055197000,
    2.10561800,
    2.148988000,
    2.184960000,
    2.213242000,
    2.23360800,
    2.24589400
]

X  = np.zeros(len(list))

for i in range(len(list)) :
    X[i] = list[i]


plt.plot(X)
plt.show()