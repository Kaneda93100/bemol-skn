import pathlib as p
import pandas as pd
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
import os
import yaml 

path_blade_vor = p.Path('/home/arthur/Documents/GitHub/bemol-skn/bemol/rotors/mexico_vortex/blade.dat')
temp = pd.read_csv(path_blade_vor, sep = '\s+', names = ['radius', 'twist', 'chord', 'airfoil'], header = 0)
temp['radius'] *= 1/2.25
temp.to_csv(('/home/arthur/Documents/GitHub/bemol-skn/bemol/rotors/mexico_vortex/blade.dat'), sep = '\t', header=True)

path_blade_yml = p.Path('/home/arthur/Documents/GitHub/bemol-skn/bemol/rotors/mexico_vortex/rotor.yml')

with open(path_blade_yml, 'r') as file :
    data_rot = yaml.safe_load(file)

data_rot['pitchRated'] *= -1

with open(path_blade_yml, 'w') as file :
    yaml.dump(data_rot, file)



x = 0