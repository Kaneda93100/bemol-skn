
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bemol

print(dir(bemol))
mexico_vortex = bemol.rotor.mexico_vortex
