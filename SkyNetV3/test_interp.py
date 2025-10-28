import numpy as np
import brain
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

interp1d = interpolate.interp1d


polar = pd.read_csv("SkyNetV3/DU91-W2-250.foil", delim_whitespace = True, comment = '#', header = None, names=['AoA', 'Cl', 'Cd'])

X= polar['AoA'].to_numpy()
CD = polar['Cd'].to_numpy()
CL = polar['Cl'].to_numpy()



f_ad = brain.autodiff_interp(X,CD)
f_sp = interp1d(X, CD)

I = np.linspace(-180, 180, 1000)

fig1 = plt.figure(figsize=(20,15))
#plt.scatter(X, CD, color = 'purple', label = 'Data')
plt.plot(I, f_ad(I), color = 'red', label = 'Interpolator autodiff')
plt.plot(I, f_sp(I), color = 'blue', label = 'Interpolator scipy')
plt.legend()
plt.grid()
plt.show()

print(np.max(np.abs(f_sp(I)-f_ad(I).detach().numpy())))
fig2 = plt.figure(figsize=(20,15))
plt.plot(I, np.abs(f_sp(I)-f_ad(I).detach().numpy()), label = 'error')
plt.legend()
plt.grid()
plt.show()

