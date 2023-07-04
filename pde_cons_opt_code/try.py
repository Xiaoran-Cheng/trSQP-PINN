import numpy as np

x = np.linspace(0, 2*np.pi, 256, endpoint=False).reshape(-1, 1) # not inclusive
t = np.linspace(0, 1, 100).reshape(-1, 1)
X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data




