import jax.numpy as jnp
from Transport_eq import Transport_eq
from jax import random
import numpy as np

class Data:
    def __init__(self, N:int, M:int, dim:int) -> None:
        self.N = N
        self.M = M
        self.dim = dim


    def data_grid(self, xgrid, nt, x_min, x_max, t_min, t_max):
        x = jnp.arange(x_min, x_max, x_max/xgrid)
        t = jnp.linspace(t_min, t_max, nt).reshape(-1, 1)
        X, T = np.meshgrid(x, t)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        return X_star
    

    def generate_data(self, xgrid, nt, x_min, x_max, t_min, t_max, beta, key_num):
        key = random.PRNGKey(key_num)
        X_star = self.data_grid(xgrid, nt, x_min, x_max, t_min, t_max)
        data_index = random.randint(key, shape=(self.N,), minval=0, maxval=len(X_star))
        X_star = X_star[data_index,:]
        xi = X_star[:,0].reshape(self.dim-1,self.N)
        ti = X_star[:,1].reshape(self.dim-1,self.N)
        ui = Transport_eq(beta=beta).solution(xi, ti)
        return xi, ti, ui
    

    def sample_data(self, xgrid, nt, x_min, x_max, t_min, t_max, key_num):
        key = random.PRNGKey(key_num)
        X_star = self.data_grid(xgrid, nt, x_min, x_max, t_min, t_max)
        data_index = random.randint(key, shape=(self.M,), minval=0, maxval=len(X_star))
        X_star = X_star[data_index,:]
        xj = X_star[:,0].reshape(self.dim-1,self.M)
        tj = X_star[:,1].reshape(self.dim-1,self.M)
        return xj, tj
    


