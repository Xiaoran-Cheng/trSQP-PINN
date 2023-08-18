import jax.numpy as jnp
from jax import random
import numpy as np

from System import Transport_eq, Reaction_Diffusion



class Data:
    def __init__(self, N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, system) -> None:
        self.N = N
        self.IC_M = IC_M
        self.pde_M = pde_M
        self.BC_M = BC_M
        self.M = IC_M + pde_M + BC_M
        self.xgrid = xgrid
        self.nt = nt
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max
        self.beta = beta
        self.noise_level = noise_level
        self.system = system
        self.nu = nu
        self.rho = rho


    def data_grid(self):
        x = jnp.arange(self.x_min, self.x_max, self.x_max/self.xgrid)
        t = jnp.linspace(self.t_min, self.t_max, self.nt).reshape(-1, 1)
        X, T = np.meshgrid(x, t)
        X_star = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        return X_star


    def generate_data(self, key_num):
        X_star = self.data_grid()
        X_star = X_star[random.choice(random.PRNGKey(key_num), shape=(self.N,), a=len(X_star), replace=False),:]
        xi = X_star[:,0].reshape(1,self.N)
        ti = X_star[:,1].reshape(1,self.N)

        if self.system == "convection":
            ui = Transport_eq(beta=self.beta).solution(xi, ti) + random.uniform(random.PRNGKey(key_num), \
                                                    shape=(1,self.N), minval=-self.noise_level, maxval=self.noise_level)
            # ui = Transport_eq(beta=self.beta).solution(xi, ti)
        elif self.system == "reaction_diffusion":



            ui = Reaction_Diffusion(self.nu, self.rho).solution(xi, ti) + random.uniform(random.PRNGKey(key_num), \
                                                    shape=(1,self.N), minval=-self.noise_level, maxval=self.noise_level)
            # ui = Reaction_Diffusion(self.nu, self.rho).solution(xi, ti)
        data = jnp.concatenate((xi.T, ti.T), axis=1)
        return data, ui


    def sample_data(self, key_num):
        X_star = self.data_grid()
        X_star = X_star[random.choice(random.PRNGKey(key_num), shape=(self.M,), a=len(X_star), replace=False),:]
        xj = X_star[:,0].reshape(1,self.M)
        tj = X_star[:,1].reshape(1,self.M)
        sample_data_x, IC_sample_data_x= xj[:,:self.pde_M], xj[:,self.pde_M:self.pde_M+self.IC_M]
        sample_data_t, BC_sample_data_t = tj[:,:self.pde_M], tj[:,self.pde_M+self.IC_M:]
        pde_sample_data = jnp.concatenate((sample_data_x, sample_data_t), axis = 0).T
        IC_sample_data = jnp.concatenate((IC_sample_data_x, jnp.zeros((1,self.IC_M))), axis=0).T
        BC_sample_data_2pi = jnp.concatenate((jnp.ones((1,self.BC_M)) * 2 * jnp.pi, BC_sample_data_t), axis=0).T
        BC_sample_data_zero = jnp.concatenate((jnp.zeros((1,self.BC_M)), BC_sample_data_t), axis=0).T
        return pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi


    def get_eval_data(self):
        X_star = self.data_grid()
        data_grid_len = self.xgrid*self.nt
        xi = X_star[:,0].reshape(1,data_grid_len)
        ti = X_star[:,1].reshape(1,data_grid_len)
        if self.system == "convection":
            ui = Transport_eq(self.beta).solution(xi, ti)
        elif self.system == "reaction_diffusion":
            xi = jnp.arange(self.x_min, self.x_max, self.x_max/self.xgrid)
            ti = jnp.linspace(self.t_min, self.t_max, self.nt).reshape(-1, 1)
            ui = Reaction_Diffusion(self.nu, self.rho).solution(xi, ti)
        return X_star, ui





# from Visualization import Visualization
# import sys
# import os
# import numpy as np
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# current_dir = os.getcwd().replace("\\", "/")
# sys.path.append(parent_dir)
# visual = Visualization(current_dir)

# # x = jnp.arange(0, 2*jnp.pi, 2*jnp.pi/256)
# # t = jnp.linspace(0, 1, 100).reshape(-1, 1)
# # X, T = np.meshgrid(x, t)
# # X_star = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))


# # xgrid = x.shape[0]

# # sol= Reaction_Diffusion(5,5).solution(x, t)
# N=256
# IC_M=3
# pde_M=3
# BC_M=3
# xgrid=256
# nt=100 
# x_min=0
# x_max=2*jnp.pi
# t_min=0
# t_max=1
# beta=30
# noise_level=0.01
# nu=5 
# rho=5 
# system='reaction_diffusion'
# X_star, sol = Data(N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, system).generate_data(256)


# visual.heatmap(X_star, sol, "", "", "", 1, 100, 256)

