import jax.numpy as jnp
from jax import random
import numpy as np

from System import Transport_eq, Reaction_Diffusion, Reaction



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
        if self.system == "convection":
            X_star = self.data_grid()
            X_star = X_star[random.choice(random.PRNGKey(key_num), shape=(self.N,), a=len(X_star), replace=False),:]
            xi = X_star[:,0].reshape(1,self.N)
            ti = X_star[:,1].reshape(1,self.N)
            ui = Transport_eq(beta=self.beta).solution(xi, ti) + random.uniform(random.PRNGKey(key_num), \
                                                    shape=(1,self.N), minval=-self.noise_level, maxval=self.noise_level)
        elif self.system == "reaction_diffusion":
            x = jnp.arange(self.x_min, self.x_max, self.x_max/self.xgrid)
            t = jnp.linspace(self.t_min, self.t_max, self.nt).reshape(-1, 1)
            X_star = self.data_grid()
            index = random.choice(random.PRNGKey(key_num), shape=(self.N,), a=len(X_star), replace=False)
            ui = Reaction_Diffusion(self.nu, self.rho).solution(x, t)[index] + random.uniform(random.PRNGKey(key_num), \
                                                    shape=(self.N,), minval=-self.noise_level, maxval=self.noise_level)
            ui = ui.reshape(1,self.N)
            X_star = X_star[index,:]
            xi = X_star[:,0].reshape(1,self.N)
            ti = X_star[:,1].reshape(1,self.N)
        elif self.system == "reaction":
            X_star = self.data_grid()
            index = random.choice(random.PRNGKey(key_num), shape=(self.N,), a=len(X_star), replace=False)
            X_star = X_star[index]
            reaction = Reaction(self.rho)
            ui = reaction.solution(reaction.u0(X_star[:,0]), X_star[:,1]).reshape(1,self.N)
            xi = X_star[:,0].reshape(1,self.N)
            ti = X_star[:,1].reshape(1,self.N)
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
            ui = Reaction_Diffusion(self.nu, self.rho).solution(xi, ti).reshape(1, data_grid_len)
        elif self.system == "reaction":
            reaction = Reaction(self.rho)
            ui = reaction.solution(reaction.u0(xi[:,:256][0]), ti.reshape(100, 256)).reshape(1,data_grid_len)
        return X_star, ui
