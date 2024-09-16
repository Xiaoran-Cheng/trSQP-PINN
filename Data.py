import jax.numpy as jnp
from jax import random
import numpy as np

from System import Transport_eq, Reaction_Diffusion, Reaction



class Data:
    def __init__(self, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, alpha, system) -> None:
        ''' PDE system parameters, including number of constraints, omega, T, and PDE coefficients'''
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
        self.alpha = alpha

    def generate_data(self, N, key_num, X_star, eval_ui):
        ''' Generate labeled data points and corresponding solutions with noise added '''
        if self.system == "transport":
            xi = random.uniform(random.PRNGKey(key_num), shape=(1,N), minval=self.x_min, maxval=self.x_max)
            ti = random.uniform(random.PRNGKey(key_num+1), shape=(1,N), minval=self.t_min, maxval=self.t_max)
            
            ui = Transport_eq(beta=self.beta).solution(xi, ti) + random.uniform(random.PRNGKey(key_num), \
                                                    shape=(1,N), minval=-self.noise_level, maxval=self.noise_level)
        elif self.system == "reaction_diffusion":
            index = random.choice(random.PRNGKey(key_num), shape=(N,), a=len(X_star), replace=False)
            data_grid_len = self.xgrid*self.nt
            ui = eval_ui.reshape(data_grid_len, 1)[index] + random.uniform(random.PRNGKey(key_num), \
                                                    shape=(N, 1), minval=-self.noise_level, maxval=self.noise_level)
            ui = ui.reshape(1,N)
            X_star = X_star[index,:]
            xi = X_star[:,0].reshape(1,N)
            ti = X_star[:,1].reshape(1,N)

        elif self.system == "reaction":
            reaction = Reaction(self.rho)
            xi = random.uniform(random.PRNGKey(key_num), shape=(1,N), minval=self.x_min, maxval=self.x_max)
            ti = random.uniform(random.PRNGKey(key_num+1), shape=(1,N), minval=self.t_min, maxval=self.t_max)
            ui = reaction.solution(reaction.u0(xi), ti).reshape(1,N) + random.uniform(random.PRNGKey(key_num), \
                                                    shape=(1,N), minval=-self.noise_level, maxval=self.noise_level)
        data = jnp.concatenate((xi.T, ti.T), axis=1)
        return data, ui


    def sample_data(self, key_num, X_star, eval_ui):
        ''' Generate unlabeled data for pretraining and constraints for transport, reaction and reaction-diffusion equations '''
        xj = random.uniform(random.PRNGKey(key_num), shape=(1,self.M), minval=self.x_min, maxval=self.x_max)
        tj = random.uniform(random.PRNGKey(key_num+1), shape=(1,self.M), minval=self.t_min, maxval=self.t_max)
        
        sample_data_x, IC_sample_data_x= xj[:,:self.pde_M], xj[:,self.pde_M:self.pde_M+self.IC_M]
        sample_data_t, BC_sample_data_t = tj[:,:self.pde_M], tj[:,self.pde_M+self.IC_M:]
        pde_sample_data = jnp.concatenate((sample_data_x, sample_data_t), axis = 0).T
        IC_sample_data = jnp.concatenate((IC_sample_data_x, jnp.zeros((1,self.IC_M))), axis=0).T
        BC_sample_data_2pi = jnp.concatenate((jnp.ones((1,self.BC_M)) * 2 * jnp.pi, BC_sample_data_t), axis=0).T
        BC_sample_data_zero = jnp.concatenate((jnp.zeros((1,self.BC_M)), BC_sample_data_t), axis=0).T
        IC_sample_data_sol = []

        if self.system == 'reaction_diffusion':
          IC_sample_data_sol = Reaction_Diffusion(self.nu, self.rho).u0(IC_sample_data_x).reshape(self.IC_M,)
        return pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi


    def get_eval_data(self, X_star):
        ''' Generate evaluation grid points and corresponding solutions '''
        data_grid_len = self.xgrid*self.nt
        xi = X_star[:,0].reshape(1,data_grid_len)
        ti = X_star[:,1].reshape(1,data_grid_len)
        if self.system == "transport":
            ui = Transport_eq(self.beta).solution(xi, ti)
        elif self.system == "reaction_diffusion":
            xi = jnp.arange(self.x_min, self.x_max, self.x_max/self.xgrid)
            ti = jnp.linspace(self.t_min, self.t_max, self.nt).reshape(-1, 1)
            ui = Reaction_Diffusion(self.nu, self.rho).solution(xi, ti).reshape(1, data_grid_len)
        elif self.system == "reaction":
            reaction = Reaction(self.rho)
            ui = reaction.solution(reaction.u0(xi[:,:self.xgrid][0]), ti.reshape(self.nt, self.xgrid)).reshape(1,data_grid_len)
        return X_star, ui

