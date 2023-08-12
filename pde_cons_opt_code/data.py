import jax.numpy as jnp
from Transport_eq import Transport_eq
from jax import random
import numpy as np



class Data:
    def __init__(self, N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level) -> None:
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
        ui = Transport_eq(beta=self.beta).solution(xi, ti) + random.uniform(random.PRNGKey(key_num), \
                                                shape=(1,self.N), minval=-self.noise_level, maxval=self.noise_level)
        # ui = Transport_eq(beta=self.beta).solution(xi, ti)
        data = jnp.concatenate((xi.T, ti.T), axis=1)
        return data, ui


    # def sample_data(self, pde_key_num, IC_key_num, BC_key_num):
    #     X_star, X_star_noBC_noIC = self.data_grid()
    #     pde_sample_data = X_star_noBC_noIC[random.choice(random.PRNGKey(pde_key_num), shape=(self.pde_M,), a=X_star_noBC_noIC.shape[0], replace=False),:]
    #     IC_sample_data = X_star[X_star[:, 1] == 0][random.choice(random.PRNGKey(IC_key_num), shape=(self.IC_M,), a=self.xgrid, replace=False),:]
    #     BC_sample_data_zero = X_star[X_star[:, 0] == 0][random.choice(random.PRNGKey(BC_key_num), shape=(self.BC_M,), a=self.nt, replace=False),:]
    #     BC_sample_data_2pi = BC_sample_data_zero.copy()
    #     BC_sample_data_2pi = BC_sample_data_2pi.at[:, 0].set(2 * jnp.pi)
    #     return pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi


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
        ui = Transport_eq(beta=self.beta).solution(xi, ti)
        return X_star, ui


beta = 30
xgrid = 256
nt = 100
N=1000
IC_M, pde_M, BC_M = 70,70,70
M = IC_M + pde_M + BC_M
data_key_num, sample_key_num = 100,256
x_min = 0
x_max = 2*jnp.pi
t_min = 0
t_max = 1
noise_level = 0.001
Datas = Data(N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level)
data, ui = Datas.generate_data(data_key_num)
pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num)

