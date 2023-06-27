import jax.numpy as jnp
from Transport_eq import Transport_eq
from jax import random

class Data:
    def __init__(self, N:int, M:int, key_num:int, dim:int) -> None:
        self.N = N
        self.M = M
        self.key_num = key_num
        self.dim = dim
    

    def generate_data(self, x_min, x_max, t_min, t_max, beta):
        key = random.PRNGKey(self.key_num)
        xi = random.uniform(key, shape=(self.dim-1,self.N), minval=x_min, maxval=x_max)
        ti = random.uniform(key, shape=(self.dim-1,self.N), minval=t_min, maxval=t_max)
        ui = Transport_eq(beta=beta).solution(xi, ti)
        return xi, ti, ui
    

    def sample_data(self, x_min, x_max, t_min, t_max):
        key = random.PRNGKey(self.key_num)
        xj = random.uniform(key, shape=(self.dim-1,self.M), minval=x_min, maxval=x_max)
        tj = random.uniform(key, shape=(self.dim-1,self.M), minval=t_min, maxval=t_max)
        return xj, tj
    

    def generate_IC_data(self, min, max, beta):
        key = random.PRNGKey(self.key_num)
        IC_data = random.uniform(key, shape=(self.dim-1,self.N), minval=min, maxval=max)
        xi = IC_data
        ti = jnp.zeros(self.N)
        ui = Transport_eq(beta=beta).solution(xi, ti)
        return xi, ti, ui
    

    def generate_boundary_data(self, min, max, beta):
        key = random.PRNGKey(self.key_num)
        boundary_data = random.uniform(key, shape=(self.dim-1,self.N), minval=min, maxval=max)
        ti = boundary_data
        xi = jnp.zeros(self.N)
        ui = Transport_eq(beta=beta).solution(2*jnp.pi, ti)
        return xi, ti, ui

