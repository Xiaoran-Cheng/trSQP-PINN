import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd



class linfinityPenalty:
    def __init__(self, model, data, sample_data, IC_sample_data, ui, beta, N, M):
        self.model = model
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.ui = ui
        self.N = N
        self.M = M


    def l_k(self, params):
        u_theta = self.model.u_theta(params=params, data=self.data)
        return jnp.linalg.norm(u_theta - self.ui, ord=jnp.inf)
    

    def IC_cons(self, params):
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        return jnp.linalg.norm(Transport_eq(beta=self.beta).solution(\
        self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta, ord=jnp.inf)
    

    def pde_cons(self, params):
        grad_x = jacfwd(self.model.u_theta, 1)(params, self.sample_data)
        return jnp.linalg.norm(Transport_eq(beta=self.beta).pde(jnp.diag(grad_x[:,:,0]),\
                                         jnp.diag(grad_x[:,:,1])), ord=jnp.inf)
    

    def loss(self, params, IC_cons_penalty_param, pde_cons_penalty_param):
        data_loss = self.l_k(params=params)
        IC_loss = self.IC_cons(params=params)
        pde_loss = self.pde_cons(params=params)
        loss = 1 / self.N * data_loss + \
            1 / self.M * IC_cons_penalty_param * IC_loss + \
            1 / self.M * pde_cons_penalty_param * pde_loss
        return loss    
    


