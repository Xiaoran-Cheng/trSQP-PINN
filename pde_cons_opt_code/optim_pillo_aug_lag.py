import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd
import jax
from jaxopt._src import tree_util


class PilloAugLag:
    def __init__(self, model, data, pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi, ui, beta, N):
        self.model = model
        self.beta = beta
        self.data = data
        self.pde_sample_data = pde_sample_data
        self.IC_sample_data = IC_sample_data
        self.BC_sample_data_zero = BC_sample_data_zero
        self.BC_sample_data_2pi = BC_sample_data_2pi
        self.ui = ui
        self.N = N


    def l_k(self, params):
        u_theta = self.model.u_theta(params=params, data=self.data)
        return 1 / self.N * jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
    

    def IC_cons(self, params):
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        return Transport_eq(beta=self.beta).solution(\
            self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta
    
    
    def BC_cons(self, params):
        # u_theta_2pi = self.model.u_theta(params=params, data=self.BC_sample_data_2pi)
        u_theta_0 = self.model.u_theta(params=params, data=self.BC_sample_data_2pi)
        return Transport_eq(beta=self.beta).solution(\
            self.BC_sample_data_2pi[:,0], self.BC_sample_data_2pi[:,1]) - u_theta_0
        # return jnp.concatenate([Transport_eq(beta=self.beta).solution(\
        #     self.BC_sample_data_zero[:,0], self.BC_sample_data_zero[:,1]) - u_theta_0, Transport_eq(beta=self.beta).solution(\
        #     self.BC_sample_data_2pi[:,0], self.BC_sample_data_2pi[:,1]) - u_theta_2pi])
    
    
    def pde_cons(self, params):
        grad_x = jacfwd(self.model.u_theta, 1)(params, self.pde_sample_data)
        return Transport_eq(beta=self.beta).pde(jnp.diag(grad_x[:,:,0]),\
            jnp.diag(grad_x[:,:,1]))
    

    def eq_cons(self, params):
        return jnp.concatenate([self.IC_cons(params), self.BC_cons(params), self.pde_cons(params)])
    

    def eq_cons_loss(self, params):
        return jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))


    def L(self, params, mul):
        return self.l_k(params) + self.eq_cons(params) @ mul


    def loss(self, params_mul, penalty_param_mu, penalty_param_v):
        params = params_mul['params']
        mul = params_mul['mul']
        opt_error_penalty = jnp.square(tree_util.tree_l2_norm(jacfwd(self.L, 0)(params, mul)['params']))
        return self.L(params, mul) + 0.5 * penalty_param_mu * self.eq_cons_loss(params) + 0.5 * penalty_param_v * opt_error_penalty
    

