import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd
import pandas as pd
from flax.core.frozen_dict import unfreeze



class NewAugLag:
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
        return jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
    

    def IC_cons(self, params):
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        return Transport_eq(beta=self.beta).solution(\
            self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta
    
    
    def pde_cons(self, params):
        grad_x = jacfwd(self.model.u_theta, 1)(params, self.sample_data)
        return Transport_eq(beta=self.beta).pde(jnp.diag(grad_x[:,:,0]),\
            jnp.diag(grad_x[:,:,1]))
    

    def eq_cons(self, params):
        return jnp.concatenate([self.IC_cons(params), self.pde_cons(params)])
        

    def loss(self, params, mul, penalty_param, alpha):
        aug_part = self.l_k(params) + self.eq_cons(params) @ mul + \
                + penalty_param * jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))
        grads_fx = pd.DataFrame.from_dict(unfreeze(jacfwd(self.l_k, 0)(params)["params"]))
        grads_eq_cons = pd.DataFrame.from_dict(unfreeze(\
            jacfwd(self.eq_cons, 0)(params)["params"]))
        Mx = alpha * grads_eq_cons
        Axgx = lambda x, y: (x * y).sum(axis=(1,2)) if y.ndim == 3 else (x * y).sum(axis=1)
        pen_part = jnp.square(jnp.linalg.norm(jnp.array(list(map(Axgx, \
                    (grads_fx + grads_eq_cons.applymap(lambda x: (x.T @ mul).T)).values.flatten(), \
                        Mx.values.flatten()))).sum(axis=0), ord=2))
        return aug_part + pen_part




