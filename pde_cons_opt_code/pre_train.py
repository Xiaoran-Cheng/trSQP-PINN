import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd
import jaxopt
import numpy as np



class PreTrain:
    def __init__(self, model, pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi, beta, pretrain_loss_list):
        self.model = model
        self.beta = beta
        self.pde_sample_data = pde_sample_data
        self.IC_sample_data = IC_sample_data
        self.BC_sample_data_zero = BC_sample_data_zero
        self.BC_sample_data_2pi = BC_sample_data_2pi
        self.pretrain_loss_list = pretrain_loss_list


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
    

    def loss(self, params):
        return jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))
    

    def callback_func(self, params):
        self.pretrain_loss_list.append(self.loss(params).item())


    def evaluation(self, params, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = jnp.mean(np.abs(u_theta-ui))
        l2_relative_error = jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2)
        return absolute_error, l2_relative_error, u_theta
    

    def update(self, params, pretrain_maxiter, pretrain_gtol, pretrain_ftol):
        LBFGS_opt = jaxopt.ScipyMinimize(method='L-BFGS-B', \
                                fun=self.loss, \
                                maxiter=pretrain_maxiter, \
                                options={'gtol': pretrain_gtol, 'ftol': pretrain_ftol}, \
                                callback=self.callback_func)
        params, _ = LBFGS_opt.run(params)
        return params


