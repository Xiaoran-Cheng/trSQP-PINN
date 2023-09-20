import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from System import Transport_eq, Reaction_Diffusion, Reaction, Burger

from jax import numpy as jnp
from jax import jacfwd, hessian



class l2Penalty:
    def __init__(self, model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui, beta, N, nu, rho, alpha, system):
        self.model = model
        self.beta = beta
        self.data = data
        self.pde_sample_data = pde_sample_data
        self.IC_sample_data = IC_sample_data
        self.IC_sample_data_sol = IC_sample_data_sol
        self.BC_sample_data_zero = BC_sample_data_zero
        self.BC_sample_data_2pi = BC_sample_data_2pi
        self.ui = ui
        self.N = N
        self.nu = nu
        self.rho = rho
        self.alpha = alpha
        self.system = system


    def l_k(self, params):
        u_theta = self.model.u_theta(params=params, data=self.data)
        return 1 / self.N * jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
    

    def IC_cons(self, params):
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        if self.system == "convection":
            return Transport_eq(beta=self.beta).solution(\
                self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta
        elif self.system == "reaction_diffusion":
            # return Reaction_Diffusion(self.nu, self.rho).u0(self.IC_sample_data[:,0]) - u_theta
            return self.IC_sample_data_sol - u_theta
        elif self.system == "reaction":
            return Reaction(self.rho).u0(self.IC_sample_data[:,0]) - u_theta
        elif self.system == "burger":
            return Burger(self.alpha).u0(self.IC_sample_data[:,0]) - u_theta
    
    
    def BC_cons(self, params):
        u_theta_2pi = self.model.u_theta(params=params, data=self.BC_sample_data_2pi)
        u_theta_0 = self.model.u_theta(params=params, data=self.BC_sample_data_zero)
        return u_theta_2pi - u_theta_0
    
    
    def pde_cons(self, params):
        if self.system == "convection":
            grad_x = jacfwd(self.model.u_theta, 1)(params, self.pde_sample_data)
            return Transport_eq(beta=self.beta).pde(jnp.diag(grad_x[:,:,0]),\
                jnp.diag(grad_x[:,:,1]))
        elif self.system == "reaction_diffusion":
            u_theta = self.model.u_theta(params=params, data=self.pde_sample_data)
            grad_x = jacfwd(self.model.u_theta, 1)(params, self.pde_sample_data)
            dudt = jnp.diag(grad_x[:,:,1])
            grad_xx = hessian(self.model.u_theta, 1)(params, self.pde_sample_data)
            du2dx2 = jnp.diag(jnp.diagonal(grad_xx[:, :, 0, :, 0], axis1=1, axis2=2))
            return Reaction_Diffusion(self.nu, self.rho).pde(dudt, du2dx2, u_theta)
        elif self.system == "reaction":
            u_theta = self.model.u_theta(params=params, data=self.pde_sample_data)
            grad_x = jacfwd(self.model.u_theta, 1)(params, self.pde_sample_data)
            dudt = jnp.diag(grad_x[:,:,1])
            return Reaction(self.rho).pde(dudt, u_theta)
        elif self.system == "burger":
            u_theta = self.model.u_theta(params=params, data=self.pde_sample_data)
            grad_x = jacfwd(self.model.u_theta, 1)(params, self.pde_sample_data)
            dudt = jnp.diag(grad_x[:,:,1])
            dudx = jnp.diag(grad_x[:,:,0])
            grad_xx = hessian(self.model.u_theta, 1)(params, self.pde_sample_data)
            du2dx2 = jnp.diag(jnp.diagonal(grad_xx[:, :, 0, :, 0], axis1=1, axis2=2))
            return Burger(self.alpha).pde(dudt, dudx, du2dx2, u_theta)
    
    
    def eq_cons(self, params):
        return jnp.concatenate([self.IC_cons(params), self.BC_cons(params), self.pde_cons(params)])
    

    def eq_cons_loss(self, params):
        return jnp.linalg.norm(self.eq_cons(params), ord=2)


    def loss(self, params, penalty_param):
        return  self.l_k(params=params) + 0.5 * penalty_param * self.eq_cons_loss(params)


