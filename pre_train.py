import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from System import Transport_eq, Reaction_Diffusion, Reaction

from jax import numpy as jnp
from jax import jacfwd, hessian
import jaxopt
import numpy as np
import jax

class PreTrain:
    def __init__(self, model, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, beta, eval_data, eval_ui, pretrain_gtol, pretrain_ftol, pretrain_maxiter, nu, rho, alpha, system, num_params, params):
        self.model = model
        self.beta = beta
        self.pde_sample_data = pde_sample_data
        self.IC_sample_data = IC_sample_data
        self.IC_sample_data_sol = IC_sample_data_sol
        self.BC_sample_data_zero = BC_sample_data_zero
        self.BC_sample_data_2pi = BC_sample_data_2pi
        self.pretrain_loss_list = []
        self.absolute_error_pretrain_list = []
        self.l2_relative_error_pretrain_list = []
        self.eval_data = eval_data
        self.eval_ui = eval_ui
        self.nu = nu
        self.rho = rho
        self.alpha = alpha
        self.pretrain_gtol = pretrain_gtol
        self.pretrain_ftol = pretrain_ftol
        self.pretrain_maxiter = pretrain_maxiter
        self.system = system
        self.stop_optimization = False
        self.params_list = [np.zeros(num_params)]

        shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
        self.shapes, self.sizes = zip(*shapes_and_sizes)
        self.indices = jnp.cumsum(jnp.array(self.sizes)[:-1])


    def l_k(self, params):
        u_theta = self.model.u_theta(params=params, data=self.data)
        return 1 / self.N * jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))


    def IC_cons(self, params):
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        if self.system == "transport":
            return Transport_eq(beta=self.beta).solution(\
                self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta
        elif self.system == "reaction_diffusion":
            return self.IC_sample_data_sol - u_theta
        elif self.system == "reaction":
            return Reaction(self.rho).u0(self.IC_sample_data[:,0]) - u_theta
    
    
    def BC_cons(self, params):
        u_theta_2pi = self.model.u_theta(params=params, data=self.BC_sample_data_2pi)
        u_theta_0 = self.model.u_theta(params=params, data=self.BC_sample_data_zero)
        return u_theta_2pi - u_theta_0
    
    
    def pde_cons(self, params):
        if self.system == "transport":
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
    

    def eq_cons(self, params):
        return jnp.concatenate([self.IC_cons(params), self.BC_cons(params), self.pde_cons(params)])
    

    def loss(self, params):
        return jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))
    

    def flatten_params(self, params):
        flat_params_list, treedef = jax.tree_util.tree_flatten(params)
        return np.concatenate([param.ravel() for param in flat_params_list], axis=0), treedef
    

    def unflatten_params(self, param_list, treedef):
        param_groups = jnp.split(param_list, self.indices)
        reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, self.shapes)]
        return jax.tree_util.tree_unflatten(treedef, reshaped_params)


    def callback_func(self, params):
        ''' Recording optimization information and termination condition '''
        self.pretrain_loss_list.append(self.loss(params).item())
        u_theta = self.model.u_theta(params=params, data=self.eval_data)
        self.absolute_error_pretrain_list.append(jnp.mean(np.abs(u_theta-self.eval_ui)))
        self.l2_relative_error_pretrain_list.append(jnp.linalg.norm((u_theta-self.eval_ui), ord = 2) / jnp.linalg.norm((self.eval_ui), ord = 2))
        list_params, treedef = self.flatten_params(params)
        self.params_list.append(list_params)
        params_diff = self.params_list[-1] - self.params_list[-2]
        if jnp.linalg.norm(params_diff, ord=2) <= self.pretrain_ftol:
            self.stop_optimization = True
            raise TerminationCondition("Stopping criterion met")



    def evaluation(self, params, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = jnp.mean(np.abs(u_theta-ui))
        l2_relative_error = jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2)
        return absolute_error, l2_relative_error, u_theta
    

    def update(self, params, pretrain_maxiter, pretrain_gtol):
        ''' LBFGS for pretraining process '''
        LBFGS_opt = jaxopt.ScipyMinimize(method='L-BFGS-B', \
                                fun=self.loss, \
                                maxiter=pretrain_maxiter, \
                                options={'gtol': pretrain_gtol, 'ftol': 0}, \
                                callback=self.callback_func)

        try:
            params, _ = LBFGS_opt.run(params)
        except TerminationCondition as e:
            print(str(e))

        return params
    

    def lbfgs(self, params):
        solver = jaxopt.LBFGS(fun=self.loss, maxiter=200000, tol = 1e-9)
        res = solver.run(params)
        optimized_params, solver_state = res
        return optimized_params

class TerminationCondition(Exception):
    pass




