import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from System import Transport_eq, Reaction_Diffusion, Reaction, Burger

from jax import numpy as jnp
from jax import jacfwd, hessian
import numpy as np
from scipy.optimize import minimize
import jax
import time
import jaxlib.xla_extension as xla
import pandas as pd


class SQP_Optim:
    def __init__(self, model, params, beta, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui, N, eval_data, eval_ui, nu, rho, alpha, system, intermediate_data_frame_path) -> None:
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
        shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
        self.shapes, self.sizes = zip(*shapes_and_sizes)
        self.indices = jnp.cumsum(jnp.array(self.sizes)[:-1])
        self.eval_data = eval_data
        self.eval_ui = eval_ui
        self.time_iter = []
        self.nu = nu
        self.rho = rho
        self.alpha = alpha
        self.system = system
        self.start_time = time.time()
        self.intermediate_data_frame_path = intermediate_data_frame_path
        

    def obj(self, param_list, treedef, loss_values):
        params = self.unflatten_params(param_list, treedef)
        u_theta = self.model.u_theta(params=params, data=self.data)
        obj_value = 1 / self.N * jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
        len_loss_values = len([i.item() for i in loss_values if isinstance(i, xla.ArrayImpl)])
        loss_values.append(obj_value)
        if len([i.item() for i in loss_values if isinstance(i, xla.ArrayImpl)]) - len_loss_values > 0:
            self.time_iter.append(time.time() - self.start_time)
            try:
                param_df = param_list.primal
            except:
                param_df = param_list
            try:
                pd.DataFrame(param_df, columns=['params']).\
                    to_csv(self.intermediate_data_frame_path+"param_{index}.csv".\
                           format(index=len_loss_values), index=False)
            except:
              pass
        return obj_value


    def grad_objective(self, param_list, treedef, loss_values):
        return jacfwd(self.obj, 0)(param_list, treedef, loss_values)


    def IC_cons(self, param_list, treedef):
        params = self.unflatten_params(param_list, treedef)
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        if self.system == "convection":
            return Transport_eq(beta=self.beta).solution(\
                self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta
        elif self.system == "reaction_diffusion":
            return self.IC_sample_data_sol - u_theta
        elif self.system == "reaction":
            return Reaction(self.rho).u0(self.IC_sample_data[:,0]) - u_theta
        elif self.system == "burger":
            return Burger(self.alpha).u0(self.IC_sample_data[:,0]) - u_theta


    def BC_cons(self, param_list, treedef):
        params = self.unflatten_params(param_list, treedef)
        u_theta_0 = self.model.u_theta(params=params, data=self.BC_sample_data_zero)
        u_theta_2pi = self.model.u_theta(params=params, data=self.BC_sample_data_2pi)
        return u_theta_2pi - u_theta_0
    
    
    def pde_cons(self, param_list, treedef):
        params = self.unflatten_params(param_list, treedef)
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
    

    
    def eq_cons(self, param_list, treedef, eq_cons_loss_values, loss_values, kkt_residual):
        eq_cons = jnp.concatenate([self.IC_cons(param_list, treedef), self.BC_cons(param_list, treedef), self.pde_cons(param_list, treedef)])
        eq_cons_loss = jnp.square(jnp.linalg.norm(eq_cons, ord=2))
        eq_cons_loss_values.append(eq_cons_loss)
        return eq_cons


    def grads_eq_cons(self, param_list, treedef, eq_cons_loss_values, loss_values, kkt_residual):
        eq_cons_jac = jacfwd(self.eq_cons, 0)(param_list, treedef, eq_cons_loss_values, loss_values, kkt_residual)
        cond_num = jnp.linalg.cond(eq_cons_jac)
        print("condition number: ", str(cond_num))
        lambdas = (jnp.linalg.inv(eq_cons_jac @ eq_cons_jac.T) @ eq_cons_jac) @ self.grad_objective(param_list, treedef, loss_values)
        L = lambda param_list: self.obj(param_list, treedef, loss_values) - lambdas @ self.eq_cons(param_list, treedef, eq_cons_loss_values, loss_values, kkt_residual)
        kkt_residual.append(jnp.linalg.norm(jacfwd(L, 0)(param_list), ord=jnp.inf))
        return eq_cons_jac


    def flatten_params(self, params):
        flat_params_list, treedef = jax.tree_util.tree_flatten(params)
        return np.concatenate([param.ravel( ) for param in flat_params_list], axis=0), treedef


    def unflatten_params(self, param_list, treedef):
        param_groups = jnp.split(param_list, self.indices)
        reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, self.shapes)]
        return jax.tree_util.tree_unflatten(treedef, reshaped_params)


    def SQP_optim(self, params, loss_values, eq_cons_loss_values, kkt_residual, maxiter, sqp_hessian, sqp_gtol, sqp_xtol, sqp_initial_constr_penalty, sqp_initial_tr_radius):
        flat_params, treedef = self.flatten_params(params)
        constraints = {
            'type': 'eq',
            'fun': self.eq_cons,
            'jac': self.grads_eq_cons,
            'args': (treedef, eq_cons_loss_values, loss_values, kkt_residual)}
        solution = minimize(self.obj, \
                            flat_params, \
                            args=(treedef,loss_values), \
                            jac=self.grad_objective, \
                            method='trust-constr', \
                            options={'maxiter': maxiter, \
                                    'gtol': sqp_gtol, \
                                    'xtol': sqp_xtol, \
                                    'initial_tr_radius': sqp_initial_tr_radius, \
                                    'initial_constr_penalty': sqp_initial_constr_penalty, \
                                    'verbose': 3}, \
                            constraints=constraints, \
                            hess = sqp_hessian)
        params_opt = self.unflatten_params(solution.x, treedef)
        print(solution)
        return params_opt


    def evaluation(self, params):
        u_theta = self.model.u_theta(params=params, data=self.eval_data)
        absolute_error = jnp.mean(jnp.abs(u_theta-self.eval_ui))
        l2_relative_error = jnp.power(jnp.power((u_theta-self.eval_ui), 2).sum(), 1/2) / jnp.power(jnp.power((self.eval_ui), 2).sum(), 1/2)
        return absolute_error, l2_relative_error, u_theta
 


