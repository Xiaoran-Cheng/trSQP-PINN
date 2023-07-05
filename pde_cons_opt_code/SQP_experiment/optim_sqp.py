import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd
# from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
import pandas as pd
from flax.core.frozen_dict import FrozenDict, unfreeze
from jaxopt import BacktrackingLineSearch, HagerZhangLineSearch
from functools import partial
from jax.interpreters.xla import DeviceArray



class OptimComponents:
    def __init__(self, model, data, sample_data, IC_sample_data, ui, beta, N, M, group_labels):
        self.model = model
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.ui = ui
        self.N = N
        self.M = M
        self.group_labels = group_labels


    def obj(self, params):
        u_theta = self.model.u_theta(params=params, data=self.data)
        return 1 / self.N * jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
    

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
    

    def eq_cons_loss(self, params):
        return jnp.linalg.norm(self.eq_cons(params), ord=1)


    def L(self, params, mul):
        return self.obj(params) + self.eq_cons(params) @ mul
    

    def flat_single_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
                        applymap(lambda x: x.flatten()).values.flatten())
    

    def flat_multi_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(\
                unfreeze(dicts['params'])).\
                    apply(lambda x: x.explode()).set_index([self.group_labels]).\
                        sort_index().applymap(lambda x: x.flatten()).values.flatten())


    # def merit_func(self, params, mul, penalty_param_mu, penalty_param_v):
    #     grads_L = self.flat_single_dict(jacfwd(self.L, 0)(params, mul))
    #     flatted_gra_eq_cons = jnp.array(jnp.split(self.flat_multi_dict(jacfwd(self.eq_cons, 0)(params)), 2*self.M))
    #     second_penalty_part = jnp.square(jnp.linalg.norm(flatted_gra_eq_cons @ grads_L, ord=2))
    #     return self.L(params, mul) + 0.5 * penalty_param_mu * self.eq_cons_loss(params) + 0.5 * penalty_param_v * second_penalty_part
    

    def merit_func(self, params, merit_func_penalty_param):
        return self.obj(params=params) + 1 / (2*self.M) * merit_func_penalty_param *  self.eq_cons_loss(params)


    def get_grads(self, params):
        gra_obj = jacfwd(self.obj, 0)(params)
        gra_eq_cons = jacfwd(self.eq_cons, 0)(params)
        # Hlag = hessian(self.lag, 0)(params, mul)
        return gra_obj, gra_eq_cons
    

        
class SQP_Optim:
    def __init__(self, model, optim_components, qp, feature, group_labels, hessian_param, M, params) -> None:
        self.model = model
        self.optim_components = optim_components
        self.qp = qp
        self.feature = feature
        self.group_labels = group_labels
        self.hessian_param = hessian_param
        self.M = M
        self.layer_names = params["params"].keys()
    

    def get_li_in_cons_index(self, mat, qr_ind_tol):
        _, R = jnp.linalg.qr(mat)
        independent = jnp.where(jnp.abs(R.diagonal()) > qr_ind_tol)[0]
        return independent


    def get_recovered_dict(self, flatted_target, shapes, sizes):
            subarrays = np.split(flatted_target, np.cumsum(sizes)[:-1])
            reshaped_arrays = [subarray.reshape(shape) for subarray, shape in zip(subarrays, shapes)]
            flatted_target_df = pd.DataFrame(np.array(reshaped_arrays, dtype=object).\
                        reshape(2,len(self.feature))).applymap(lambda x: x)
            flatted_target_df.columns = self.layer_names
            flatted_target_df.index = ["bias", "kernel"]
            flatted_target_df.sort_index(ascending=False, inplace=True)
            recovered_target = FrozenDict({"params": flatted_target_df.to_dict()})
            return recovered_target


    # def penalty_updating_condition(self, params, mul, delta_params, delta_mul, penalty_param_mu, penalty_param_v):
    #     dicts = jacfwd(self.optim_components.obj, 0)(params)
    #     dd = pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
    #                     applymap(lambda x: x.flatten(x)).values.flatten()
    #     return dd
        # grads_params_L = jacfwd(self.optim_components.merit_func, 0)(params, mul, penalty_param_mu, penalty_param_v)
        # grads_params_mul = jacfwd(self.optim_components.merit_func, 1)(params, mul, penalty_param_mu, penalty_param_v)
        # return self.optim_components.merit_func(params, mul, penalty_param_mu, penalty_param_v)


    def SQP_optim(self, params, num_iter, maxiter, condition, decrease_factor, init_stepsize, line_search_tol, qr_ind_tol, merit_func_penalty_param):
        obj_list = []
        eq_con_list = []
        kkt_residual_list = []
        shapes = pd.DataFrame.from_dict(unfreeze(params["params"])).applymap(lambda x: x.shape).values.flatten()
        sizes = [np.prod(shape) for shape in shapes]
        for _ in tqdm(range(num_iter)):
            gra_obj, gra_eq_cons = self.optim_components.get_grads(params=params)
            eq_cons = self.optim_components.eq_cons(params=params)
            current_obj = self.optim_components.obj(params=params)
            flatted_gra_obj = self.optim_components.flat_single_dict(gra_obj)
            flatted_current_params = self.optim_components.flat_single_dict(params)
            flatted_gra_eq_cons = self.optim_components.flat_multi_dict(gra_eq_cons)
            Q = self.hessian_param * jnp.identity(flatted_gra_obj.shape[0])
            c = flatted_gra_obj
            A = jnp.array(jnp.split(flatted_gra_eq_cons, 2*self.M))
            li_ind_index = self.get_li_in_cons_index(A, qr_ind_tol)
            A = A[li_ind_index, :]
            b = -eq_cons[li_ind_index]
            sol = self.qp.run(init_params=params, params_obj=(Q, c), params_eq=(A, b), params_ineq=None).params
            flatted_delta_params = sol.primal
            kkt_residual = self.qp.l2_optimality_error(params=sol, params_obj=(Q, c), params_eq=(A, b), params_ineq=None)
            delta_params = self.get_recovered_dict(flatted_delta_params, shapes, sizes)
            partial_optim_components_merit_func = partial(self.optim_components.merit_func, merit_func_penalty_param=merit_func_penalty_param)
            ls = BacktrackingLineSearch(fun=partial_optim_components_merit_func, maxiter=maxiter, condition=condition,
                                        decrease_factor=decrease_factor, tol=line_search_tol)
            stepsize, _ = ls.run(init_stepsize=init_stepsize, params=params,
                                    descent_direction=delta_params,
                                            value=current_obj, grad=gra_obj)
            
            flatted_updated_params = stepsize * flatted_delta_params + flatted_current_params
            params = self.get_recovered_dict(flatted_updated_params, shapes, sizes)
            obj_list.append(self.optim_components.obj(params))
            eq_con_list.append(self.optim_components.eq_cons_loss(params))
            kkt_residual_list.append(kkt_residual)
        return params, obj_list, eq_con_list, kkt_residual_list
        

    def evaluation(self, params, N, data, ui):
        u_theta = self.model.u_theta(params = params, data=data)
        absolute_error = 1/N * jnp.linalg.norm(u_theta-ui, ord = 2)
        l2_relative_error = 1/N * (jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2))
        return absolute_error, l2_relative_error, u_theta
 











