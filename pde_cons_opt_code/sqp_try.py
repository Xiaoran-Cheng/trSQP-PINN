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
import time
import jax


class SQP_Optim:
    def __init__(self, model, qp, feature, group_labels, hessian_param, M, params, beta, data, sample_data, IC_sample_data, ui, N, merit_func_penalty_param) -> None:
        self.model = model
        self.qp = qp
        self.feature = feature
        self.group_labels = group_labels
        self.hessian_param = hessian_param
        self.M = M
        self.layer_names = params["params"].keys()
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.ui = ui
        self.N = N
        self.merit_func_penalty_param = merit_func_penalty_param 


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
        # return jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))
        return jnp.linalg.norm(self.eq_cons(params), ord=1)


    def L(self, params, mul):
        return self.obj(params) + self.eq_cons(params) @ mul


    # def make_psd(self, mat):
    #     eigenvalues = jnp.linalg.eigvalsh(mat)
    #     min_eigenvalue = jnp.min(eigenvalues)
    #     if min_eigenvalue < 0.0:
    #         mat_psd = mat - 2 * min_eigenvalue * jnp.eye(mat.shape[0])
    #         return mat_psd
    #     else:
    #         return mat


    def is_psd(self, mat):
        eigenvalues = jnp.linalg.eigvalsh(mat)
        return jnp.all(eigenvalues > 0)
    

    def bfgs_hessian(self, paramsk, paramsk1, mulk1, step_size, p, Hk):
        mulk1 = mulk1.reshape(-1,1)
        p = p.reshape(-1,1)
        sk = step_size * p
        yk = self.flat_single_dict(jacfwd(self.L, 0)(paramsk1, mulk1)) - self.flat_single_dict(jacfwd(self.L, 0)(paramsk, mulk1))
        yk = yk.reshape(-1,1)
        skyk = sk.T @ yk
        skHksk = sk.T @ Hk @ sk
        if skyk >= 0.2 * skHksk:
            thetak = 1
        else:
            thetak = (0.8 * skHksk) / (skHksk - skyk)
        rk = thetak * yk + (1 - thetak) * Hk @ sk
        Hk1 = Hk - ((Hk @ sk @ sk.T @ Hk) / skHksk) + ((rk @ rk.T) / (sk.T @ rk))
        Hk1 = 0.5 * (Hk1 + Hk1.T)
        if self.is_psd(Hk1):
            return Hk1
        else:
            return Hk


    # def flat_single_dict(self, dicts):
    #     flat_params_list = pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
    #                     applymap(lambda x: x.flatten()).values.flatten()
    #     return jnp.concatenate([param.ravel() for param in flat_params_list])


    # def flat_multi_dict(self, dicts):
    #     flat_params_list = pd.DataFrame.from_dict(\
    #             unfreeze(dicts['params'])).\
    #                 apply(lambda x: x.explode()).set_index([self.group_labels]).\
    #                     sort_index().applymap(lambda x: x.flatten()).values.flatten()
    #     return jnp.concatenate([param.ravel() for param in flat_params_list])


    def flat_single_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
                        applymap(lambda x: x.flatten()).values.flatten())
    

    def flat_multi_dict(self, dicts, group_labels):
        return np.concatenate(pd.DataFrame.from_dict(\
                unfreeze(dicts['params'])).\
                    apply(lambda x: x.explode()).set_index([group_labels]).\
                        sort_index().applymap(lambda x: x.flatten()).values.flatten())


    def merit_func(self, params):
        return self.obj(params=params) + 0.5 * self.merit_func_penalty_param * self.eq_cons_loss(params)
        # flatted_gra_l_k = self.flat_single_dict(jacfwd(self.obj, 0)(params))
        # flatted_gra_eq_cons = jnp.array(jnp.split(self.flat_multi_dict(jacfwd(self.eq_cons, 0)(params)), 2*self.M))
        # lambdax = jnp.linalg.pinv(flatted_gra_eq_cons @ flatted_gra_eq_cons.T) @ flatted_gra_eq_cons @ flatted_gra_l_k
        # return self.obj(params) - lambdax.T @ self.eq_cons(params) + 0.5 * self.merit_func_penalty_param * self.eq_cons_loss(params)

    
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


    def SQP_optim(self, params, num_iter, maxiter, condition, decrease_factor, init_stepsize, line_search_tol, qr_ind_tol, init_mul):
        obj_list = []
        eq_con_list = []
        kkt_residual_list = []
        shapes = pd.DataFrame.from_dict(unfreeze(params["params"])).applymap(lambda x: x.shape).values.flatten()
        sizes = [np.prod(shape) for shape in shapes]
        updated_Hk = self.hessian_param * jnp.identity(sum(sizes))
        mulk = init_mul
        for _ in tqdm(range(num_iter)):
            gra_obj = jacfwd(self.obj, 0)(params)
            gra_eq_cons = jacfwd(self.eq_cons, 0)(params)
            eq_cons = self.eq_cons(params=params)
            flatted_gra_obj = self.flat_single_dict(gra_obj)
            flatted_current_params = self.flat_single_dict(params)
            flatted_gra_eq_cons = self.flat_multi_dict(gra_eq_cons, self.group_labels)

            c = flatted_gra_obj
            A = jnp.array(jnp.split(flatted_gra_eq_cons, 2*self.M))
            li_ind_index = self.get_li_in_cons_index(A, qr_ind_tol)
            A = A[li_ind_index, :]
            print(jnp.linalg.cond(A))
            b = -eq_cons[li_ind_index]
            li_d_index = jnp.sort(jnp.setdiff1d(jnp.arange(2*self.M), li_ind_index))

            try:
                Q = updated_Hk
                sol = self.qp.run(init_params=params, params_obj=(Q, c), params_eq=(A, b), params_ineq=None)
            except:
                Q = self.hessian_param * jnp.identity(sum(sizes))
                sol = self.qp.run(init_params=params, params_obj=(Q, c), params_eq=(A, b), params_ineq=None)
            else:
                Hk = updated_Hk

            flatted_delta_params = sol.params.primal
            kkt_residual = self.qp.l2_optimality_error(params=sol.params, params_obj=(Q, c), params_eq=(A, b), params_ineq=None)
            delta_params = self.get_recovered_dict(flatted_delta_params, shapes, sizes)









            # ls = BacktrackingLineSearch(fun=self.merit_func, maxiter=maxiter, condition=condition,
            #                             decrease_factor=decrease_factor, tol=line_search_tol)
            # stepsize, _ = ls.run(init_stepsize=init_stepsize, \
            #                      params=params,
            #                     descent_direction=delta_params)
            stepsize = 0.5
            # print(stepsize, flatted_delta_params.sum())
            flatted_updated_params = stepsize * flatted_delta_params + flatted_current_params
            updated_params = self.get_recovered_dict(flatted_updated_params, shapes, sizes)
            mul_lambda = sol.params.dual_eq
            mulk1 = mulk + stepsize * (mul_lambda - mulk)
            
            if len(li_d_index) != 0:
                for i, index in enumerate(li_d_index):
                    if i != 0:
                        mulk1 = jnp.insert(mulk1, index+1, 1.0)
                    else:
                        mulk1 = jnp.insert(mulk1, index, 1.0)


            # updated_Hk = self.bfgs_hessian(params, updated_params, mulk1, stepsize, flatted_delta_params, Hk)
            params = updated_params
            mulk = mulk1
            obj_list.append(self.obj(params))
            eq_con_list.append(self.eq_cons_loss(params))
            kkt_residual_list.append(kkt_residual)



        # pd.DataFrame(self.flat_single_dict(params)).to_csv("ddd.csv", index = False)
        
        return params, obj_list, eq_con_list, kkt_residual_list
        

    def evaluation(self, params, N, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = 1/N * jnp.linalg.norm(u_theta-ui, ord = 2)
        l2_relative_error = 1/N * (jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2))
        return absolute_error, l2_relative_error, u_theta
 






