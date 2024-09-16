import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from jax import numpy as jnp
import numpy as np
from jaxopt.tree_util import tree_l2_norm, tree_zeros_like, tree_add, tree_scalar_mul, tree_add_scalar_mul
from tqdm import tqdm
import jaxopt
import time

class Optim:
    def __init__(self, model, Loss, LBFGS_maxiter, LBFGS_gtol, LBFGS_ftol, total_l_k_loss_list, \
                 total_eq_cons_loss_list, absolute_error_iter, l2_relative_error_iter, total_loss_list, time_iter, data, ui) -> None:
        self.model = model
        self.Loss = Loss
        self.LBFGS_maxiter = LBFGS_maxiter
        self.LBFGS_gtol = LBFGS_gtol
        self.LBFGS_ftol = LBFGS_ftol
        self.total_l_k_loss_list = total_l_k_loss_list
        self.total_eq_cons_loss_list = total_eq_cons_loss_list
        self.absolute_error_iter = absolute_error_iter
        self.l2_relative_error_iter = l2_relative_error_iter
        self.total_loss_list = total_loss_list
        self.data = data
        self.ui = ui
        self.time_iter = time_iter
        self.stop_optimization = False


    # def adam_update(self, opt, grads, optim_object):
    #     opt_state = opt.init(optim_object)
    #     grads, opt_state = opt.update(grads, opt_state)
    #     optim_object = optax.apply_updates(optim_object, grads)
    #     return optim_object


    def update(self, params, \
                    penalty_param, experiment, \
                    mul, LBFGS_opt):
    
        if experiment == "ALM":
            params, _ = LBFGS_opt.run(params, penalty_param=penalty_param, mul=mul)
        else:
            params, _ = LBFGS_opt.run(params, penalty_param=penalty_param)

        print(Exception) # if hits defualt stopping condition, it displays only <class 'Exception'>

        return params, self.Loss.eq_cons(params)
    


    def lbfgs(self, params, \
                    penalty_param, \
                    experiment, \
                    mul, start_time):
        solver = jaxopt.LBFGS(fun=self.Loss.loss, maxiter=self.LBFGS_maxiter)
        state = solver.init_state(params)
        prev_norm = 0
        prev_params = tree_zeros_like(params)
        for _ in tqdm(range(solver.maxiter)):
            if experiment == "ALM":
                params, state = solver.update(params, state, mul, penalty_param)
                self.total_loss_list.append(self.Loss.loss(params, mul, penalty_param))
            else:
                params, state = solver.update(params, state, penalty_param)
                self.total_loss_list.append(self.Loss.loss(params, penalty_param))
            norm = state.error
            norm_diff = jnp.absolute(norm - prev_norm)
            params_diff = tree_l2_norm(tree_add_scalar_mul(params, -1, prev_params))
            self.total_l_k_loss_list.append(self.Loss.l_k(params))
            self.total_eq_cons_loss_list.append(self.Loss.eq_cons_loss(params))
            self.absolute_error_iter.append(self.evaluation(params, self.data, self.ui)[0])
            self.l2_relative_error_iter.append(self.evaluation(params, self.data, self.ui)[1])
            self.time_iter.append(time.time() - start_time)
            

            if params_diff <= self.LBFGS_ftol or norm_diff <= self.LBFGS_gtol:
                break

            prev_params = params
            prev_norm = norm
        return params


    def evaluation(self, params, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = jnp.mean(np.abs(u_theta-ui))
        l2_relative_error = jnp.power(jnp.power((u_theta-ui), 2).sum(), 1/2) / jnp.power(jnp.power((ui), 2).sum(), 1/2)
        return absolute_error, l2_relative_error, u_theta
    



class TerminationCondition(Exception):
    pass







