import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from jax import numpy as jnp
from jax import value_and_grad, jacfwd
import optax
# from tqdm.notebook import tqdm
import pandas as pd
from flax.core.frozen_dict import unfreeze
from tqdm import tqdm
import numpy as np
import jaxopt


class Optim:
    def __init__(self, model, Loss, panalty_param_upper_bound, LBFGS_linesearch, LBFGS_tol, LBFGS_maxiter, LBFGS_history_size) -> None:
        self.model = model
        self.Loss = Loss
        self.panalty_param_upper_bound = panalty_param_upper_bound
        self.LBFGS_linesearch = LBFGS_linesearch
        self.LBFGS_tol = LBFGS_tol
        self.LBFGS_maxiter = LBFGS_maxiter
        self.LBFGS_history_size = LBFGS_history_size


    def adam_update(self, opt, grads, optim_object):
        opt_state = opt.init(optim_object)
        grads, opt_state = opt.update(grads, opt_state)
        optim_object = optax.apply_updates(optim_object, grads)
        return optim_object


    def update(self, params, num_echos, \
                    penalty_param, experiment, \
                    mul, alpha, group_labels, \
                    params_mul, \
                    penalty_param_mu, \
                    penalty_param_v):

        
        opt = optax.adam(0.001)
        loss_list = []
        eq_cons_loss_list = []
        l_k_loss_list = []
        solver = jaxopt.LBFGS(fun=self.Loss.loss, maxiter=self.LBFGS_maxiter, \
                            linesearch=self.LBFGS_linesearch, tol=self.LBFGS_tol, 
                                stop_if_linesearch_fails=True, history_size=self.LBFGS_history_size)
        
        if experiment == "Augmented_Lag_experiment":
            params, _ = solver.run(params, penalty_param = penalty_param, mul = mul)

            loss_list.append(self.Loss.loss(params, mul, penalty_param))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))
                
        # elif experiment == "New_Augmented_Lag_experiment":
        #     for _ in tqdm(range(num_echos)):
        #         l, grads = value_and_grad(self.Loss.loss, 0)(params_mul, penalty_param, alpha, group_labels)
        #         params_mul = self.update(opt=opt, grads=grads, optim_object=params_mul)

        #         params, mul = params_mul
        #         eq_cons_loss = self.Loss.eq_cons_loss(params)
        #         l_k_loss = self.Loss.l_k(params)
        #         loss_list.append(l)
        #         eq_cons_loss_list.append(eq_cons_loss)
        #         l_k_loss_list.append(l_k_loss)

        elif experiment == "Fletcher_Penalty_experiment":         
            params, _ = solver.run(params, penalty_param = penalty_param, group_labels = group_labels)

            loss_list.append(self.Loss.loss(params, penalty_param, group_labels))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))

        elif experiment == "Bert_Aug_Lag_experiment":
            params_mul, _ = solver.run(params_mul, penalty_param_mu=penalty_param_mu, penalty_param_v=penalty_param_v)
            params, mul = params_mul

            loss_list.append(self.Loss.loss(params, penalty_param))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))

                # l, grads = value_and_grad(self.Loss.loss, 0)(params_mul, penalty_param_mu, penalty_param_v)
                # params_mul = self.update(opt=opt, grads=grads, optim_object=params_mul)


        else:
            for _ in range(100):
                l, grads = value_and_grad(self.Loss.loss, 0)(params, penalty_param)
                params = self.adam_update(opt=opt, grads=grads, optim_object=params)

            solver = jaxopt.LBFGS(fun=self.Loss.loss, maxiter=self.LBFGS_maxiter, \
                                linesearch=self.LBFGS_linesearch, tol=self.LBFGS_tol, 
                                    stop_if_linesearch_fails=True)
            params, _ = solver.run(params, penalty_param = penalty_param)

            loss_list.append(self.Loss.loss(params, penalty_param))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))

        return params, params_mul, jnp.array(loss_list), jnp.array(eq_cons_loss_list), jnp.array(l_k_loss_list), self.Loss.eq_cons(params)


    def evaluation(self, params, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = jnp.mean(np.abs(u_theta-ui))
        l2_relative_error = jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2)
        return absolute_error, l2_relative_error, u_theta
    











