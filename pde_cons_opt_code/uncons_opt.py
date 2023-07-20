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
    def __init__(self, model, Loss, panalty_param_upper_bound) -> None:
        self.model = model
        self.Loss = Loss
        self.panalty_param_upper_bound = panalty_param_upper_bound


    def flat_single_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
                        applymap(lambda x: x.flatten()).values.flatten())


    def update(self, opt, grads, optim_object):
        opt_state = opt.init(optim_object)
        grads, opt_state = opt.update(grads, opt_state)
        optim_object = optax.apply_updates(optim_object, grads)
        return optim_object


    def LBFGS_update(self, params, maxiter, linesearch, penalty_param):
        solver = jaxopt.LBFGS(fun=self.Loss.loss, maxiter=maxiter, \
                              linesearch=linesearch, tol=1e-3, 
                                stop_if_linesearch_fails=True)
        params, _ = solver.run(params, penalty_param = penalty_param)
        return params

    def adam_update(self, params, num_echos, \
                    penalty_param, experiment, \
                    mul, mul_num_echos, alpha, \
                    lr_schedule, group_labels, \
                    penalty_param_for_mul, \
                    params_mul, \
                    penalty_param_mu, \
                    penalty_param_v):

        
        opt = optax.adam(lr_schedule)
        loss_list = []
        eq_cons_loss_list = []
        l_k_loss_list = []
        # if experiment == "Pillo_Penalty_experiment":
        #     for step in tqdm(range(num_echos)):
        #         for i in range(mul_num_echos):
        #             mul_l, mul_grads = value_and_grad(self.Loss.get_mul_obj, 1)(params, mul, penalty_param_for_mul)
        #             mul = self.update(opt=opt, grads=mul_grads, optim_object=mul)

        #         l, grads = value_and_grad(self.Loss.loss, 0)(params, mul, penalty_param)
        #         params = self.update(opt=opt, grads=grads, optim_object=params)
        #         eq_cons_loss = self.Loss.eq_cons_loss(params)
        #         l_k_loss = self.Loss.l_k(params)
        #         loss_list.append(l)
        #         eq_cons_loss_list.append(eq_cons_loss)
        #         l_k_loss_list.append(l_k_loss)


        if experiment == "Augmented_Lag_experiment":
            for _ in tqdm(range(num_echos)):
                l, grads = value_and_grad(self.Loss.loss, 0)(params, mul, penalty_param)
                params = self.update(opt=opt, grads= grads, optim_object=params)

                eq_cons_loss = self.Loss.eq_cons_loss(params)
                l_k_loss = self.Loss.l_k(params)
                loss_list.append(l)
                eq_cons_loss_list.append(eq_cons_loss)
                l_k_loss_list.append(l_k_loss)
                # print(jnp.linalg.norm(self.flat_single_dict(jacfwd(self.Loss.loss, 0)(params, mul, penalty_param)), ord=jnp.inf))
                
                
        elif experiment == "New_Augmented_Lag_experiment":
            for _ in tqdm(range(num_echos)):
                l, grads = value_and_grad(self.Loss.loss, 0)(params_mul, penalty_param, alpha, group_labels)
                params_mul = self.update(opt=opt, grads=grads, optim_object=params_mul)

                params, mul = params_mul
                eq_cons_loss = self.Loss.eq_cons_loss(params)
                l_k_loss = self.Loss.l_k(params)
                loss_list.append(l)
                eq_cons_loss_list.append(eq_cons_loss)
                l_k_loss_list.append(l_k_loss)

        elif experiment == "Fletcher_Penalty_experiment":
            for _ in tqdm(range(num_echos)):
                l, grads = value_and_grad(self.Loss.loss, 0)(params, penalty_param, group_labels)
                params = self.update(opt=opt, grads= grads, optim_object=params)

                eq_cons_loss = self.Loss.eq_cons_loss(params)
                l_k_loss = self.Loss.l_k(params)
                loss_list.append(l)
                eq_cons_loss_list.append(eq_cons_loss)
                l_k_loss_list.append(l_k_loss)

        elif experiment == "Bert_Aug_Lag_experiment":
            for _ in tqdm(range(num_echos)):
                l, grads = value_and_grad(self.Loss.loss, 0)(params_mul, penalty_param_mu, penalty_param_v)
                params_mul = self.update(opt=opt, grads=grads, optim_object=params_mul)

                params, mul = params_mul
                eq_cons_loss = self.Loss.eq_cons_loss(params)
                l_k_loss = self.Loss.l_k(params)
                loss_list.append(l)
                eq_cons_loss_list.append(eq_cons_loss)
                l_k_loss_list.append(l_k_loss)

        else:
            params = self.LBFGS_update(params, num_echos, "hager-zhang", penalty_param)
            loss_list.append(self.Loss.loss(params, penalty_param))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))
            # for _ in tqdm(range(num_echos)):
            #     l, grads = value_and_grad(self.Loss.loss, 0)(params, penalty_param)
            #     params = self.update(opt=opt, grads = grads, optim_object=params)

            #     eq_cons_loss = self.Loss.eq_cons_loss(params)
            #     l_k_loss = self.Loss.l_k(params)
            #     loss_list.append(l)
            #     eq_cons_loss_list.append(eq_cons_loss)
            #     l_k_loss_list.append(l_k_loss)

        # return params, params_mul, jnp.array(loss_list), lr_schedule(num_echos), eq_cons_loss_list, l_k_loss_list, self.Loss.eq_cons(params)
        return params, params_mul, jnp.array(loss_list), jnp.array(eq_cons_loss_list), jnp.array(l_k_loss_list), self.Loss.eq_cons(params)


    def evaluation(self, params, data, ui):
        n = data.shape[0]
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = 1/n * jnp.linalg.norm(u_theta-ui, ord = 2)
        l2_relative_error = 1/n * (jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2))
        return absolute_error, l2_relative_error, u_theta
    











