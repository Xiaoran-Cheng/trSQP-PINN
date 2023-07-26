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
import jax
from scipy.optimize import minimize, BFGS

# LBFGS_linesearch, LBFGS_tol, LBFGS_maxiter, LBFGS_history_size
class Optim:
    def __init__(self, model, Loss) -> None:
        self.model = model
        self.Loss = Loss
        # self.LBFGS_linesearch = LBFGS_linesearch
        # self.LBFGS_tol = LBFGS_tol
        # self.LBFGS_maxiter = LBFGS_maxiter
        # self.LBFGS_history_size = LBFGS_history_size
        

    # def flatten_params(self, params):
    #     _, treedef = jax.tree_util.tree_flatten(params)
    #     return jax.flatten_util.ravel_pytree(params)[0], treedef
    

    # def unflatten_params(self, param_list, treedef):
    #     param_groups = jnp.split(param_list, self.indices)
    #     reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, self.shapes)]
    #     return jax.tree_util.tree_unflatten(treedef, reshaped_params)
    

    def adam_update(self, opt, grads, optim_object):
        opt_state = opt.init(optim_object)
        grads, opt_state = opt.update(grads, opt_state)
        optim_object = optax.apply_updates(optim_object, grads)
        return optim_object


    def update(self, params, \
                    penalty_param, experiment, \
                    mul, params_mul, \
                    penalty_param_mu, \
                    penalty_param_v, adam_iter, adam_opt, LBFGS_opt, LBFGS_maxiter, loss_values, eq_cons_loss_values):

        
        # adam_opt = optax.adam(adam_lr)
        loss_list = []
        eq_cons_loss_list = []
        l_k_loss_list = []
        # LBFGS_opt = jaxopt.LBFGS(fun=self.Loss.loss, maxiter=self.LBFGS_maxiter, \
        #                     linesearch=self.LBFGS_linesearch, tol=self.LBFGS_tol, 
        #                         stop_if_linesearch_fails=True, history_size=self.LBFGS_history_size)
        
        if experiment == "Augmented_Lag_experiment":
            # for _ in range(adam_iter):
            #     l, grads = value_and_grad(self.Loss.loss, 0)(params, penalty_param, mul)
            #     params = self.update(opt=adam_opt, grads=grads, optim_object=params)

            params, _ = LBFGS_opt.run(params, penalty_param = penalty_param, mul = mul)
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
            # for _ in range(adam_iter):
            #     l, grads = value_and_grad(self.Loss.loss, 0)(params, penalty_param)
            #     params = self.update(opt=adam_opt, grads=grads, optim_object=params)

            params, _ = LBFGS_opt.run(params, penalty_param = penalty_param)
            loss_list.append(self.Loss.loss(params, penalty_param))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))

        elif experiment == "Bert_Aug_Lag_experiment":
            # for _ in range(adam_iter):
            #     l, grads = value_and_grad(self.Loss.loss, 0)(params_mul, penalty_param_mu, penalty_param_v)
            #     params_mul = self.update(opt=adam_opt, grads=grads, optim_object=params_mul)
            # for _ in tqdm(range(10)):
            #     init_state = LBFGS_opt.init_state(init_params = params_mul, penalty_param_mu=penalty_param_mu, penalty_param_v=penalty_param_v)
            #     params_mul, _ = LBFGS_opt.update(params = params_mul, state = init_state, penalty_param_mu=penalty_param_mu, penalty_param_v=penalty_param_v)
                # print(LBFGS_opt.l2_optimality_error(params = params_mul, penalty_param_mu=penalty_param_mu, penalty_param_v=penalty_param_v))

            params_mul, _ = LBFGS_opt.run(params_mul, penalty_param_mu=penalty_param_mu, penalty_param_v=penalty_param_v)
            params = params_mul['params']
            mul = params_mul['mul']
            loss_list.append(self.Loss.loss(params_mul, penalty_param_mu, penalty_param_v))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))

        else:
            # params, _ = LBFGS_opt.run(params, penalty_param = penalty_param, dd = dd)
            # loss_list.append(self.Loss.loss(params, penalty_param, dd))
            # eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            # l_k_loss_list.append(self.Loss.l_k(params))


            # flat_params, treedef = self.Loss.flatten_params(params)
            # solution = minimize(self.Loss.loss, \
            #                     flat_params, \
            #                     args=(treedef, penalty_param, loss_values, eq_cons_loss_values), \
            #                     jac=self.Loss.grad_objective, \
            #                     method='L-BFGS-B', \
            #                     options={'maxiter': 10})
            # params = self.Loss.unflatten_params(solution.x, treedef)
            # print(solution)

            # loss_list.append(self.Loss.loss(solution.x, treedef, penalty_param, loss_values, eq_cons_loss_values))
            # eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(solution.x, treedef, eq_cons_loss_values), ord=2)))
            # l_k_loss_list.append(self.Loss.l_k(solution.x, treedef))


            



        return params, params_mul, jnp.array(loss_list), jnp.array(eq_cons_loss_list), jnp.array(l_k_loss_list), self.Loss.eq_cons(params)


    def evaluation(self, params, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = jnp.mean(np.abs(u_theta-ui))
        l2_relative_error = jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2)
        return absolute_error, l2_relative_error, u_theta
    











