import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from jax import numpy as jnp
import optax
import numpy as np


class Optim:
    def __init__(self, model, Loss) -> None:
        self.model = model
        self.Loss = Loss


    def adam_update(self, opt, grads, optim_object):
        opt_state = opt.init(optim_object)
        grads, opt_state = opt.update(grads, opt_state)
        optim_object = optax.apply_updates(optim_object, grads)
        return optim_object


    def update(self, params, \
                    penalty_param, experiment, \
                    mul, params_mul, \
                    penalty_param_mu, \
                    penalty_param_v, LBFGS_opt):

        loss_list = []
        eq_cons_loss_list = []
        l_k_loss_list = []        
        if experiment == "Augmented_Lag_experiment":
            params, _ = LBFGS_opt.run(params, penalty_param = penalty_param, mul = mul)
            loss_list.append(self.Loss.loss(params, mul, penalty_param))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))


        elif experiment == "Fletcher_Penalty_experiment":
            params, _ = LBFGS_opt.run(params, penalty_param = penalty_param)
            loss_list.append(self.Loss.loss(params, penalty_param))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))

        elif experiment == "Pillo_Aug_Lag_experiment":
            params_mul, _ = LBFGS_opt.run(params_mul, penalty_param_mu=penalty_param_mu, penalty_param_v=penalty_param_v)
            params = params_mul['params']
            mul = params_mul['mul']
            loss_list.append(self.Loss.loss(params_mul, penalty_param_mu, penalty_param_v))
            eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            l_k_loss_list.append(self.Loss.l_k(params))

        else:
            params, _ = LBFGS_opt.run(params, penalty_param = penalty_param)
            # loss_list.append(self.Loss.loss(params, penalty_param))
            # eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(self.Loss.eq_cons(params), ord=2)))
            # l_k_loss_list.append(self.Loss.l_k(params))

        # return params, params_mul, jnp.array(loss_list), jnp.array(eq_cons_loss_list), jnp.array(l_k_loss_list), self.Loss.eq_cons(params)
        return params, params_mul, self.Loss.eq_cons(params)


    def evaluation(self, params, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = jnp.mean(np.abs(u_theta-ui))
        l2_relative_error = jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2)
        return absolute_error, l2_relative_error, u_theta
    











