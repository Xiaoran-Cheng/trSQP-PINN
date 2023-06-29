import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from jax import numpy as jnp
from jax import value_and_grad, jacfwd
import optax
from tqdm.notebook import tqdm
import pandas as pd
from flax.core.frozen_dict import unfreeze
# from tqdm import tqdm
# import jaxopt

# from flat_params import FlatParams

class Optim:
    def __init__(self, model, Loss, cons_violation, panalty_param_upper_bound) -> None:
        self.model = model
        self.Loss = Loss
        self.cons_violation = cons_violation
        self.panalty_param_upper_bound = panalty_param_upper_bound


    def update(self, opt, grads, optim_object):
        opt_state = opt.init(optim_object)
        grads, opt_state = opt.update(grads, opt_state)
        optim_object = optax.apply_updates(optim_object, grads)
        return optim_object


    def adam_update(self, params, num_echos, learning_rate, \
                    penalty_param, penalty_param_update_factor, experiment, \
                    mul, params_mul, mul_num_echos, alpha):
        
        opt = optax.adam(learning_rate)
        loss_list = []
        if experiment == "Pillo_Penalty_experiment":
            for _ in tqdm(range(num_echos)):
                for i in range(mul_num_echos):
                    mul_l, mul_grads = value_and_grad(self.Loss.get_mul_obj, 1)(params, mul, penalty_param)
                    updated_mul = self.update(opt=opt, grads=mul_grads, optim_object=mul)

                l, grads = value_and_grad(self.Loss.loss, 0)(params, updated_mul, penalty_param)
                eq_cons = self.Loss.eq_cons(params)
                eq_cons_violation = jnp.square(jnp.linalg.norm(eq_cons,ord=2))
                if eq_cons_violation > self.cons_violation and penalty_param < self.panalty_param_upper_bound:
                    penalty_param = penalty_param_update_factor * penalty_param
                params = self.update(opt=opt, grads= grads, optim_object=params)
                loss_list.append(l)

        elif experiment == "Augmented_Lag_experiment":
            for _ in tqdm(range(num_echos)):
                l, grads = value_and_grad(self.Loss.loss, 0)(params, mul, penalty_param)
                eq_cons = self.Loss.eq_cons(params)
                eq_cons_violation = jnp.square(jnp.linalg.norm(eq_cons,ord=2))
                mul = mul + penalty_param * eq_cons
                if eq_cons_violation > self.cons_violation and penalty_param < self.panalty_param_upper_bound:
                    penalty_param = penalty_param_update_factor * penalty_param
                params = self.update(opt=opt, grads= grads, optim_object=params)
                loss_list.append(l)

        elif experiment == "New_Augmented_Lag_experiment":
            for _ in tqdm(range(num_echos)):
                l, grads = value_and_grad(self.Loss.loss, 0)(params, mul, penalty_param, alpha)
                eq_cons = self.Loss.eq_cons(params)
                eq_cons_violation = jnp.square(jnp.linalg.norm(eq_cons,ord=2))

                Ax_pinv = -pd.DataFrame.from_dict(unfreeze(jacfwd(self.Loss.eq_cons, 0)(params)["params"])).\
                            applymap(lambda x: jnp.transpose(jnp.linalg.pinv(x), \
                            axes=(0,2,1)) if x.ndim == 3 else jnp.transpose(jnp.linalg.pinv(x))).values.flatten()
                
                gx = pd.DataFrame.from_dict(unfreeze(jacfwd(self.Loss.l_k, 0)(params)["params"])).values.flatten()
                Ax_pinv_gx = lambda x, y: (x * y).sum(axis=(1,2)) if y.ndim == 3 else (x * y).sum(axis=1)
                mul = jnp.array(list(map(Ax_pinv_gx, gx, Ax_pinv))).sum(axis=0)
                if eq_cons_violation > self.cons_violation and penalty_param < self.panalty_param_upper_bound:
                    penalty_param = penalty_param_update_factor * penalty_param
                params = self.update(opt=opt, grads= grads, optim_object=params)
                loss_list.append(l)

        elif experiment == "Fletcher_Penalty_experiment":
            for _ in tqdm(range(num_echos)):
                l, grads = value_and_grad(self.Loss.loss, 0)(params, penalty_param)
                eq_cons = self.Loss.eq_cons(params)
                eq_cons_violation = jnp.square(jnp.linalg.norm(eq_cons,ord=2))
                if eq_cons_violation > self.cons_violation and penalty_param < self.panalty_param_upper_bound:
                    penalty_param = penalty_param_update_factor * penalty_param
                params = self.update(opt=opt, grads= grads, optim_object=params)
                loss_list.append(l)

        else:
            for _ in tqdm(range(num_echos)):
                l, grads = value_and_grad(self.Loss.loss, 0)(params, penalty_param)
                params = self.update(opt=opt, grads = grads, optim_object=params)
                eq_cons = self.Loss.eq_cons(params)
                eq_cons_violation = jnp.square(jnp.linalg.norm(eq_cons,ord=2))
                if eq_cons_violation > self.cons_violation and penalty_param < self.panalty_param_upper_bound:
                    penalty_param = penalty_param_update_factor * penalty_param
                loss_list.append(l)
        return params, jnp.array(loss_list)


    # def LBFGS_update(self, params, maxiter, linesearch, penalty_param):
    #     solver = jaxopt.LBFGS(fun=self.Loss.loss, maxiter=maxiter, \
    #                           linesearch=linesearch)
    #     params, _ = solver.run(params)
    #     loss = self.Loss.loss(params, penalty_param)
    #     return params, loss
    

    def evaluation(self, params, N, data, ui):
        u_theta = self.model.u_theta(params = params, data=data)
        absolute_error = 1/N * jnp.linalg.norm(u_theta-ui, ord = 2)
        l2_relative_error = 1/N * (jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2))
        return absolute_error, l2_relative_error, u_theta
    
    