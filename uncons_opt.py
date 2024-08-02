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
        self.stop_optimization = False


    def adam_update(self, opt, grads, optim_object):
        opt_state = opt.init(optim_object)
        grads, opt_state = opt.update(grads, opt_state)
        optim_object = optax.apply_updates(optim_object, grads)
        return optim_object


    def update(self, params, \
                    penalty_param, experiment, \
                    mul, LBFGS_opt):
     
        # if experiment == "Augmented_Lag_experiment":
        #     # params, _ = LBFGS_opt.run(params, penalty_param = penalty_param, mul = mul)
        #     try:
        #         params, _ = LBFGS_opt.run(params, penalty_param = penalty_param, mul = mul)
        #     except TerminationCondition as e:
        #         pass

        # else:
        #     # params, _ = LBFGS_opt.run(params, penalty_param = penalty_param)
        #     try:
        #         params, _ = LBFGS_opt.run(params, penalty_param = penalty_param)
        #     except TerminationCondition as e:
        #         pass


        # try:
        if experiment == "Augmented_Lag_experiment":
            params, _ = LBFGS_opt.run(params, penalty_param=penalty_param, mul=mul)
        else:
            params, _ = LBFGS_opt.run(params, penalty_param=penalty_param)

        print(Exception)

        # except TerminationCondition as e:
        #     print("Termination condition reached:", str(e))

        return params, self.Loss.eq_cons(params)


    def evaluation(self, params, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = jnp.mean(np.abs(u_theta-ui))
        l2_relative_error = jnp.power(jnp.power((u_theta-ui), 2).sum(), 1/2) / jnp.power(jnp.power((ui), 2).sum(), 1/2)
        return absolute_error, l2_relative_error, u_theta
    



class TerminationCondition(Exception):
    pass







