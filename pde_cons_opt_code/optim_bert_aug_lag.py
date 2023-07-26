import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd
import jax


class BertAugLag:
    def __init__(self, model, data, sample_data, IC_sample_data, BC_sample_data, ui, beta, N, M):
        self.model = model
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.BC_sample_data = BC_sample_data
        self.ui = ui
        self.N = N
        self.M = M


    def l_k(self, params):
        # params = params_mul['params']
        # mul = params_mul['mul']
        u_theta = self.model.u_theta(params=params, data=self.data)
        return 1 / self.N * jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
    

    def IC_cons(self, params):
        # params = params_mul['params']
        # mul = params_mul['mul']
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        return Transport_eq(beta=self.beta).solution(\
            self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta
    
    
    def BC_cons(self, params):
        # params = params_mul['params']
        # mul = params_mul['mul']
        u_theta = self.model.u_theta(params=params, data=self.BC_sample_data)
        return Transport_eq(beta=self.beta).solution(\
            self.BC_sample_data[:,0], self.BC_sample_data[:,1]) - u_theta
    
    
    def pde_cons(self, params):
        # params = params_mul['params']
        # mul = params_mul['mul']
        grad_x = jacfwd(self.model.u_theta, 1)(params, self.sample_data)
        return Transport_eq(beta=self.beta).pde(jnp.diag(grad_x[:,:,0]),\
            jnp.diag(grad_x[:,:,1]))
    

    def eq_cons(self, params):
        # params = params_mul['params']
        # mul = params_mul['mul']
        return jnp.concatenate([self.IC_cons(params), self.BC_cons(params), self.pde_cons(params)])
    

    def eq_cons_loss(self, params):
        # params = params_mul['params']
        # mul = params_mul['mul']
        return jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))


    def L(self, params_mul):
        params = params_mul['params']
        mul = params_mul['mul']
        return self.l_k(params) + self.eq_cons(params) @ mul
    
    
    # def flat_single_dict(self, dicts):
    #     # return jnp.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
    #     #                 applymap(lambda x: x.flatten()).values.flatten().tolist())
    #     return jax.flatten_util.ravel_pytree(dicts)[0]


    def loss(self, params_mul, penalty_param_mu, penalty_param_v):
        params = params_mul['params']
        # mul = params_mul['mul']
        opt_error_penalty = jnp.square(jnp.linalg.norm(jax.flatten_util.ravel_pytree(jacfwd(self.L, 0)(params_mul)['params'])[0],ord=2))
        return self.L(params_mul) + 0.5 * penalty_param_mu * self.eq_cons_loss(params) + 0.5 * penalty_param_v * opt_error_penalty
    




# import time
# start_time = time.time()

# import sys
# import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# current_dir = os.getcwd().replace("\\", "/")
# sys.path.append(parent_dir)

# from optim_PINN import PINN
# from optim_l1_penalty import l1Penalty
# from optim_l2_penalty import l2Penalty
# # from optim_linfinity_penalty import linfinityPenalty
# from optim_aug_lag import AugLag
# # from optim_pillo_penalty import PilloPenalty
# # from optim_new_aug_lag import NewAugLag
# from optim_fletcher_penalty import FletcherPenalty
# # from optim_bert_aug_lag import BertAugLag
# from optim_sqp import SQP_Optim

# from data import Data
# from NN import NN
# from DataLoader import DataLoader
# from Visualization import Visualization
# from uncons_opt import Optim

# from jax import random
# import pandas as pd
# from jax import numpy as jnp
# from flax import linen as nn
# # from jaxopt import EqualityConstrainedQP, CvxpyQP, OSQP
# import jax.numpy as jnp
# # from flax.core.frozen_dict import FrozenDict, unfreeze
# import numpy as np
# import jaxlib.xla_extension as xla
# import jaxopt
# import optax
# from tqdm import tqdm

# #######################################config for data#######################################
# beta_list = [0.0001]
# xgrid = 256
# nt = 100
# N=10
# M=120
# data_key_num, sample_data_key_num = 100, 256
# eval_data_key_num, eval_sample_data_key_num = 300, 756
# dim = 2
# Datas = Data(N=N, M=M, dim=dim)
# dataloader = DataLoader(Data=Datas)

# x_data_min = 0
# x_data_max = 2*jnp.pi
# t_data_min = 0
# t_data_max = 1
# x_sample_min = 0
# x_sample_max = 2*jnp.pi
# t_sample_min = 0
# t_sample_max = 1
# ####################################### config for data #######################################


# ####################################### config for NN #######################################
# NN_key_num = 345
# key = random.PRNGKey(NN_key_num)
# # features = [50, 50, 50, 50, 1]
# features = [3, 3, 1]
# ####################################### config for NN #######################################


# ####################################### config for penalty param #######################################
# penalty_param_update_factor = 10
# init_penalty_param = 1
# panalty_param_upper_bound = 10**6
# LBFGS_maxiter = 2
# # init_uncons_optim_learning_rate = 0.001
# # transition_steps = uncons_optim_num_echos
# # decay_rate = 0.9
# # end_value = 0.0001
# # transition_begin = 0
# # staircase = True
# max_iter_train = 1
# penalty_param_for_mul = 5
# init_penalty_param_v = init_penalty_param
# init_penalty_param_mu = init_penalty_param
# LBFGS_linesearch = "hager-zhang"
# LBFGS_tol = 1e-10
# LBFGS_history_size = 50

# adam_lr, adam_iter = 0.001, 100
# ####################################### config for penalty param #######################################


# ####################################### config for lagrange multiplier #######################################
# init_mul = jnp.ones(M) # initial  for Pillo_Penalty_experiment, Augmented_Lag_experiment, New_Augmented_Lag_experiment
# # alpha = 10**6 # for New_Augmented_Lag_experiment
# ####################################### config for lagrange multiplier #######################################


# ####################################### visualization #######################################
# visual = Visualization(current_dir)
# ####################################### visualization #######################################


# ####################################### config for SQP #######################################
# # # qp = EqualityConstrainedQP(tol=0.001) # , refine_regularization=3, refine_maxiter=50
# # qp = CvxpyQP(solver='OSQP') # "OSQP", "ECOS", "SCS" , implicit_diff_solve=True
# # SQP_num_iter = 100
# # hessian_param = 0.6
# # init_stepsize = 1.0
# # line_search_tol = 0
# # line_search_max_iter = 100
# # line_search_condition = "armijo"  # armijo, goldstein, strong-wolfe or wolfe.
# # line_search_decrease_factor = 0.8
# maxiter = 100000
# # group_labels = list(range(1,M+1)) * 2
# # qr_ind_tol = 1e-5
# # merit_func_penalty_param = 1
# ####################################### config for SQP #######################################


# error_df_list = []
# # for experiment in ['PINN_experiment', 
# #                    'l1_Penalty_experiment',
# #                     'l2_Penalty_experiment', 
# #                     'Augmented_Lag_experiment',
# #                     'SQP_experiment',
# #                     'Bert_Aug_Lag_experiment']:


# for experiment in ['Bert_Aug_Lag_experiment']:


#     for activation_input in ['tanh']:

#         if activation_input == "sin":
#             activation = jnp.sin
#         elif activation_input == "tanh":
#             activation = nn.tanh
#         elif activation_input == "cos":
#             activation = jnp.cos
#         elif activation_input == "identity":
#             def identity(x):
#                 return x
#             activation = identity

#         activation_name = activation.__name__
#         model = NN(features=features, activation=activation)
#         absolute_error_list = []
#         l2_relative_error_list = []

#         # lr_schedule = optax.exponential_decay(
#         # init_value=init_uncons_optim_learning_rate, 
#         # transition_steps = transition_steps, 
#         # decay_rate=decay_rate,
#         # end_value = end_value,
#         # transition_begin  = transition_begin,
#         # staircase = staircase)
        
#         for beta in beta_list:
#             data, sample_data, IC_sample_data, BC_sample_data, ui = dataloader.get_data(\
#                 xgrid, nt, x_data_min, x_data_max, t_data_min, t_data_max, \
#                     x_sample_min, x_sample_max, t_sample_min, t_sample_max, \
#                         beta, M, data_key_num, sample_data_key_num)
            
#             params = model.init_params(key=key, data=data)
#             params_mul = {"params": params, "mul": init_mul}

#             eval_data, eval_ui = dataloader.get_eval_data(xgrid, nt, x_data_min, x_data_max, t_data_min, t_data_max, beta)

#             penalty_param = init_penalty_param
#             penalty_param_v = init_penalty_param_v
#             penalty_param_mu = init_penalty_param_mu
#             mul = init_mul
            
#             if experiment == "SQP_experiment":
#                 loss_values = []
#                 eq_cons_loss_values = []
#                 sqp_optim = SQP_Optim(model, features, M, params, beta, data, sample_data, IC_sample_data, BC_sample_data, ui, N)
#                 params = sqp_optim.SQP_optim(params, loss_values, eq_cons_loss_values, maxiter)
#                 total_l_k_loss_list = [i.item() for i in loss_values if isinstance(i, xla.ArrayImpl)]
#                 total_eq_cons_loss_list = [i.item() for i in eq_cons_loss_values if isinstance(i, xla.ArrayImpl)]

#                 absolute_error, l2_relative_error, eval_u_theta = \
#                     sqp_optim.evaluation(params, eval_data, eval_ui[0])
                
#             else:
#                 if experiment == "PINN_experiment":
#                     loss = PINN(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "l1_Penalty_experiment":
#                     loss = l1Penalty(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "l2_Penalty_experiment":
#                     loss = l2Penalty(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                                 N, M)
#                 # elif experiment == "linfinity_Penalty_experiment":
#                 #     loss = linfinityPenalty(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                 #                 N, M)
#                 elif experiment == "Augmented_Lag_experiment":
#                     loss = AugLag(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                                 N, M)
#                 # elif experiment == "Pillo_Penalty_experiment":
#                 #     loss = PilloPenalty(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                 #                 N, M)
#                 # elif experiment == "New_Augmented_Lag_experiment":
#                 #     loss = NewAugLag(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                 #                 N, M)
#                 elif experiment == "Fletcher_Penalty_experiment":
#                     loss = FletcherPenalty(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "Bert_Aug_Lag_experiment":
#                     loss = BertAugLag(model, data, sample_data, IC_sample_data, BC_sample_data, ui[0], beta, \
#                                 N, M)
                    
#                     # print(jnp.square(jnp.linalg.norm(loss.flat_single_dict(jacfwd(loss.L, 0)(params_mul)['params']),ord=2)))


#                     print(jnp.square(jnp.linalg.norm(jax.flatten_util.ravel_pytree(jacfwd(loss.L, 0)(params_mul)['params'])[0],ord=2)))
#                     end_time = time.time()
#                     print(f"Execution Time: {end_time - start_time} seconds")




