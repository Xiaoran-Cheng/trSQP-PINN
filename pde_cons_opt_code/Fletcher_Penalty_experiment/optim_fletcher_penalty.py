import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd, vmap
import pandas as pd
from flax.core.frozen_dict import unfreeze
import numpy as np


class FletcherPenalty:
    def __init__(self, model, data, sample_data, IC_sample_data, ui, beta, N, M):
        self.model = model
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.ui = ui
        self.N = N
        self.M = M


    def l_k(self, params):
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
    

    def eq_cons_loss(self, params, penalty_param):
        return 0.5 * penalty_param * \
                jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))
    

    def flat_single_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
                        applymap(lambda x: x.primal.flatten()).values.flatten())
    

    def flat_multi_dict(self, dicts, group_labels):
        return np.concatenate(pd.DataFrame.from_dict(\
                unfreeze(dicts['params'])).\
                    apply(lambda x: x.explode()).set_index([group_labels]).\
                        sort_index().applymap(lambda x: x.primal.flatten()).values.flatten())
    
    
    def loss(self, params, penalty_param, group_labels):
        # gx = pd.DataFrame.from_dict(unfreeze(jacfwd(self.l_k, 0)(params)["params"])).values.flatten()
        # Ax_pinv = pd.DataFrame.from_dict(unfreeze(jacfwd(self.eq_cons, 0)(params)["params"])).\
        #                     applymap(lambda x: jnp.transpose(jnp.linalg.pinv(x), \
        #                     axes=(0,2,1)) if x.ndim == 3 else jnp.transpose(jnp.linalg.pinv(x))).values.flatten()
        # Axgx = lambda x, y: (x * y).sum(axis=(1,2)) if y.ndim == 3 else (x * y).sum(axis=1)
        # lambdax1 = jnp.array(list(map(Axgx, gx, Ax_pinv))).sum(axis=0)
        # return self.l_k(params) + self.eq_cons(params) @ lambdax + self.eq_cons_loss(params, penalty_param)


        flatted_gra_l_k = self.flat_single_dict(jacfwd(self.l_k, 0)(params))
        flatted_gra_eq_cons = jnp.array(jnp.split(self.flat_multi_dict(jacfwd(self.eq_cons, 0)(params), group_labels), 2*self.M))
        lambdax = jnp.linalg.pinv(flatted_gra_eq_cons.T) @ flatted_gra_l_k
        return self.l_k(params) + self.eq_cons(params) @ lambdax + self.eq_cons_loss(params, penalty_param)
        






# import time
# start_time = time.time()

# import sys
# import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# current_dir = os.getcwd().replace("\\", "/")
# sys.path.append(parent_dir)

# from PINN_experiment.optim_PINN import PINN
# from l1_Penalty_experiment.optim_l1_penalty import l1Penalty
# from l2_Penalty_experiment.optim_l2_penalty import l2Penalty
# from linfinity_Penalty_experiment.optim_linfinity_penalty import linfinityPenalty
# from Augmented_Lag_experiment.optim_aug_lag import AugLag
# from Pillo_Penalty_experiment.optim_pillo_penalty import PilloPenalty
# from SQP_experiment.optim_sqp import OptimComponents, SQP_Optim
# from New_Augmented_Lag_experiment.optim_new_aug_lag import NewAugLag
# # from Fletcher_Penalty_experiment.optim_fletcher_penalty import FletcherPenalty

# from data import Data
# from NN import NN
# from DataLoader import DataLoader
# from Visualization import Visualization
# from uncons_opt import Optim

# from jax import random
# import pandas as pd
# from jax import numpy as jnp
# from flax import linen as nn
# from jaxopt import EqualityConstrainedQP
# import jax.numpy as jnp
# import optax

# from multiprocessing import Pool




# #######################################config for data#######################################
# # beta_list = [10**-4, 30]
# beta_list = [30]
# N=100
# M=5
# data_key_num = 1000
# dim = 2
# Datas = Data(N=N, M=M, key_num=data_key_num, dim=dim)
# dataloader = DataLoader(Data=Datas)

# x_data_min = 0
# x_data_max = 2*jnp.pi
# t_data_min = 0
# t_data_max = 1
# x_sample_min = 0
# x_sample_max = 2*jnp.pi
# t_sample_min = 0
# t_sample_max = 1

# evaluation_data_key_num = 256
# eval_Datas = Data(N=N, M=M, key_num=evaluation_data_key_num, dim=dim)
# eval_dataloader = DataLoader(Data=eval_Datas)
# ####################################### config for data #######################################


# ####################################### config for NN #######################################
# NN_key_num = 345
# key = random.PRNGKey(NN_key_num)
# # features = [10, 10, 10, 10, 1]
# # features = [10, 10, 1] # 搭配 SQP_num_iter = 100， hessian_param = 0.6 # 0.6最好， init_stepsize = 1.0， line_search_tol = 0.001， line_search_max_iter = 30， line_search_condition = "strong-wolfe" ，line_search_decrease_factor = 0.8
# features = [2, 3, 1]
# ####################################### config for NN #######################################


# ####################################### config for penalty param #######################################
# penalty_param_update_factor = 2
# init_penalty_param = 1
# panalty_param_upper_bound = 150
# # converge_tol = 0.001
# uncons_optim_num_echos = 1
# init_uncons_optim_learning_rate = 0.001
# transition_steps = uncons_optim_num_echos
# decay_rate = 0.9
# end_value = 0.0001
# transition_begin = 0
# staircase = True
# max_iter_retrain = 0
# # cons_violation = 0.001 # threshold for updating penalty param
# ####################################### config for penalty param #######################################


# ####################################### config for lagrange multiplier #######################################
# init_mul = jnp.zeros(2*M) # initial  for Pillo_Penalty_experiment, Augmented_Lag_experiment, New_Augmented_Lag_experiment
# mul_num_echos = 10 # for Pillo_Penalty_experiment
# alpha = 150 # for New_Augmented_Lag_experiment
# ####################################### config for lagrange multiplier #######################################


# ####################################### visualization #######################################
# visual = Visualization(current_dir)
# ####################################### visualization #######################################


# ####################################### config for SQP #######################################
# qp = EqualityConstrainedQP(tol=1e-5, refine_regularization=3., refine_maxiter=50)
# SQP_num_iter = 100
# hessian_param = 0.6 # 0.6最好
# init_stepsize = 1.0
# line_search_tol = 0.001
# line_search_max_iter = 30
# line_search_condition = "strong-wolfe"  # armijo, goldstein, strong-wolfe or wolfe.
# line_search_decrease_factor = 0.8
# group_labels = list(range(1,2*M+1)) * 2
# ####################################### config for SQP #######################################





# # for experiment in ['PINN_experiment', \
# #                     'l1_Penalty_experiment', \
# #                     'l2_Penalty_experiment', \
# #                     'linfinity_Penalty_experiment', \
# #                     'Augmented_Lag_experiment', \    
# #                     'Pillo_Penalty_experiment', \ # check     2
# #                     'New_Augmented_Lag_experiment',\ # check  3
# #                     'Fletcher_Penalty_experiment', \  # check 1
# #                     'SQP_experiment']:            # check     4

# for experiment in ['Fletcher_Penalty_experiment']:

#     # for activation_input in ['sin', \
#     #                         'tanh', \
#     #                         'cos']:
#     for activation_input in ['identity']:

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

#         lr_schedule = optax.exponential_decay(
#         init_value=init_uncons_optim_learning_rate, 
#         transition_steps = transition_steps, 
#         decay_rate=decay_rate,
#         end_value = end_value,
#         transition_begin  = transition_begin,
#         staircase = staircase)
        
#         for beta in beta_list:
#             data, sample_data, IC_sample_data, ui = dataloader.get_data(\
#                 x_data_min, x_data_max, t_data_min, t_data_max, x_sample_min, \
#                     x_sample_max, t_sample_min, t_sample_max, beta, M)
#             params = model.init_params(key=key, data=data)

#             eval_data, _, _, eval_ui = eval_dataloader.get_data(x_data_min, x_data_max, \
#             t_data_min, t_data_max, x_sample_min, x_sample_max, t_sample_min, \
#             t_sample_max, beta, M)
            
#             if experiment == "SQP_experiment":
#                 optim_components = OptimComponents(model, data, sample_data, IC_sample_data, ui[0], beta)
#                 sqp_optim = SQP_Optim(model, optim_components, qp, features, group_labels, hessian_param, M, params)
#                 params, loss_list = sqp_optim.SQP_optim(params, SQP_num_iter, \
#                                             line_search_max_iter, line_search_condition, \
#                                                 line_search_decrease_factor, init_stepsize, line_search_tol)

#                 # absolute_error, l2_relative_error, eval_u_theta = \
#                 #     sqp_optim.evaluation(params, N, eval_data, eval_ui[0])
                    
#             else:
#                 if experiment == "PINN_experiment":
#                     loss = PINN(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "l1_Penalty_experiment":
#                     loss = l1Penalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "l2_Penalty_experiment":
#                     loss = l2Penalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "linfinity_Penalty_experiment":
#                     loss = linfinityPenalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "Augmented_Lag_experiment":
#                     loss = AugLag(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "Pillo_Penalty_experiment":
#                     loss = PilloPenalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "New_Augmented_Lag_experiment":
#                     loss = NewAugLag(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "Fletcher_Penalty_experiment":
#                     loss = FletcherPenalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
                    
#             optim = Optim(model, loss, panalty_param_upper_bound)
#             penalty_param = init_penalty_param
#             uncons_optim_learning_rate = init_uncons_optim_learning_rate
#             mul = init_mul
#             total_loss_list = []
#             total_eq_cons_loss_list = []
#             total_l_k_loss_list = []
#             iter_retrain = 0
#             while iter_retrain <= max_iter_retrain:
#                 params, loss_list, uncons_optim_learning_rate, eq_cons_loss_list, l_k_loss_list, eq_cons = optim.adam_update(params, uncons_optim_num_echos, \
#                                                                                 penalty_param, experiment, \
#                                                                                 mul, mul_num_echos, alpha, \
#                                                                                 lr_schedule, group_labels)
#                 iter_retrain+=1
#                 uncons_optim_learning_rate = lr_schedule(uncons_optim_num_echos * iter_retrain)
#                 if experiment == "Augmented_Lag_experiment":
#                     mul = mul + penalty_param * 2 * eq_cons
#                 if penalty_param < panalty_param_upper_bound:
#                     penalty_param = penalty_param_update_factor * penalty_param
       