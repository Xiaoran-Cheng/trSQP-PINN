import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from flax.core.frozen_dict import FrozenDict, unfreeze
from jaxopt import BacktrackingLineSearch, HagerZhangLineSearch



class OptimComponents:
    def __init__(self, model, data, sample_data, IC_sample_data, ui, beta):
        self.model = model
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.ui = ui


    def obj(self, params):
        u_theta = self.model.u_theta(params=params, data=self.data)
        return jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
    

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
    

    def L(self, params, mul):
        return self.l_k(params) + self.eq_cons(params) @ mul
    

    def get_grads(self, params):
        gra_obj = jacfwd(self.obj, 0)(params)
        gra_eq_cons = jacfwd(self.eq_cons, 0)(params)
        # Hlag = hessian(self.lag, 0)(params, mul)
        return gra_obj, gra_eq_cons
    
        
        
class SQP_Optim:
    def __init__(self, model, optim_components, qp, feature, group_labels, hessian_param, M, params) -> None:
        self.model = model
        self.optim_components = optim_components
        self.qp = qp
        self.feature = feature
        self.group_labels = group_labels
        self.hessian_param = hessian_param
        self.M = M
        self.layer_names = params["params"].keys()


    # def get_shapes_size(self, params):
    #     shapes = pd.DataFrame.from_dict(unfreeze(params["params"])).applymap(lambda x: x.shape).values.flatten()
    #     sizes = [np.prod(shape) for shape in shapes]
    #     return shapes, sizes
    

    def flat_single_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
                        applymap(lambda x: x.flatten()).values.flatten())
    

    def flat_multi_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(\
                unfreeze(dicts['params'])).\
                    apply(lambda x: x.explode()).set_index([self.group_labels]).\
                        sort_index().applymap(lambda x: x.flatten()).values.flatten())
    

    # def get_direction(self, flatted_gra_obj, flatted_gra_eq_cons, eq_cons):
    #     Q = self.hessiam_param * jnp.identity(flatted_gra_obj.shape[0])
    #     c = flatted_gra_obj
    #     A = jnp.array(jnp.split(flatted_gra_eq_cons, 2*self.M))
    #     b = -eq_cons
    #     flatted_delta_params = self.qp.run(params_obj=(Q, c), params_eq=(A, b)).params.primal
    #     return flatted_delta_params
    

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
    

    # def get_step_size(self, maxiter, condition, decrease_factor, params, delta_params, current_obj, gra_obj, init_stepsize):
    #         ls = BacktrackingLineSearch(fun=self.optim_components.obj, maxiter=maxiter, condition=condition,
    #                                     decrease_factor=decrease_factor)
    #         # ls = HagerZhangLineSearch(fun=self.optim_components.obj)
    #         stepsize, _ = ls.run(init_stepsize=init_stepsize, params=params,
    #                                 descent_direction=delta_params,
    #                                 value=current_obj, grad=gra_obj)
    #         return stepsize


    def SQP_optim(self, params, num_iter, maxiter, condition, decrease_factor, init_stepsize, line_search_tol):
        obj_list = []
        shapes = pd.DataFrame.from_dict(unfreeze(params["params"])).applymap(lambda x: x.shape).values.flatten()
        sizes = [np.prod(shape) for shape in shapes]
        for _ in tqdm(range(num_iter)):
            gra_obj, gra_eq_cons = self.optim_components.get_grads(params=params)
            eq_cons = self.optim_components.eq_cons(params=params)
            current_obj = self.optim_components.obj(params=params)

            flatted_gra_obj = self.flat_single_dict(gra_obj)
            flatted_current_params = self.flat_single_dict(params)
            flatted_gra_eq_cons = self.flat_multi_dict(gra_eq_cons)
            


            # flatted_delta_params = self.get_direction(flatted_gra_obj, flatted_gra_eq_cons, eq_cons)

            Q = self.hessian_param * jnp.identity(flatted_gra_obj.shape[0])
            c = flatted_gra_obj
            A = jnp.array(jnp.split(flatted_gra_eq_cons, 2*self.M))
            b = -eq_cons
            flatted_delta_params = self.qp.run(params_obj=(Q, c), params_eq=(A, b)).params.primal



            delta_params = self.get_recovered_dict(flatted_delta_params, shapes, sizes)

            # stepsize = self.get_step_size(maxiter=maxiter, condition=condition, decrease_factor=decrease_factor, \
            #             params=params, delta_params=delta_params, current_obj=current_obj, gra_obj=gra_obj, init_stepsize=init_stepsize)
            
            # ls = BacktrackingLineSearch(fun=self.optim_components.obj, maxiter=maxiter, condition=condition,
            #                             decrease_factor=decrease_factor)
            # # ls = HagerZhangLineSearch(fun=self.optim_components.obj)
            # stepsize, _ = ls.run(init_stepsize=init_stepsize, params=params,
            #                         descent_direction=delta_params,
            #                         value=current_obj, grad=gra_obj)
            ls = BacktrackingLineSearch(fun=self.optim_components.obj, maxiter=maxiter, condition=condition,
                                        decrease_factor=decrease_factor, tol=line_search_tol)
            stepsize, _ = ls.run(init_stepsize=init_stepsize, params=params,
                                    descent_direction=delta_params,
                                            value=current_obj, grad=gra_obj)



            flatted_updated_params = stepsize * flatted_delta_params + flatted_current_params
            params = self.get_recovered_dict(flatted_updated_params, shapes, sizes)
            obj_list.append(self.optim_components.obj(params))
        return params, obj_list
        

    def evaluation(self, params, N, data, ui):
        u_theta = self.model.u_theta(params = params, data=data)
        absolute_error = 1/N * jnp.linalg.norm(u_theta-ui)
        l2_relative_error = 1/N * jnp.linalg.norm((u_theta-ui)/ui)
        return absolute_error, l2_relative_error, u_theta
    







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
# from Han_Penalty_experiment.optim_han_penalty import HanPenalty
# from Cubic_Penalty_experiment.optim_cubic_penalty import CubicPenalty
# from Augmented_Lag_experiment.optim_aug_lag import AugLag
# from Pillo_Penalty_experiment.optim_pillo_penalty import PilloPenalty
# # from SQP_experiment.optim_sqp import OptimComponents, SQP_Optim
# from New_Augmented_Lag_experiment.optim_new_aug_lag import NewAugLag
# from Fletcher_Penalty_experiment.optim_fletcher_penalty import FletcherPenalty

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

# from multiprocessing import Pool




# #######################################config for data#######################################
# # beta_list = [10**-4, 30]
# beta_list = [10**-4]
# N=3
# M=4
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
# # features = [50, 50, 50, 50, 1]
# features = [2, 3, 1]
# ####################################### config for NN #######################################


# ####################################### config for penalty param #######################################
# penalty_param_update_factor = 1.01
# penalty_param = 1
# ####################################### config for penalty param #######################################


# ####################################### config for lagrange multiplier #######################################
# mul = jnp.ones(2*M) # initial  for Han_penalty_experiment, Pillo_Penalty_experiment, Augmented_Lag_experiment, New_Augmented_Lag_experiment
# mul_num_echos = 10 # for Pillo_Penalty_experiment
# alpha = 150 # for New_Augmented_Lag_experiment
# ####################################### config for lagrange multiplier #######################################


# ####################################### config for Adam #######################################
# uncons_optim_num_echos = 1000
# uncons_optim_learning_rate = 0.001
# cons_violation = 10 # threshold for updating penalty param
# ####################################### config for Adam #######################################


# ####################################### visualization #######################################
# visual = Visualization(current_dir)
# ####################################### visualization #######################################


# ####################################### config for SQP #######################################
# qp = EqualityConstrainedQP()
# SQP_num_iter = 4
# hessiam_param = 1.0
# init_stepsize = 0.8
# line_search_max_iter = 20
# line_search_condition = "strong-wolfe" #  “armijo”, “goldstein”, “strong-wolfe” or “wolfe”.
# line_search_decrease_factor = 0.8
# group_labels = list(range(1,2*M+1)) * 2
# ####################################### config for SQP #######################################





# # for experiment in ['PINN_experiment', \
# #                     'l1_Penalty_experiment', \
# #                     'l2_Penalty_experiment', \
# #                     'linfinity_Penalty_experiment', \
# #                     'Cubic_Penalty_experiment', \
# #                     'Han_penalty_experiment', \
# #                     'Augmented_Lag_experiment', \    #得有increase penalty param
# #                     'Pillo_Penalty_experiment', \
# #                     'New_Augmented_Lag_experiment',\  #问题
# #                     'Fletcher_Penalty_experiment', \  #问题
# #                     'SQP_experiment']:
# for experiment in ['SQP_experiment']:

#     # for activation_input in ['sin', \
#     #                         'tanh', \
#     #                         'cos']:
#     for activation_input in ['sin']:

#         if activation_input == "sin":
#             activation = jnp.sin
#         elif activation_input == "tanh":
#             activation = nn.tanh
#         elif activation_input == "cos":
#             activation = jnp.cos
            
#         activation_name = activation.__name__

#         absolute_error_list = []
#         l2_relative_error_list = []
#         for beta in beta_list:

#             data, sample_data, IC_sample_data, ui = dataloader.get_data(\
#                 x_data_min, x_data_max, t_data_min, t_data_max, x_sample_min, \
#                     x_sample_max, t_sample_min, t_sample_max, beta, M)

#             model = NN(features=features, activation=activation)
#             params = model.init_params(key=key, data=data)
#             params_mul = [params, mul]

#             eval_data, _, _, eval_ui = eval_dataloader.get_data(x_data_min, x_data_max, \
#             t_data_min, t_data_max, x_sample_min, x_sample_max, t_sample_min, \
#             t_sample_max, beta, M)

#             if experiment == "SQP_experiment":
#                 optim_components = OptimComponents(model, data, sample_data, IC_sample_data, ui[0], beta)
#                 sqp_optim = SQP_Optim(model, optim_components, qp, features, group_labels, hessiam_param, M, params)
#                 updated_params, loss_list = sqp_optim.SQP_optim(params, SQP_num_iter, \
#                                             line_search_max_iter, line_search_condition, line_search_decrease_factor, init_stepsize)

#                 absolute_error, l2_relative_error, eval_u_theta = \
#                     sqp_optim.evaluation(updated_params, N, eval_data, eval_ui[0])
                
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
#                 elif experiment == "Han_penalty_experiment":
#                     loss = HanPenalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
#                                 N, M)
#                 elif experiment == "Cubic_Penalty_experiment":
#                     loss = CubicPenalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
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
                
#                 optim = Optim(model, loss, cons_violation)
#                 updated_params, loss_list = optim.adam_update(params, \
#                                                         uncons_optim_num_echos, uncons_optim_learning_rate, \
#                                                         penalty_param, penalty_param_update_factor, \
#                                                         experiment, mul, params_mul, mul_num_echos, alpha)

#                 absolute_error, l2_relative_error, eval_u_theta = optim.evaluation(\
#                                                 updated_params, N, eval_data, eval_ui[0])
                
#             visual.line_graph(loss_list, "/{activation_name}/Total_Loss".\
#                 format(activation_name=activation_name), experiment=experiment, beta=beta)
#             visual.line_graph(eval_ui[0], "/{activation_name}/True_sol".\
#                     format(activation_name=activation_name), experiment=experiment, beta=beta)
#             visual.line_graph(eval_u_theta, "/{activation_name}/u_theta".\
#                     format(activation_name=activation_name), experiment=experiment, beta=beta)
#             visual.heatmap(eval_data, eval_ui[0], "/{activation_name}/True_sol_heatmap".\
#                     format(activation_name=activation_name), experiment=experiment, beta=beta)
#             visual.heatmap(eval_data, eval_u_theta, "/{activation_name}/u_theta_heatmap".\
#                     format(activation_name=activation_name), experiment=experiment, beta=beta)

#             absolute_error_list.append(absolute_error)
#             l2_relative_error_list.append(l2_relative_error)

#             print("last loss: "+str(loss_list[-1]))
#             print("absolute_error: " + str(absolute_error))
#             print("l2_relative_error: " + str(l2_relative_error))

#         error_df = pd.DataFrame({'Beta': beta_list, 'absolute_error': absolute_error_list, \
#                                 'l2_relative_error': l2_relative_error_list}).astype(float)
#         folder_path = "{current_dir}/{experiment}/pics/{activation_name}/error".\
#             format(experiment=experiment, \
#                     current_dir=current_dir, activation_name=activation_name)
        
#         visual.error_graph(error_df, folder_path, "/{activation_name}".\
#                 format(activation_name=activation_name), experiment=experiment)

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time} seconds")

