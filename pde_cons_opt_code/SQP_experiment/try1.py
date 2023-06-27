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
    
        

# class SQP_Optim:
#     def __init__(self, model, optim_components, qp, feature, group_labels, hessiam_param, M) -> None:
#         self.model = model
#         self.optim_components = optim_components
#         self.qp = qp
#         self.feature = feature
#         self.group_labels = group_labels
#         self.hessiam_param = hessiam_param
#         self.M = M


#     def SQP_optim(self, params, num_iter, maxiter, condition, decrease_factor, init_stepsize):
#         obj_list = []
#         for _ in tqdm(range(num_iter)):
#             shapes = pd.DataFrame.from_dict(unfreeze(params["params"])).applymap(lambda x: x.shape).values.flatten()
#             sizes = [np.prod(shape) for shape in shapes]
#             gra_obj, gra_eq_cons = self.optim_components.get_grads(params=params)
#             eq_cons = self.optim_components.eq_cons(params=params)
#             flatted_gra_obj = np.concatenate(pd.DataFrame.from_dict(unfreeze(gra_obj["params"])).\
#                         applymap(lambda x: x.flatten()).values.flatten())
#             flatted_current_params = np.concatenate(pd.DataFrame.from_dict(unfreeze(params["params"])).\
#                         applymap(lambda x: x.flatten()).values.flatten())
#             flatted_gra_eq_cons = np.concatenate(pd.DataFrame.from_dict(\
#                 unfreeze(gra_eq_cons['params'])).\
#                     apply(lambda x: x.explode()).set_index([self.group_labels]).\
#                         sort_index().applymap(lambda x: x.flatten()).values.flatten())
            
#             Q = self.hessiam_param * jnp.identity(flatted_gra_obj.shape[0])
#             c = flatted_gra_obj
#             A = jnp.array(jnp.split(flatted_gra_eq_cons, 2*self.M))
#             b = -eq_cons
#             qp = EqualityConstrainedQP()
#             sol = qp.run(params_obj=(Q, c), params_eq=(A, b)).params
#             delta_params = sol.primal


#             subarrays = np.split(delta_params, np.cumsum(sizes)[:-1])
#             reshaped_arrays = [subarray.reshape(shape) for subarray, shape in zip(subarrays, shapes)]
#             delta_params = pd.DataFrame(np.array(reshaped_arrays, dtype=object).\
#                         reshape(2,len(self.feature))).applymap(lambda x: x)
#             delta_params.columns = params["params"].keys()
#             delta_params.index = ["bias", "kernel"]
#             delta_params.sort_index(ascending=False, inplace=True)
#             delta_params = FrozenDict({"params": delta_params.to_dict()})


#             ls = BacktrackingLineSearch(fun=self.optim_components.obj, maxiter=100, condition="armijo",
#                                         decrease_factor=0.5)
#             stepsize, _ = ls.run(init_stepsize=0.8, params=params,
#                                     descent_direction=delta_params)
            
#             flatted_updated_params = stepsize * sol.primal + flatted_current_params

#             subarrays = np.split(flatted_updated_params, np.cumsum(sizes)[:-1])
#             reshaped_arrays = [subarray.reshape(shape) for subarray, shape in zip(subarrays, shapes)]
#             flatted_updated_params_df = pd.DataFrame(np.array(reshaped_arrays, dtype=object).\
#                         reshape(2,len(self.feature))).applymap(lambda x: x)
#             flatted_updated_params_df.columns = params["params"].keys()
#             flatted_updated_params_df.index = ["bias", "kernel"]
#             flatted_updated_params_df.sort_index(ascending=False, inplace=True)
#             params = FrozenDict({"params": flatted_updated_params_df.to_dict()})

#             print(self.optim_components.obj(params), stepsize)
#             # print(stepsize)


#         return params, [1.0]
        

#     def evaluation(self, params, N, data, ui):
#         u_theta = self.model.u_theta(params = params, data=data)
#         absolute_error = 1/N * jnp.linalg.norm(u_theta-ui)
#         l2_relative_error = 1/N * jnp.linalg.norm((u_theta-ui)/ui)
#         return absolute_error, l2_relative_error, u_theta
    









import time
start_time = time.time()

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
current_dir = os.getcwd().replace("\\", "/")
sys.path.append(parent_dir)

from PINN_experiment.optim_PINN import PINN
from l1_Penalty_experiment.optim_l1_penalty import l1Penalty
from l2_Penalty_experiment.optim_l2_penalty import l2Penalty
from linfinity_Penalty_experiment.optim_linfinity_penalty import linfinityPenalty
from Han_Penalty_experiment.optim_han_penalty import HanPenalty
from Cubic_Penalty_experiment.optim_cubic_penalty import CubicPenalty
from Augmented_Lag_experiment.optim_aug_lag import AugLag
from Pillo_Penalty_experiment.optim_pillo_penalty import PilloPenalty
# from SQP_experiment.optim_sqp import OptimComponents, SQP_Optim
from New_Augmented_Lag_experiment.optim_new_aug_lag import NewAugLag
from Fletcher_Penalty_experiment.optim_fletcher_penalty import FletcherPenalty

from data import Data
from NN import NN
from DataLoader import DataLoader
from Visualization import Visualization
from uncons_opt import Optim

from jax import random
import pandas as pd
from jax import numpy as jnp
from flax import linen as nn
from jaxopt import EqualityConstrainedQP

from multiprocessing import Pool




#######################################config for data#######################################
# beta_list = [10**-4, 30]
beta = 10**-4
N=3
M=4
data_key_num = 1000
dim = 2
Datas = Data(N=N, M=M, key_num=data_key_num, dim=dim)
dataloader = DataLoader(Data=Datas)

x_data_min = 0
x_data_max = 2*jnp.pi
t_data_min = 0
t_data_max = 1
x_sample_min = 0
x_sample_max = 2*jnp.pi
t_sample_min = 0
t_sample_max = 1

evaluation_data_key_num = 256
eval_Datas = Data(N=N, M=M, key_num=evaluation_data_key_num, dim=dim)
eval_dataloader = DataLoader(Data=eval_Datas)
####################################### config for data #######################################


####################################### config for NN #######################################
NN_key_num = 345
key = random.PRNGKey(NN_key_num)
# features = [50, 50, 50, 50, 1]
features = [2, 3, 1]
####################################### config for NN #######################################


####################################### config for penalty param #######################################
penalty_param_update_factor = 1.01
penalty_param = 1
####################################### config for penalty param #######################################


####################################### config for lagrange multiplier #######################################
mul = jnp.ones(2*M) # initial  for Han_penalty_experiment, Pillo_Penalty_experiment, Augmented_Lag_experiment, New_Augmented_Lag_experiment
mul_num_echos = 10 # for Pillo_Penalty_experiment
alpha = 150 # for New_Augmented_Lag_experiment
####################################### config for lagrange multiplier #######################################


####################################### config for Adam #######################################
uncons_optim_num_echos = 10
uncons_optim_learning_rate = 0.001
cons_violation = 10 # threshold for updating penalty param
####################################### config for Adam #######################################


####################################### visualization #######################################
visual = Visualization(current_dir)
####################################### visualization #######################################


####################################### config for SQP #######################################
qp = EqualityConstrainedQP()
SQP_num_iter = 4
hessiam_param = 0.5
init_stepsize = 1.0
line_search_max_iter = 20
line_search_condition = "strong-wolfe" #  armijo, goldstein, strong-wolfe or wolfe.
line_search_decrease_factor = 0.8
group_labels = list(range(1,2*M+1)) * 2
####################################### config for SQP #######################################

experiment = "SQP_experiment"
activation = jnp.sin
activation_name = activation.__name__

absolute_error_list = []
l2_relative_error_list = []

data, sample_data, IC_sample_data, ui = dataloader.get_data(\
    x_data_min, x_data_max, t_data_min, t_data_max, x_sample_min, \
        x_sample_max, t_sample_min, t_sample_max, beta, M)

model = NN(features=features, activation=activation)
params = model.init_params(key=key, data=data)
optim_components = OptimComponents(model, data, sample_data, IC_sample_data, ui[0], beta)

shapes = pd.DataFrame.from_dict(unfreeze(params["params"])).applymap(lambda x: x.shape).values.flatten()
sizes = [np.prod(shape) for shape in shapes]
for _ in tqdm(range(10)):

    gra_obj, gra_eq_cons = optim_components.get_grads(params=params)

    eq_cons = optim_components.eq_cons(params=params)
    current_obj = optim_components.obj(params=params)

    
    flatted_gra_obj = np.concatenate(pd.DataFrame.from_dict(unfreeze(gra_obj["params"])).\
                applymap(lambda x: x.flatten()).values.flatten())

    flatted_current_params = np.concatenate(pd.DataFrame.from_dict(unfreeze(params["params"])).\
                applymap(lambda x: x.flatten()).values.flatten())

    flatted_gra_eq_cons = np.concatenate(pd.DataFrame.from_dict(\
        unfreeze(gra_eq_cons['params'])).\
            apply(lambda x: x.explode()).set_index([group_labels]).\
                sort_index().applymap(lambda x: x.flatten()).values.flatten())

    Q = hessiam_param * jnp.identity(flatted_gra_obj.shape[0])
    c = flatted_gra_obj
    A = jnp.array(jnp.split(flatted_gra_eq_cons, 2*M))
    b = -eq_cons
    qp = EqualityConstrainedQP()
    flat_delta_params = qp.run(params_obj=(Q, c), params_eq=(A, b)).params.primal


    subarrays = np.split(flat_delta_params, np.cumsum(sizes)[:-1])
    reshaped_arrays = [subarray.reshape(shape) for subarray, shape in zip(subarrays, shapes)]
    flat_delta_params_df = pd.DataFrame(np.array(reshaped_arrays, dtype=object).\
                reshape(2,len(features))).applymap(lambda x: x)
    flat_delta_params_df.columns = params["params"].keys()
    flat_delta_params_df.index = ["bias", "kernel"]
    flat_delta_params_df.sort_index(ascending=False, inplace=True)
    delta_params = FrozenDict({"params": flat_delta_params_df.to_dict()})


    ls = BacktrackingLineSearch(fun=optim_components.obj, maxiter=20, condition="goldstein",
                                decrease_factor=0.8)
    stepsize, _ = ls.run(init_stepsize=1.0, params=params,
                            descent_direction=delta_params,
                                    value=current_obj, grad=gra_obj)

    flatted_updated_params = stepsize * flat_delta_params + flatted_current_params
    subarrays = np.split(flatted_updated_params, np.cumsum(sizes)[:-1])
    reshaped_arrays = [subarray.reshape(shape) for subarray, shape in zip(subarrays, shapes)]
    flatted_updated_params_df = pd.DataFrame(np.array(reshaped_arrays, dtype=object).\
                reshape(2,len(features))).applymap(lambda x: x)
    flatted_updated_params_df.columns = params["params"].keys()
    flatted_updated_params_df.index = ["bias", "kernel"]
    flatted_updated_params_df.sort_index(ascending=False, inplace=True)
    params = FrozenDict({"params": flatted_updated_params_df.to_dict()})
    print(params)
