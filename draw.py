import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
current_dir = os.getcwd()
sys.path.append(parent_dir)

from Data import Data
from NN import NN
from Visualization import Visualization

from jax import numpy as jnp
from flax import linen as nn
import jax.numpy as jnp
import jax
import pandas as pd
import numpy as np


dicts = {'params_Augmented_Lag_experiment.csv': "ALM",
         'params_l2^2_Penalty_experiment.csv': "L2^2",
         'params_l2_Penalty_experiment.csv': "L2",
         'params_Pillo_Aug_Lag_experiment.csv': "ALP",
         'params_SQP_experiment.csv': "SQP",
         'params_PINN_experiment_1.csv': "PINN_1",
         'params_PINN_experiment_10.csv': "PINN_10",
         'params_PINN_experiment_100.csv': "PINN_100",
         'params_PINN_experiment_1000.csv': "PINN_1000",
         "params_505050_L2.csv": "Pre_Train"
}

beta = 30
nu = 2
rho = 20
alpha = 10

xgrid = 256
nt = 10000
N=1000
IC_M, pde_M, BC_M = 2,2,2                          #check
M = IC_M + pde_M + BC_M
data_key_num, sample_key_num = 100,256
x_min = 0
x_max = 2*jnp.pi
t_min = 0
t_max = 1
noise_level = 0.005                                                       #check
system = "reaction_diffusion"                                            #check

NN_key_num = 7654
features = [50,50,50,50,1]                      

visual = Visualization(current_dir)

def unflatten_params(param_list, treedef):
    param_groups = jnp.split(param_list, indices)
    reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, shapes)]
    return jax.tree_util.tree_unflatten(treedef, reshaped_params)

def flatten_params(params):
    flat_params_list, treedef = jax.tree_util.tree_flatten(params)
    return np.concatenate([param.ravel( ) for param in flat_params_list], axis=0), treedef


def get_params_dirs(problem):
    reaction_params_dirs = []
    reaction = os.path.join(current_dir, problem)
    for i in os.listdir(reaction):
        for j in os.listdir(os.path.join(reaction, i)):
            if "csv" in j:
                reaction_params_dirs.append(os.path.join(os.path.join(reaction, i), j))
    return reaction_params_dirs

activation = nn.tanh
activation_name = activation.__name__
model = NN(features=features, activation=activation)
Datas = Data(N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, alpha, system)
data, ui = Datas.generate_data(data_key_num)
eval_data, eval_ui = Datas.get_eval_data()
color_bar_bounds = [eval_ui.min(), eval_ui.max()]
params = model.init_params(NN_key_num=NN_key_num, data=data)
_, treedef = flatten_params(params)
shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
shapes, sizes = zip(*shapes_and_sizes)
indices = jnp.cumsum(jnp.array(sizes)[:-1])




for i in get_params_dirs(system):
    params = pd.read_csv(i).values.flatten()
    params = unflatten_params(params, treedef)
    eval_u_theta = model.u_theta(params=params, data=eval_data)
    pretrain_or_not_folder = os.path.basename(os.path.dirname(i))
    visual.heatmap(eval_data, eval_u_theta, pretrain_or_not_folder, experiment=dicts[os.path.basename(i)], nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds, figure_type=dicts[os.path.basename(i)])

visual.heatmap(eval_data, eval_ui[0], "True_sol", experiment="True_sol", nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds, figure_type = "True_sol")