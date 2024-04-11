import time
full_start_time = time.time()
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
current_dir = os.getcwd().replace("\\", "/")
sys.path.append(parent_dir)

from Data import Data
from NN import NN
from Visualization import Visualization
import pandas as pd
from jax import numpy as jnp
from flax import linen as nn
import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd


def flatten_params(params):
    flat_params_list, treedef = jax.tree_util.tree_flatten(params)
    return np.concatenate([param.ravel( ) for param in flat_params_list], axis=0), treedef


def unflatten_params(param_list, treedef, indices, shapes):
    param_groups = jnp.split(param_list, indices)
    reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, shapes)]
    return jax.tree_util.tree_unflatten(treedef, reshaped_params)


def check_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_params_dirs(problem):
    params_dirs = []
    params = os.path.join(current_dir, problem)
    for i in os.listdir(params):
        for j in os.listdir(os.path.join(params, i)):
            if "csv" in j:
                params_dirs.append(os.path.join(os.path.join(params, i), j))
    return params_dirs

xgrid = 256
nt = 10000
N=1000
IC_M, pde_M, BC_M = 2,2,2            
M = IC_M + pde_M + BC_M
data_key_num, sample_key_num = 23312,952
x_min = 0
x_max = 2*jnp.pi
t_min = 0
t_max = 1
noise_level = 0.005       
system = "reaction_diffusion"         
NN_key_num = 345
visual = Visualization(current_dir)
activation = nn.tanh
x = jnp.arange(x_min, x_max, x_max/xgrid)
t = jnp.linspace(t_min, t_max, nt).reshape(-1, 1)
X, T = np.meshgrid(x, t)
X_star = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))
if system == "reaction_diffusion":
    for i in get_params_dirs(system):
        method = "PINN" if "l2^2" in os.path.basename(i) else ("ALM" if "Augmented_Lag" in os.path.basename(i) else ("SQP" if "SQP" in os.path.basename(i) else "ERROR"))
        experiment_type = os.path.basename(os.path.dirname(i))
        experiment_config_nv = os.path.basename(i).rsplit("_", 1)[-1].replace(".csv", "")
        experiment_config_rho = os.path.basename(i).rsplit("_", 2)[1]
        experiment = os.path.basename(os.path.dirname(i)) + "_" + experiment_config_rho + "_" + experiment_config_nv
        pic_name = method + "_" + experiment
        beta = 30
        # # # beta = float(experiment_config) if experiment_type == "Varying_System_Complexity_Coefficient" else 30
        rho = float(experiment_config_rho) if experiment_type == "Varying_System_Complexity_Coefficient" else 20
        nu = float(experiment_config_nv) if experiment_type == "Varying_System_Complexity_Coefficient" else 2
        alpha = 10
        Datas = Data(N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, alpha, system)
        eval_data, eval_ui = Datas.get_eval_data(X_star)
        data, ui = Datas.generate_data(data_key_num, X_star, eval_ui)
        pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num, X_star, eval_ui)
        color_bar_bounds = [eval_ui.min(), eval_ui.max()]


        features = [50,1] if experiment == "NN_Depth_1" else \
                    ([50,50,1] if experiment == "NN_Depth_2" else \
                    ([50,50,50,1] if experiment == "NN_Depth_3" else \
                    ([50,50,50,50,1] if experiment == "NN_Depth_4" else \
                    ([10,10,10,10,1] if experiment == "NN_Width_10" else \
                    ([20,20,20,20,1] if experiment == "NN_Width_20" else \
                        ([30,30,30,30,1] if experiment == "NN_Width_30" else \
                        ([40,40,40,40,1] if experiment == "NN_Width_40" else \
                        ([50,50,50,50,1] if experiment == "NN_Width_50" else \
                        [50,50,50,50,1]))))))))

        model = NN(features=features, activation=activation)
        params = model.init_params(NN_key_num=NN_key_num, data=data)
        shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
        shapes, sizes = zip(*shapes_and_sizes)
        indices = jnp.cumsum(jnp.array(sizes)[:-1])
        _, treedef = flatten_params(params)

        params = pd.read_csv(i).values.flatten()
        params = unflatten_params(params, treedef, indices, shapes)
        eval_u_theta = model.u_theta(params=params, data=eval_data)
        visual.heatmap(eval_data, eval_u_theta, os.path.basename(os.path.dirname(i)), method, experiment=pic_name, nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds, figure_type=method)

        visual.heatmap(eval_data, eval_ui, os.path.basename(os.path.dirname(i)), "True_sol", experiment=f"True_sol_{rho}_{nu}", nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds, figure_type="True_sol")

else:
    for i in get_params_dirs(system):
        method = "PINN" if "l2^2" in os.path.basename(i) else ("ALM" if "Augmented_Lag" in os.path.basename(i) else ("SQP" if "SQP" in os.path.basename(i) else "ERROR"))
        experiment_type = os.path.basename(os.path.dirname(i))
        experiment_config = os.path.basename(i).rsplit("_", 1)[-1].replace(".csv", "")
        experiment = os.path.basename(os.path.dirname(i)) + "_" + os.path.basename(i).rsplit("_", 1)[-1].replace(".csv", "")
        pic_name = method + "_" + experiment
        beta = 30
        beta = float(experiment_config) if experiment_type == "Varying_System_Complexity_Coefficient" else 30
        rho = 20
        nu = 2
        alpha = 10
        Datas = Data(N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, alpha, system)
        data, ui = Datas.generate_data(data_key_num)
        pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num)
        eval_data, eval_ui = Datas.get_eval_data()
        color_bar_bounds = [eval_ui.min(), eval_ui.max()]


        features = [50,1] if experiment == "NN_Depth_1" else \
                    ([50,50,1] if experiment == "NN_Depth_2" else \
                    ([50,50,50,1] if experiment == "NN_Depth_3" else \
                    ([50,50,50,50,1] if experiment == "NN_Depth_4" else \
                    ([10,10,10,10,1] if experiment == "NN_Width_10" else \
                    ([20,20,20,20,1] if experiment == "NN_Width_20" else \
                        ([30,30,30,30,1] if experiment == "NN_Width_30" else \
                        ([40,40,40,40,1] if experiment == "NN_Width_40" else \
                        ([50,50,50,50,1] if experiment == "NN_Width_50" else \
                        [50,50,50,50,1]))))))))

        model = NN(features=features, activation=activation)
        params = model.init_params(NN_key_num=NN_key_num, data=data)
        shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
        shapes, sizes = zip(*shapes_and_sizes)
        indices = jnp.cumsum(jnp.array(sizes)[:-1])
        _, treedef = flatten_params(params)

        params = pd.read_csv(i).values.flatten()
        params = unflatten_params(params, treedef, indices, shapes)
        eval_u_theta = model.u_theta(params=params, data=eval_data)
        visual.heatmap(eval_data, eval_u_theta, os.path.basename(os.path.dirname(i)), method, experiment=pic_name, nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds, figure_type=method)

        visual.heatmap(eval_data, eval_ui, os.path.basename(os.path.dirname(i)), "True_sol", experiment=f"True_sol_{rho}", nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds, figure_type="True_sol")