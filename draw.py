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


file_path = []
path1 = os.path.join(current_dir, "params_for_figs").replace("\\", "/")
path2 = [path1 + "/" + i for i in os.listdir(path1)]
for i in path2:
    for j in os.listdir(i):
        path3 = i+"/"+j
        for k in os.listdir(path3):
            file_path.append(path3+"/"+k)

dicts = {'params_Augmented_Lag_experiment.csv': "ALM",
         'params_l2^2_Penalty_experiment.csv': "L2^2",
         'params_SQP_experiment.csv': "SQP",
         'params_PINN_experiment_1.csv': "PINN_1",
         'params_PINN_experiment_10.csv': "PINN_10",
         'params_PINN_experiment_100.csv': "PINN_100",
         'params_PINN_experiment_1000.csv': "PINN_1000",
         "params_505050_L2.csv": "Pre_Train",
         'params_303030_L2.csv': "Pre_Train"
}













from Data import Data
from NN import NN
from Visualization import Visualization
from uncons_opt import Optim
from pre_train import PreTrain

import pandas as pd
from jax import numpy as jnp
from flax import linen as nn
import jax.numpy as jnp
import jaxlib.xla_extension as xla
import jaxopt
from tqdm import tqdm
from scipy.optimize import BFGS, SR1
import jax
import numpy as np
import pandas as pd


def flatten_params(params):
    flat_params_list, treedef = jax.tree_util.tree_flatten(params)
    return np.concatenate([param.ravel( ) for param in flat_params_list], axis=0), treedef


def unflatten_params(param_list, treedef):
    param_groups = jnp.split(param_list, indices)
    reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, shapes)]
    return jax.tree_util.tree_unflatten(treedef, reshaped_params)

def check_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# method = file_path[-1]



rho = 20
beta = 30
nu = 2
alpha = 10

xgrid = 256
nt = 10000
N=1000
IC_M, pde_M, BC_M = 50,50,50 
M = IC_M + pde_M + BC_M
data_key_num, sample_key_num = 92640,72072   # pretrain nu 20

x_min = 0
x_max = 2*jnp.pi
t_min = 0
t_max = 1
noise_level = 0.005                                                       #check
system = "reaction_diffusion"                                            #check
NN_key_num = 7654
features = [50,50,50,50,1]                                                #check
LBFGS_maxiter = 100000000
max_iter_train = 11                                                       #check
LBFGS_gtol = 1e-9
LBFGS_ftol = 1e-9
init_mul = jnp.zeros(M)
visual = Visualization(current_dir)

activation = nn.tanh

activation_name = activation.__name__
model = NN(features=features, activation=activation)
x = jnp.arange(x_min, x_max, x_max/xgrid)
t = jnp.linspace(t_min, t_max, nt).reshape(-1, 1)
X, T = np.meshgrid(x, t)
X_star = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))

Datas = Data(N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, alpha, system)
eval_data, eval_ui = Datas.get_eval_data(X_star)
data, ui = Datas.generate_data(data_key_num, X_star, eval_ui)
pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num, X_star, eval_ui)
color_bar_bounds = [eval_ui.min(), eval_ui.max()]


for method in file_path:

    params = model.init_params(NN_key_num=NN_key_num, data=data)
    shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
    shapes, sizes = zip(*shapes_and_sizes)
    indices = jnp.cumsum(jnp.array(sizes)[:-1])
    _, treedef = flatten_params(params)
    params = pd.read_csv(method).values.flatten()
    params = unflatten_params(params, treedef)        

    eval_u_theta = model.u_theta(params=params, data=eval_data)

    visual.heatmap(eval_data, eval_ui[0], "","True_sol_heatmap", experiment="True_sol", nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)
    visual.heatmap(eval_data, eval_u_theta, "",os.path.basename(os.path.dirname(method)), experiment=dicts[os.path.basename(method)], nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)


