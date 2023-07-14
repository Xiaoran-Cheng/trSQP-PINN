import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd
# from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
import pandas as pd
from flax.core.frozen_dict import FrozenDict, unfreeze
from scipy.optimize import minimize
import jaxlib.xla_extension as xla



class SQP_Optim:
    def __init__(self, model, feature, M, params, beta, data, sample_data, IC_sample_data, ui, N) -> None:
        self.model = model
        self.feature = feature
        self.M = M
        self.layer_names = params["params"].keys()
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.ui = ui
        self.N = N


    def obj(self, param_list, treedef, loss_values):
        params = self.unflatten_params(param_list, treedef)
        u_theta = self.model.u_theta(params=params, data=self.data)
        obj_value = 1 / self.N * jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
        loss_values.append(obj_value)
        return obj_value
    

    def grad_objective(self, param_list, treedef, loss_values):
        return jacfwd(self.obj, 0)(param_list, treedef, loss_values)


    def IC_cons(self, param_list, treedef):
        params = self.unflatten_params(param_list, treedef)
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        return Transport_eq(beta=self.beta).solution(\
            self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta
    
    
    def pde_cons(self, param_list, treedef):
        params = self.unflatten_params(param_list, treedef)
        grad_x = jacfwd(self.model.u_theta, 1)(params, self.sample_data)
        return Transport_eq(beta=self.beta).pde(jnp.diag(grad_x[:,:,0]),\
            jnp.diag(grad_x[:,:,1]))

    
    def eq_cons(self, param_list, treedef, eq_cons_loss_values):
        eq_cons = jnp.concatenate([self.IC_cons(param_list, treedef), self.pde_cons(param_list, treedef)])
        eq_cons_loss = jnp.square(jnp.linalg.norm(eq_cons, ord=2))
        eq_cons_loss_values.append(eq_cons_loss)
        return eq_cons
    

    def grads_eq_cons(self, param_list, treedef, eq_cons_loss_values):
        eq_cons_jac = jacfwd(self.eq_cons, 0)(param_list, treedef, eq_cons_loss_values)
        return eq_cons_jac

    def get_li_in_eq_cons_index(self, param_list, treedef, eq_cons_loss_values):
        eq_cons_jac = jacfwd(self.eq_cons, 0)(param_list, treedef, eq_cons_loss_values)
        li_in_cons_index = self.get_li_in_cons_index(eq_cons_jac, 1e-5)
        return li_in_cons_index


    def get_li_in_eq_cons(self, param_list, li_in_cons_index, treedef, eq_cons_loss_values):
        li_in_cons_index = self.get_li_in_eq_cons_index(param_list, treedef, eq_cons_loss_values)
        eq_cons = self.eq_cons(param_list, treedef, eq_cons_loss_values)
        return eq_cons[li_in_cons_index]


    def get_li_in_eq_grads(self, param_list, li_in_cons_index, treedef, eq_cons_loss_values):
        eq_cons_jac = jacfwd(self.eq_cons, 0)(param_list, treedef, eq_cons_loss_values)
        li_in_cons_index = self.get_li_in_eq_cons_index(param_list, treedef, eq_cons_loss_values)
        return eq_cons_jac[li_in_cons_index, :]


    def flatten_params(self, params):
        flat_params_list, treedef = jax.tree_util.tree_flatten(params)
        return np.concatenate([param.ravel( ) for param in flat_params_list], axis=0), treedef
    
    def flat_single_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
                        applymap(lambda x: x.flatten()).values.flatten())


    def unflatten_params(self, param_list, treedef):
        shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
        shapes, sizes = zip(*shapes_and_sizes)
        indices = jnp.cumsum(jnp.array(sizes)[:-1])
        param_groups = jnp.split(param_list, indices)
        reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, shapes)]
        return jax.tree_util.tree_unflatten(treedef, reshaped_params)


    def get_li_in_cons_index(self, mat, qr_ind_tol):
        _, R = jnp.linalg.qr(mat)
        independent = jnp.where(jnp.abs(R.diagonal()) > qr_ind_tol)[0]
        return independent


    def SQP_optim(self, params, loss_values, eq_cons_loss_values):
        flat_params, treedef = self.flatten_params(params)
        li_in_cons_index = self.get_li_in_eq_cons_index(flat_params, treedef, eq_cons_loss_values)
        constraints = {
            'type': 'eq',
            'fun': self.eq_cons,
            'jac': self.grads_eq_cons,
            'args': (treedef, eq_cons_loss_values)}
        solution = minimize(self.obj, \
                            flat_params, \
                            args=(treedef,loss_values), \
                            jac=self.grad_objective, \
                            method='trust-constr', \
                            options={'maxiter': 10000}, \
                            constraints=constraints)
        params_opt = self.unflatten_params(solution.x, treedef)
        print(solution)
        return params_opt
    

    def evaluation(self, params, N, data, ui):
        u_theta = self.model.u_theta(params=params, data=data)
        absolute_error = 1/N * jnp.linalg.norm(u_theta-ui, ord = 2)
        l2_relative_error = 1/N * (jnp.linalg.norm((u_theta-ui), ord = 2) / jnp.linalg.norm((ui), ord = 2))
        return absolute_error, l2_relative_error, u_theta
 









import time
start_time = time.time()

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
current_dir = os.getcwd().replace("\\", "/")
sys.path.append(parent_dir)


from data import Data
from NN import NN
from DataLoader import DataLoader

from jax import random
import pandas as pd
from jax import numpy as jnp
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, unfreeze
import numpy as np
import jax




#######################################config for data#######################################
# beta_list = [10**-4, 30]
beta = 10
xgrid = 256
nt = 100
N=100
M=5
data_key_num, sample_data_key_num = 100, 256
eval_data_key_num, eval_sample_data_key_num = 300, 756
dim = 2
Datas = Data(N=N, M=M, dim=dim)
dataloader = DataLoader(Data=Datas)

x_data_min = 0
x_data_max = 2*jnp.pi
t_data_min = 0
t_data_max = 1
x_sample_min = 0
x_sample_max = 2*jnp.pi
t_sample_min = 0
t_sample_max = 1

# evaluation_data_key_num = 256
# eval_Datas = Data(N=N, M=M, dim=dim)
# eval_dataloader = DataLoader(Data=eval_Datas)
####################################### config for data #######################################


####################################### config for NN #######################################
NN_key_num = 345
key = random.PRNGKey(NN_key_num)
# features = [50, 50, 50, 50, 1]
# features = [10, 10, 10, 10, 1] # 搭配 SQP_num_iter = 100， hessian_param = 0.6 # 0.6最好， init_stepsize = 1.0， line_search_tol = 0.001， line_search_max_iter = 30， line_search_condition = "strong-wolfe" ，line_search_decrease_factor = 0.8
features = [2, 3, 1]
####################################### config for NN #######################################

def get_recovered_dict(flatted_target, shapes, sizes):
    subarrays = np.split(flatted_target, np.cumsum(sizes)[:-1])
    reshaped_arrays = [subarray.reshape(shape) for subarray, shape in zip(subarrays, shapes)]
    flatted_target_df = pd.DataFrame(np.array(reshaped_arrays, dtype=object).\
                reshape(2,len(features))).applymap(lambda x: x)
    flatted_target_df.columns = ['Dense_0', 'Dense_1', 'Dense_2']
    flatted_target_df.index = ["bias", "kernel"]
    flatted_target_df.sort_index(ascending=False, inplace=True)
    recovered_target = FrozenDict({"params": flatted_target_df.to_dict()})
    return recovered_target
sizes = [2, 3, 1, 4, 6, 3]
shapes = [(2,), (3,), (1,), (2, 2), (2, 3), (3, 1)]



experiment = 'SQP_experiment'
activation = jnp.sin

activation_name = activation.__name__
model = NN(features=features, activation=activation)
absolute_error_list = []
l2_relative_error_list = []



data, sample_data, IC_sample_data, ui = dataloader.get_data(\
    xgrid, nt, x_data_min, x_data_max, t_data_min, t_data_max, \
        x_sample_min, x_sample_max, t_sample_min, t_sample_max, \
            beta, M, data_key_num, sample_data_key_num)

params = model.init_params(key=key, data=data)
eval_data, eval_ui = dataloader.get_eval_data(xgrid, nt, x_data_min, x_data_max, t_data_min, t_data_max, beta)
shapes = pd.DataFrame.from_dict(unfreeze(params["params"])).applymap(lambda x: x.shape).values.flatten()
sizes = [np.prod(shape) for shape in shapes]

loss_values = []
eq_cons_loss_values = []
sqp_optim = SQP_Optim(model, features, M, params, beta, data, sample_data, IC_sample_data, ui, N)
params = sqp_optim.SQP_optim(params, loss_values, eq_cons_loss_values)
pd.DataFrame(sqp_optim.flat_single_dict(params)).to_csv("params.csv", index = False)
# params = get_recovered_dict(jnp.array(pd.read_csv("ddd.csv").iloc[:,0].tolist()), shapes, sizes)
absolute_error, l2_relative_error, eval_u_theta = \
    sqp_optim.evaluation(params, N, eval_data, eval_ui[0])


total_loss_list = [i.item() for i in loss_values if isinstance(i, xla.ArrayImpl)]
total_eq_cons_loss_list = [i.item() for i in eq_cons_loss_values if isinstance(i, xla.ArrayImpl)]





from Visualization import Visualization
visual = Visualization(current_dir)
visual.line_graph(eval_ui[0], "True_sol_line", experiment="", activation="", beta=beta)
visual.line_graph(eval_u_theta, "u_theta_line", experiment=experiment, activation=activation_name, beta=beta)
visual.heatmap(eval_data, eval_ui[0], "True_sol_heatmap", experiment="", beta=beta, activation="", nt=nt, xgrid=xgrid)
visual.heatmap(eval_data, eval_u_theta, "u_theta_heatmap", experiment=experiment, activation=activation_name, beta=beta, nt=nt, xgrid=xgrid)
visual.line_graph(total_loss_list, "Total_Loss", experiment=experiment, activation=activation_name, beta=beta)
visual.line_graph(total_eq_cons_loss_list, "Total_eq_cons_Loss", experiment=experiment, activation=activation_name, beta=beta)
print("absolute_error: " + str(absolute_error))
print("l2_relative_error: " + str(l2_relative_error))
print("total_loss_list: " + str(total_loss_list))
print("total_eq_cons_loss_list: " + str(total_eq_cons_loss_list))
