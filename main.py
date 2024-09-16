import time

full_start_time = time.time()
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
current_dir = os.getcwd().replace("\\", "/")
sys.path.append(parent_dir)

from optim_PINN import PINN
from optim_aug_lag import AugLag
from optim_sqp import SQP_Optim

from Data import Data
from NN import NN
from Visualization import Visualization
from uncons_opt import Optim
from pre_train import PreTrain

import pandas as pd
from jax import numpy as jnp
from flax import linen as nn
import jax.numpy as jnp
import jaxopt
from tqdm import tqdm
import jax
import numpy as np
import pandas as pd
from numpy import False_, linalg as LA


def flatten_params(params):
    ''' flatten Pytree NN parameters '''
    flat_params_list, treedef = jax.tree_util.tree_flatten(params)
    return np.concatenate([param.ravel() for param in flat_params_list], axis=0), treedef


def unflatten_params(param_list, treedef):
    ''' unflatten Pytree NN parameters '''
    param_groups = jnp.split(param_list, indices)
    reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, shapes)]
    return jax.tree_util.tree_unflatten(treedef, reshaped_params)

def check_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def calculate_mlp_params(layers):
    total_params = 0
    total_params += (2 * layers[0]) + layers[0]
    for i in range(1, len(layers)):
        total_params += (layers[i-1] * layers[i]) + layers[i]
    total_params += (layers[-1] * 1) + 1
    return total_params

    

class TerminationCondition(Exception):
    ''' Construct termination condition for penalty and ALM '''
    pass



Pre_Train = False
pretrain_maxiter = 20000
pretrain_gtol = 1e-9
pretrain_ftol = 1e-9

error_df_list = []

beta = 30
nu = 2
rho = 20
alpha = 10

xgrid = 2560
nt = 1000
N = 1000

IC_M, pde_M, BC_M = 2,3,2
M = IC_M + pde_M + BC_M
data_key_num, sample_key_num = 100,256
x_min = 0
x_max = 2*jnp.pi
t_min = 0
t_max = 1
noise_level = 0.01
system = "reaction_diffusion"
NN_key_num = 345
features = [10,10,10,10,1]
num_params = calculate_mlp_params(features[:-1])

''' Hyperparameters for penalty and ALM '''
LBFGS_maxiter = 20000
max_iter_train = 101
penalty_param_update_factor = 1.1
init_penalty_param = 2
LBFGS_gtol = 1e-9
LBFGS_ftol = 1e-9
init_mul = jnp.zeros(M)


''' Hyperparameters for trSQP-PINN '''
x_diff_tol = 1e-9
kkt_tol = 1e-9
B0 = jnp.identity(num_params)
sqp_maxiter = 20000
L_m = 10
hessian_method = "SR1"
sqp_initial_constr_penalty = 1
sqp_initial_tr_radius = 1

visual = Visualization(current_dir)

activation_input = "tanh"

if activation_input == "sin":
    activation = jnp.sin
elif activation_input == "tanh":
    activation = nn.tanh
elif activation_input == "gelu":
    activation = nn.gelu
elif activation_input == "sigmoid":
    activation = nn.sigmoid
elif activation_input == "elu":
    activation = nn.elu
elif activation_input == "leaky_relu":
    activation = nn.leaky_relu
elif activation_input == "relu":
    activation = nn.relu
elif activation_input == "identity":
    def identity(x):
        return x
    activation = identity

activation_name = activation.__name__
model = NN(features=features, activation=activation)

test_now = "width_test_{width}".format(width=features[0])


''' Get labeled and unlabeled data points '''
x = jnp.arange(x_min, x_max, x_max/xgrid)
t = jnp.linspace(t_min, t_max, nt).reshape(-1, 1)
X, T = np.meshgrid(x, t)
X_star = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))


Datas = Data(IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, alpha, system)
eval_data, eval_ui = Datas.get_eval_data(X_star)
data, ui = Datas.generate_data(N, data_key_num, X_star, eval_ui)
pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num, X_star, eval_ui)
color_bar_bounds = [eval_ui.min(), eval_ui.max()]
params = model.init_params(NN_key_num=NN_key_num, data=data)
shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
shapes, sizes = zip(*shapes_and_sizes)
indices = jnp.cumsum(jnp.array(sizes)[:-1])
_, treedef = flatten_params(params)


''' Pretraning process '''
if Pre_Train:
    pretrain_path = "{current_dir}/pre_result/{test}/".format(\
                        test=test_now, current_dir=current_dir)
    pretrain = PreTrain(model, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, beta, eval_data, eval_ui[0], pretrain_gtol, pretrain_ftol, pretrain_maxiter, nu, rho, alpha, system, num_params, params)
    _ = pretrain.update(params, pretrain_maxiter, pretrain_gtol)
    params = unflatten_params(pretrain.params_list[-1], treedef)
    absolute_error, l2_relative_error, eval_u_theta = pretrain.evaluation(\
                                params, eval_data, eval_ui[0])
    print("absolute_error: " + str(absolute_error))
    print("l2_relative_error: " + str(l2_relative_error))
    print("pretrain_loss_list: " + str(pretrain.pretrain_loss_list[-1]))
    visual.line_graph(pretrain.pretrain_loss_list, test_now, "Pre_Train_Loss", experiment='Pre_Train', pretrain=True)
    visual.line_graph(eval_u_theta, test_now, "u_theta_line", experiment="Pre-Train", pretrain=True)
    visual.heatmap(eval_data, eval_u_theta, test_now, "u_theta_heatmap", experiment='Pre_Train', nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds, pretrain=True)
    visual.heatmap(eval_data, eval_ui[0], test_now, "True_sol", experiment='Pre_Train', nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds, pretrain=True)
    flat_params, treedef = flatten_params(params)
    df_param = pd.DataFrame(flat_params, columns=['params'])
    df_param.to_pickle(pretrain_path+f"pretrain_params_{test_now}.pkl")
    df_param.to_pickle(f"pretrain_params_{test_now}.pkl")
    pd.DataFrame({
                'experiment': "pre_train", \
                'absolute_error': [absolute_error], \
                'l2_relative_error': [l2_relative_error], \
                'width': [features[0]], \
                'sample_key_num': [sample_key_num],\
                }).to_csv(pretrain_path+"error_{test}.csv".format(test=test_now), index=False, mode="a")
    end_time = time.time()
    print(f"Execution Time: {(end_time - full_start_time)/60} minutes")

else:
    ''' Applying penalty, ALM and trSQP-PINN to solve PDEs'''

    experiment_list = ['penalty','ALM','SQP']
    for experiment in experiment_list:
        print(experiment)
        iteration_point_check_convergence = np.array([500,1000,1500,2000])

        data_frame_path = "{current_dir}/result/{test}/{experiment}_dataframes".format(experiment=experiment, \
                                                        test=test_now, current_dir=current_dir)
        check_path(data_frame_path)
        if experiment == "SQP":
            intermediate_data_frame_path = data_frame_path+"/intermediate_SQP_params/"
            check_path(intermediate_data_frame_path)
        
        # params = model.init_params(NN_key_num=NN_key_num, data=data) # random initialization for unpretrianed experiments

        ''' Read in the pretrained NN parameters '''
        init_path = f"pretrain_params_{test_now}.pkl"
        print(init_path)
        params = pd.read_pickle(init_path).values.flatten()
        params = unflatten_params(params, treedef)
        penalty_param = init_penalty_param
        mul = init_mul


        if experiment == "SQP":
            ''' Performing trSQP-PINN '''
            sqp_optim = SQP_Optim(model, params, beta, eval_data, eval_ui, nu, rho, alpha, system, intermediate_data_frame_path)
            ''' Get initial starting points '''
            x0 = flatten_params(params)[0]
            grad0 = sqp_optim.grad_objective(x0, treedef, data, ui)
            jac0 = sqp_optim.grads_eq_cons(x0,treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
            fun0 = sqp_optim.obj(x0, treedef, data, ui)
            constr0 =sqp_optim.eq_cons(x0,treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
            params_list = sqp_optim.equality_constrained_sqp(B0,
                                x0, fun0, grad0, constr0,
                                jac0,
                                sqp_initial_constr_penalty,
                                sqp_initial_tr_radius,
                                treedef, Datas, N, data_key_num, sample_key_num, X_star, eval_ui, \
                                sqp_maxiter, x_diff_tol, kkt_tol, hessian_method, L_m, data, ui)
            
            params = unflatten_params(params_list, treedef)
            
            absolute_error, l2_relative_error, eval_u_theta = \
                sqp_optim.evaluation(params)
            total_l_k_loss_list = sqp_optim.total_l_k_loss_list
            total_eq_cons_loss_list = sqp_optim.total_eq_cons_loss_list
            kkt_residual = sqp_optim.kkt_residual
            x_diffs = sqp_optim.x_diff
            absolute_error_iter, l2_relative_error_iter = sqp_optim.absolute_error_iter, sqp_optim.l2_relative_error_iter
            time_iter = sqp_optim.time_iter
        else:
            if experiment == "penalty":
                loss = PINN(model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
                            N, nu, rho, alpha, system)
            elif experiment == "ALM":
                loss = AugLag(model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
                            N, nu, rho, alpha, system)
            
            total_loss_list, total_eq_cons_loss_list, total_l_k_loss_list, absolute_error_iter, l2_relative_error_iter, time_iter = [], [], [], [], [], []
            optim = Optim(model, loss, LBFGS_maxiter, LBFGS_gtol, LBFGS_ftol, total_l_k_loss_list, \
                 total_eq_cons_loss_list, absolute_error_iter, l2_relative_error_iter, total_loss_list, time_iter, data, ui)
            params_list = [np.zeros(num_params)]
            if experiment == "ALM":
                def callback_func(params):
                    ''' For termination and recording optimization information '''
                    total_loss_list.append(loss.loss(params, mul, penalty_param).item())
                    total_l_k_loss_list.append(loss.l_k(params).item())
                    total_eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(loss.eq_cons(params), ord=2)).item())
                    u_theta = model.u_theta(params=params, data=eval_data)
                    absolute_error_iter.append(jnp.mean(np.abs(u_theta-eval_ui)))
                    l2_relative_error_iter.append(jnp.linalg.norm((u_theta-eval_ui[0]), ord = 2) / jnp.linalg.norm((eval_ui[0]), ord = 2))
                    time_iter.append(time.time() - start_time)
                    list_params, _ = flatten_params(params)
                    params_list.append(list_params)
                    params_diff = params_list[-1] - params_list[-2]
                    params_list.pop(0)
                    if jnp.linalg.norm(params_diff, ord=2) <= LBFGS_ftol:
                        optim.stop_optimization = True
                        raise Exception("Stopping criterion met")
                    
            else:
                def callback_func(params):
                    ''' For termination and recording optimization information '''
                    total_loss_list.append(loss.loss(params, penalty_param).item())
                    total_l_k_loss_list.append(loss.l_k(params).item())
                    total_eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(loss.eq_cons(params), ord=2)).item())
                    u_theta = model.u_theta(params=params, data=eval_data)
                    absolute_error_iter.append(jnp.mean(np.abs(u_theta-eval_ui)))
                    l2_relative_error_iter.append(jnp.linalg.norm((u_theta-eval_ui[0]), ord = 2) / jnp.linalg.norm((eval_ui[0]), ord = 2))
                    time_iter.append(time.time() - start_time)
                    list_params, _ = flatten_params(params)
                    params_list.append(list_params)
                    params_diff = params_list[-1] - params_list[-2]
                    params_list.pop(0)
                    if jnp.linalg.norm(params_diff, ord=2) <= LBFGS_ftol:
                        optim.stop_optimization = True
                        raise TerminationCondition("Stopping criterion met")
                        
            LBFGS_opt = jaxopt.ScipyMinimize(method='L-BFGS-B', \
                            fun=loss.loss, \
                            maxiter=LBFGS_maxiter, \
                            options={'gtol': LBFGS_gtol, 'ftol': 0}, \
                            callback=callback_func)      
            start_time = time.time()
            penalty_param_list = []
            penalty_param_list.append(penalty_param)
            ''' Outer  loop for penalty and ALM '''
            for rounds in tqdm(range(max_iter_train)):
                if experiment == "ALM":
                    print(mul)
                    mul_new = mul + penalty_param * loss.eq_cons(params)

                try:
                    _, _ = \
                        optim.update(params, penalty_param, experiment, \
                                            mul, LBFGS_opt)
                    params = unflatten_params(params_list[-1], treedef)
                    penalty_param = penalty_param_update_factor * penalty_param
                    if experiment == "ALM":
                        mul = mul_new
                    print("penalty_param: ", str(penalty_param/penalty_param_update_factor))
                    print("Number of iterations:", str(len(total_loss_list)))

                    absolute_error, l2_relative_error, eval_u_theta = optim.evaluation(\
                                                    params, eval_data, eval_ui[0])
                    print(f"absolute error: {absolute_error}")
                    print(f"relative error: {l2_relative_error}")
                    
                except Exception as ex: # check for termination conditiongs
                    params = unflatten_params(params_list[-1], treedef)
                    penalty_param = penalty_param_update_factor * penalty_param
                    if experiment == "ALM":
                        mul = mul_new
                    print("penalty_param: ", str(penalty_param/penalty_param_update_factor))
                    print("Number of iterations:", str(len(total_loss_list)))


                    absolute_error, l2_relative_error, eval_u_theta = optim.evaluation(\
                                                    params, eval_data, eval_ui[0])
                    print(f"absolute error: {absolute_error}")
                    print(f"relative error: {l2_relative_error}")
                    continue

        if experiment != "SQP":
            print("total_loss_list")
            visual.line_graph(total_loss_list, test_now, "Total_Loss", experiment=experiment)
        max_len = np.array([len(total_l_k_loss_list), len(total_eq_cons_loss_list), len(absolute_error_iter), len(l2_relative_error_iter)]).min()
        total_l_k_loss_list = total_l_k_loss_list[:max_len]
        total_eq_cons_loss_list = total_eq_cons_loss_list[:max_len]
        absolute_error_iter = absolute_error_iter[:max_len]
        l2_relative_error_iter = l2_relative_error_iter[:max_len]
        print("Total_eq_cons_Loss")
        visual.line_graph(total_eq_cons_loss_list, test_now,"Total_eq_cons_Loss", experiment=experiment)
        print("Total_l_k_Loss")
        visual.line_graph(total_l_k_loss_list, test_now,"Total_l_k_Loss", experiment=experiment)
        if experiment == "SQP":
            print("KKT_residual")
            visual.line_graph(kkt_residual, test_now,"KKT_residual", experiment=experiment)
            print("x_diff")
            visual.line_graph(x_diffs, test_now,"x_diffs", experiment=experiment)
        print("True_sol_line")
        visual.line_graph(eval_ui[0], test_now,"True_sol_line", experiment="")
        print("u_theta_line")
        visual.line_graph(eval_u_theta, test_now,"u_theta_line", experiment=experiment)
        visual.heatmap(eval_data, eval_ui[0], test_now,"True_sol_heatmap", experiment="True_sol", nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)
        visual.heatmap(eval_data, eval_u_theta, test_now,"u_theta_heatmap", experiment=experiment, nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)
        print("absolute_error_iter")
        visual.line_graph(absolute_error_iter, test_now,"absolute_error_per_iter", experiment=experiment)
        print("l2_relative_error_iter")
        visual.line_graph(l2_relative_error_iter, test_now,"l2_relative_error_per_iter", experiment=experiment)
        if experiment != "SQP":
            print("last loss: "+str(total_loss_list[-1]))
        if experiment == "SQP":
            print("last KKT residual: " + str(kkt_residual[-1]))
        print("absolute_error: " + str(absolute_error))
        print("l2_relative_error: " + str(l2_relative_error))
        print("total_l_k_loss_list: " + str(total_l_k_loss_list[-1]))
        print("total_eq_cons_loss_list: " + str(total_eq_cons_loss_list[-1]))


        pd.DataFrame(flatten_params(params)[0], columns=['params']).\
        to_pickle(data_frame_path+"/params_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now, current_dir=current_dir))
        pd.DataFrame(total_l_k_loss_list).to_pickle(data_frame_path+"/total_l_k_loss_list_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        pd.DataFrame(total_eq_cons_loss_list).to_pickle(data_frame_path+"/total_eq_cons_loss_list_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        if experiment == "SQP":
            pd.DataFrame(kkt_residual).to_pickle(data_frame_path+"/kkt_residual_list_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        pd.DataFrame(absolute_error_iter).to_pickle(data_frame_path+"/absolute_error_iter_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        pd.DataFrame(l2_relative_error_iter).to_pickle(data_frame_path+"/l2_relative_error_iter_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        pd.DataFrame(time_iter).to_pickle(data_frame_path+"/time_iter_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        if experiment != "SQP":
            pd.DataFrame(total_loss_list).to_pickle(data_frame_path+"/total_loss_list{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        iteration_point_check_convergence[iteration_point_check_convergence-1 > len(time_iter)] = len(time_iter)-1
        try:
            error_df = pd.DataFrame({
                                'experiment': [experiment], \
                                'absolute_error': [absolute_error], \
                                'l2_relative_error': [l2_relative_error], \
                                'width': [features[0]], \
                                'data_key_num': [data_key_num], \
                                'sample_key_num': [sample_key_num],\
                                'iterations': [len(time_iter)], \
                                'time_usage':[time_iter[-1]], \
                                'iteration_point_check': [iteration_point_check_convergence], \
                                'iteration_point_check_time': [np.array(time_iter)[iteration_point_check_convergence - 1]], \
                                'iteration_point_check_absolute_error': [np.array(absolute_error_iter)[iteration_point_check_convergence - 1]], \
                                'iteration_point_check_l2_relative_error': [np.array(l2_relative_error_iter)[iteration_point_check_convergence - 1]], \
                                'iteration_point_check_objective_value': [np.array(total_l_k_loss_list)[iteration_point_check_convergence - 1]], \
                                'iteration_point_check_constraints_violation': [np.array(total_eq_cons_loss_list)[iteration_point_check_convergence - 1]]
                                })
        except:
            error_df = pd.DataFrame({
                            'experiment': [experiment], \
                            'absolute_error': [absolute_error], \
                            'l2_relative_error': [l2_relative_error], \
                            'width': [features[0]], \
                            'data_key_num': [data_key_num], \
                            'sample_key_num': [sample_key_num],\
                            'iterations': [len(time_iter)], \
                            'time_usage':[time_iter[-1]], \
                            'iteration_point_check': [iteration_point_check_convergence], \
                            'iteration_point_check_time': [], \
                            'iteration_point_check_absolute_error': [], \
                            'iteration_point_check_l2_relative_error': [], \
                            'iteration_point_check_objective_value': [], \
                            'iteration_point_check_constraints_violation': []
                            })
        
        
        error_df.to_csv("{current_dir}/result/metrics_table.csv".format(current_dir=current_dir), index=False, mode="a")
        error_df_list.append(error_df)

    pd.concat(error_df_list).to_csv("{current_dir}/result/final_metrics_table.csv".format(current_dir=current_dir), index=False)
    end_time = time.time()
    print(f"Execution Time: {(end_time - full_start_time)/60} minutes")