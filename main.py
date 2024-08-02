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
import jaxlib.xla_extension as xla
import jaxopt
from tqdm import tqdm
from scipy.optimize import BFGS, SR1
import jax
import numpy as np
import pandas as pd
import random
from numpy import linalg as LA
from jax import jacfwd, hessian


def flatten_params(params):
    flat_params_list, treedef = jax.tree_util.tree_flatten(params)
    return np.concatenate([param.ravel() for param in flat_params_list], axis=0), treedef


def unflatten_params(param_list, treedef):
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

def make_positive_definite(matrix):
    if np.all(np.linalg.eigvals(matrix) > 0):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        eigenvalues[eigenvalues < 0] = 0
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    else:
        return matrix
    

class TerminationCondition(Exception):
    pass

#######################################config for pre_train######################################
Pre_Train = True                                                    #check                                                       #check
pretrain_maxiter = 20000
pretrain_gtol = 1e-9
pretrain_ftol = 1e-9
######################################config for pre_train#######################################

#######################################config for data#######################################
error_df_list = []

beta = 1
nu = 3
rho = 5
alpha = 10

xgrid = 256
nt = 100
N = 1000
# IC_M, pde_M, BC_M = 50,50,50                            #check
IC_M, pde_M, BC_M = 3,3,3
M = IC_M + pde_M + BC_M
data_key_num, sample_key_num = 100,1234
# data_key_num, sample_key_num = 1000,1234
# data_key_num, sample_key_num = 10000,1234
x_min = 0
x_max = 2*jnp.pi
t_min = 0
t_max = 1
noise_level = 0.01                                                       #check
system = "convection"                                            #check
####################################### config for data #######################################

####################################### config for NN #######################################
NN_key_num = 345
# NN_key_num = 7654
features = [50,50,50,50,1]  
# features = [10,10,1]                                            #check
num_params = calculate_mlp_params(features[:-1])
###################################### config for NN #######################################

####################################### config for unconstrained optim #######################################
LBFGS_maxiter = pretrain_maxiter
max_iter_train = 11                                                       #check

penalty_param_update_factor = 2
init_penalty_param = 1                                                    #check
panalty_param_upper_bound = penalty_param_update_factor**max_iter_train

init_penalty_param_mu = 10
init_penalty_param_v = 10**-2

LBFGS_gtol = pretrain_gtol
LBFGS_ftol = pretrain_ftol

init_mul = jnp.zeros(M)
####################################### config for unconstrained optim #####################################

####################################### visualization #######################################
visual = Visualization(current_dir)
####################################### visualization #######################################

####################################### config for SQP #######################################
sqp_maxiter = 10
sqp_hessian = SR1()
sqp_gtol = 1e-8
sqp_xtol = 1e-8
sqp_initial_constr_penalty = 0.5
sqp_initial_tr_radius = 1
####################################### config for SQP #######################################


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

test_now = "NN_depth_test_{depth}".format(depth=len(features) - 1)

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

if Pre_Train:

    pretrain_path = "{current_dir}/pre_result/{test}/".format(\
                        test=test_now, current_dir=current_dir)
    pretrain = PreTrain(model, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, beta, eval_data, eval_ui[0], pretrain_gtol, pretrain_ftol, num_params, nu, rho, alpha, system)
    _ = pretrain.update(params, pretrain_maxiter)
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
                'depth': [len(features) - 1], \
                }).to_csv(pretrain_path+"error_{test}.csv".format(test=test_now), index=False, mode="a")
    end_time = time.time()
    print(f"Execution Time: {(end_time - full_start_time)/60} minutes")

else:
    # experiment_list = ['SQP_experiment']
    # experiment_list = ['l2^2_Penalty_experiment','Augmented_Lag_experiment']
    experiment_list = ['l2^2_Penalty_experiment']

    for experiment in experiment_list:
        print(experiment)
        iteration_point_check_convergence = np.array([500,1000,1500,2000])

        data_frame_path = "{current_dir}/result/{test}/{experiment}_dataframes".format(experiment=experiment, \
                                                        test=test_now, current_dir=current_dir)
        check_path(data_frame_path)
        if experiment == "SQP_experiment":
            intermediate_data_frame_path = data_frame_path+"/intermediate_SQP_params/"
            check_path(intermediate_data_frame_path)
        #############
        # params = model.init_params(NN_key_num=NN_key_num, data=data)        #check

        init_path = f"pretrain_params_{test_now}.pkl"
        print(init_path)
        params = pd.read_pickle(init_path).values.flatten()     #check
        params = unflatten_params(params, treedef)
        #############
        penalty_param = init_penalty_param
        mul = init_mul

        if experiment == "SQP_experiment":
            x_diff_tol = 0.001
            kkt_tol = 0.5
            B0 = jnp.identity(num_params)
            maxiter = 2000
            L_m = 5
            hessian_method = "dBFGS"
            # reproduce_data_key = 100
            # random_data_keys = jax.random.randint(jax.random.PRNGKey(reproduce_data_key), (maxiter,), 1, 1000000001)
            sqp_optim = SQP_Optim(model, params, beta, eval_data, eval_ui, nu, rho, alpha, system, intermediate_data_frame_path)
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
                                maxiter, x_diff_tol, kkt_tol, hessian_method, L_m, data, ui)
            
            params = unflatten_params(params_list, treedef)
            
            absolute_error, l2_relative_error, eval_u_theta = \
                sqp_optim.evaluation(params)
            total_l_k_loss_list = sqp_optim.total_l_k_loss_list
            total_eq_cons_loss_list = sqp_optim.total_eq_cons_loss_list
            kkt_residual = sqp_optim.kkt_residual
            kkt_diffs = sqp_optim.kkt_diff
            x_diffs = sqp_optim.x_diff
            absolute_error_iter, l2_relative_error_iter = sqp_optim.absolute_error_iter, sqp_optim.l2_relative_error_iter
            time_iter = sqp_optim.time_iter
        else:
            if experiment == "l2^2_Penalty_experiment":                           # check
                loss = PINN(model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
                            N, nu, rho, alpha, system)
            elif experiment == "Augmented_Lag_experiment":
                loss = AugLag(model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
                            N, nu, rho, alpha, system)
            
            total_loss_list, total_eq_cons_loss_list, total_l_k_loss_list, absolute_error_iter, l2_relative_error_iter, time_iter = [], [], [], [], [], []
            optim = Optim(model, loss)
            params_list = [np.zeros(num_params)]
            if experiment == "Augmented_Lag_experiment":
                def callback_func(params):
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

                    loss_gradient, _ = flatten_params(jacfwd(loss.loss, 0)(params, mul, penalty_param))
                    if jnp.linalg.norm(loss_gradient, ord=2) <= LBFGS_gtol or jnp.linalg.norm(params_diff, ord=2) <= LBFGS_ftol:
                        optim.stop_optimization = True
                        raise Exception("Stopping criterion met")
        

            else:
                def callback_func(params):
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


                    loss_gradient, _ = flatten_params(jacfwd(loss.loss, 0)(params, penalty_param))
                    if jnp.linalg.norm(loss_gradient, ord=2) <= LBFGS_gtol or jnp.linalg.norm(params_diff, ord=2) <= LBFGS_ftol:
                        optim.stop_optimization = True
                        raise TerminationCondition("Stopping criterion met")
                        
                    


            LBFGS_opt = jaxopt.ScipyMinimize(method='L-BFGS-B', \
                            fun=loss.loss, \
                            maxiter=LBFGS_maxiter, \
                            options={'gtol': 0, 'ftol': 0}, \
                            callback=callback_func)

            
            # total_loss_list_pernalty_change, total_eq_cons_loss_list_pernalty_change, \
            #     total_l_k_loss_list_pernalty_change, absolute_error_pernalty_change, \
            #         l2_relative_error_pernalty_change = [], [], [], [], []
            start_time = time.time()
            penalty_param_list = []
            penalty_param_list.append(penalty_param)
            for _ in tqdm(range(max_iter_train)):
                if experiment == "Augmented_Lag_experiment":
                    print(mul)
                    mul_new = mul + penalty_param * 2 * loss.eq_cons(params)
                    # total_loss_list_pernalty_change.append(loss.loss(params, mul, penalty_param).item())
                # else:
                    # total_loss_list_pernalty_change.append(loss.loss(params, penalty_param).item())
                try:
                    _, _ = \
                        optim.update(params, penalty_param, experiment, \
                                            mul, LBFGS_opt)
                except Exception as ex:
                    print(str(ex))
                    params = unflatten_params(params_list[-1], treedef)
                    penalty_param = penalty_param_update_factor * penalty_param
                    # total_l_k_loss_list_pernalty_change.append(loss.l_k(params).item())
                    # total_eq_cons_loss_list_pernalty_change.append(jnp.square(jnp.linalg.norm(loss.eq_cons(params), ord=2)).item())
                    # u_theta = model.u_theta(params=params, data=eval_data)
                    # absolute_error_pernalty_change.append(jnp.mean(np.abs(u_theta-eval_ui)))
                    # l2_relative_error_pernalty_change.append(jnp.linalg.norm((u_theta-eval_ui[0]), ord = 2) / jnp.linalg.norm((eval_ui[0]), ord = 2))
                    if experiment == "Augmented_Lag_experiment":
                        mul = mul_new
                    print("penalty_param: ", str(penalty_param/penalty_param_update_factor))
                    print("Number of iterations:", str(len(total_loss_list)))
                    continue

            absolute_error, l2_relative_error, eval_u_theta = optim.evaluation(\
                                            params, eval_data, eval_ui[0])

        if experiment != "SQP_experiment":
            print("total_loss_list")
            visual.line_graph(total_loss_list, test_now, "Total_Loss", experiment=experiment)
            # print("absolute_error_pernalty_change")
            # visual.line_graph(absolute_error_pernalty_change, test_now, "absolute_error_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
            # print("l2_relative_error_pernalty_change")
            # visual.line_graph(l2_relative_error_pernalty_change, test_now, "l2_relative_error_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
            # print("total_loss_list_pernalty_change")
            # visual.line_graph(total_loss_list_pernalty_change, test_now, "total_loss_list_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
            # print("total_l_k_loss_list_pernalty_change")
            # visual.line_graph(total_l_k_loss_list_pernalty_change, test_now, "total_l_k_loss_list_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
            # print("total_eq_cons_loss_list_pernalty_change")
            # visual.line_graph(total_eq_cons_loss_list_pernalty_change, test_now, "total_eq_cons_loss_list_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])

        max_len = np.array([len(total_l_k_loss_list), len(total_eq_cons_loss_list), len(absolute_error_iter), len(l2_relative_error_iter)]).min()

        total_l_k_loss_list = total_l_k_loss_list[:max_len]
        total_eq_cons_loss_list = total_eq_cons_loss_list[:max_len]
        absolute_error_iter = absolute_error_iter[:max_len]
        l2_relative_error_iter = l2_relative_error_iter[:max_len]
        # print("absolute_error_iter_obj_loss")
        # visual.line_graph(absolute_error_iter, test_now, "absolute_error_iter_obj_loss", experiment=experiment, x=total_l_k_loss_list)
        # print("l2_relative_error_iter_obj_loss")
        # visual.line_graph(l2_relative_error_iter, test_now, "l2_relative_error_iter_obj_loss", experiment=experiment, x=total_l_k_loss_list)
        # print("absolute_error_iter_eq_loss")
        # visual.line_graph(absolute_error_iter, test_now, "absolute_error_iter_eq_loss", experiment=experiment, x=total_eq_cons_loss_list)
        # print("l2_relative_error_iter_eq_loss")
        # visual.line_graph(l2_relative_error_iter, test_now, "l2_relative_error_iter_eq_loss", experiment=experiment, x=total_eq_cons_loss_list)
        print("Total_eq_cons_Loss")
        visual.line_graph(total_eq_cons_loss_list, test_now,"Total_eq_cons_Loss", experiment=experiment)
        print("Total_l_k_Loss")
        visual.line_graph(total_l_k_loss_list, test_now,"Total_l_k_Loss", experiment=experiment)
        if experiment == "SQP_experiment":
            print("KKT_residual")
            visual.line_graph(kkt_residual, test_now,"KKT_residual", experiment=experiment)
            print("KKT_diff")
            visual.line_graph(kkt_diffs, test_now,"kkt_diffs", experiment=experiment)
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
        if experiment != "SQP_experiment":
            print("last loss: "+str(total_loss_list[-1]))
        if experiment == "SQP_experiment":
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
        if experiment == "SQP_experiment":
            pd.DataFrame(kkt_residual).to_pickle(data_frame_path+"/kkt_residual_list_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        pd.DataFrame(absolute_error_iter).to_pickle(data_frame_path+"/absolute_error_iter_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        pd.DataFrame(l2_relative_error_iter).to_pickle(data_frame_path+"/l2_relative_error_iter_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        pd.DataFrame(time_iter).to_pickle(data_frame_path+"/time_iter_{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
        if experiment != "SQP_experiment":
            # pd.DataFrame(total_loss_list_pernalty_change).to_pickle(data_frame_path+"/total_loss_list_pernalty_change_{experiment}_{test}.pkl".format(experiment=experiment, \
            #                                         test=test_now))
            pd.DataFrame(total_loss_list).to_pickle(data_frame_path+"/total_loss_list{experiment}_{test}.pkl".format(experiment=experiment, \
                                                    test=test_now))
            # pd.DataFrame(total_l_k_loss_list_pernalty_change).to_pickle(data_frame_path+"/total_l_k_loss_list_pernalty_change_{experiment}_{test}.pkl".format(experiment=experiment, \
            #                                         test=test_now))
            # pd.DataFrame(total_eq_cons_loss_list_pernalty_change).to_pickle(data_frame_path+"/total_eq_cons_loss_list_pernalty_change_{experiment}_{test}.pkl".format(experiment=experiment, \
            #                                         test=test_now))
            # pd.DataFrame(absolute_error_pernalty_change).to_pickle(data_frame_path+"/absolute_error_pernalty_change_{experiment}_{test}.pkl".format(experiment=experiment, \
            #                                         test=test_now))
            # pd.DataFrame(l2_relative_error_pernalty_change).to_pickle(data_frame_path+"/l2_relative_error_pernalty_change_{experiment}_{test}.pkl".format(experiment=experiment, \
            #                                         test=test_now))
        
        iteration_point_check_convergence[iteration_point_check_convergence-1 > len(time_iter)] = len(time_iter)-1
        try:
            error_df = pd.DataFrame({
                                'experiment': [experiment], \
                                'absolute_error': [absolute_error], \
                                'l2_relative_error': [l2_relative_error], \
                                'depth': [len(features) - 1], \
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
                            'depth': [len(features) - 1], \
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










