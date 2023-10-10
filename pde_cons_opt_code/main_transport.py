import time
full_start_time = time.time()
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
current_dir = os.getcwd().replace("\\", "/")
sys.path.append(parent_dir)

# from jax.config import config
# config.update("jax_enable_x64", True)

from optim_PINN import PINN
# from optim_l1_penalty import l1Penalty
from optim_l2_penalty import l2Penalty
# from optim_linfinity_penalty import linfinityPenalty
from optim_aug_lag import AugLag
# from optim_pillo_penalty import PilloPenalty
# from optim_new_aug_lag import NewAugLag
# from optim_fletcher_penalty import FletcherPenalty
from optim_pillo_aug_lag import PilloAugLag
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

error_df_list = []
beta_list = [0.001, 0.0001]
iteration_point_check_convergence = [3,9,12]
for beta in beta_list:
    #######################################config for pre_train######################################
    Pre_Train = False                                                         #check
    pretrain_maxiter = 5000000
    pretrain_gtol = 1e-9
    pretrain_ftol = 1e-9
    #######################################config for pre_train#######################################

    #######################################config for data#######################################
    # beta = 0.001
    nu = 3
    rho = 5
    alpha = 10

    xgrid = 256
    nt = 100
    N=10
    IC_M, pde_M, BC_M = 1,1,1                               #check
    M = IC_M + pde_M + BC_M
    data_key_num, sample_key_num = 100,256
    # data_key_num, sample_key_num = 23312,952
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
    # features = [50,50,50,50,1]                                                #check
    features = [2,1]                                                #check
    ###################################### config for NN #######################################

    ####################################### config for unconstrained optim #######################################
    LBFGS_maxiter = 10
    max_iter_train = 3                                                       #check

    penalty_param_update_factor = 2
    init_penalty_param = 1                                                    #check
    panalty_param_upper_bound = penalty_param_update_factor**max_iter_train

    init_penalty_param_mu = 10
    init_penalty_param_v = 10**-2

    LBFGS_gtol = 1e-9
    LBFGS_ftol = 1e-9

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
    sqp_initial_constr_penalty = 0.05
    sqp_initial_tr_radius = 1
    ####################################### config for SQP #######################################


    activation_input = "tanh"

    if activation_input == "sin":
        activation = jnp.sin
    elif activation_input == "tanh":
        activation = nn.tanh

    activation_name = activation.__name__
    model = NN(features=features, activation=activation)
    absolute_error_list = []
    l2_relative_error_list = []
    Datas = Data(N, IC_M, pde_M, BC_M, xgrid, nt, x_min, x_max, t_min, t_max, beta, noise_level, nu, rho, alpha, system)
    data, ui = Datas.generate_data(data_key_num)
    pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num)
    eval_data, eval_ui = Datas.get_eval_data()
    color_bar_bounds = [eval_ui.min(), eval_ui.max()]
    params = model.init_params(NN_key_num=NN_key_num, data=data)

    # print(pde_sample_data)
    # print(IC_sample_data)
    # print(BC_sample_data_zero)
    # print(BC_sample_data_2pi)
    if Pre_Train:
        absolute_error_list = l2_relative_error_list = []
        pretrain = PreTrain(model, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, beta, eval_data, eval_ui[0], nu, rho, alpha, system)
        params = pretrain.update(params, pretrain_maxiter, pretrain_gtol, pretrain_ftol)
        absolute_error, l2_relative_error, eval_u_theta = pretrain.evaluation(\
                                    params, eval_data, eval_ui[0])
        absolute_error_list.append(absolute_error)
        l2_relative_error_list.append(l2_relative_error)
        print("absolute_error: " + str(absolute_error))
        print("l2_relative_error: " + str(l2_relative_error))
        print("pretrain_loss_list: " + str(pretrain.pretrain_loss_list[-1]))
        visual.line_graph(pretrain.pretrain_loss_list, "beta_test_{beta}".format(beta=beta), "Pre_Train_Loss", experiment='Pre_Train')
        visual.line_graph(eval_u_theta, "beta_test_{beta}".format(beta=beta), "u_theta_line", experiment="Pre-Train")
        # visual.line_graph(pretrain.absolute_error_pretrain_list, "absolute_error", experiment="Pre-Train")
        # visual.line_graph(pretrain.l2_relative_error_pretrain_list, "l2_relative_error", experiment="Pre-Train")
        visual.heatmap(eval_data, eval_u_theta, "beta_test_{beta}".format(beta=beta), "u_theta_heatmap", experiment='Pre_Train', nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)
        visual.heatmap(eval_data, eval_ui[0], "beta_test_{beta}".format(beta=beta), "True_sol", experiment='Pre_Train', nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)
        flat_params, treedef = flatten_params(params)
        pd.DataFrame(flat_params, columns=['params']).\
        to_csv("params_303030_L2.csv", index=False)                        #check

    shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
    shapes, sizes = zip(*shapes_and_sizes)
    indices = jnp.cumsum(jnp.array(sizes)[:-1])
    _, treedef = flatten_params(params)


    # experiment_list = ['SQP_experiment',
    #                     'Pillo_Aug_Lag_experiment',
    #                     'l2^2_Penalty_experiment', 
    #                     'l2_Penalty_experiment', 
    #                     'Augmented_Lag_experiment']

    # experiment_list = ['l2^2_Penalty_experiment', 'SQP_experiment']
    experiment_list = ['SQP_experiment', 'l2^2_Penalty_experiment']

    for experiment in experiment_list:
        print(experiment)
        #############
        params = model.init_params(NN_key_num=NN_key_num, data=data)        #check
        # params = pd.read_csv("params_303030_L2.csv").values.flatten()      #check
        # params = unflatten_params(params, treedef)                          #check
        #############
        params_mul = {"params": params, "mul":init_mul}
        penalty_param = init_penalty_param
        penalty_param_v = init_penalty_param_v
        penalty_param_mu = init_penalty_param_mu
        mul = init_mul
        if experiment == "SQP_experiment":
            loss_values = []
            eq_cons_loss_values = []
            kkt_residual = []
            sqp_optim = SQP_Optim(model, params, beta, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui, N, eval_data, eval_ui, nu, rho, alpha, system)
            params = sqp_optim.SQP_optim(params, loss_values, eq_cons_loss_values, kkt_residual, sqp_maxiter, sqp_hessian, sqp_gtol, sqp_xtol, sqp_initial_constr_penalty, sqp_initial_tr_radius)
            total_l_k_loss_list = [i.item() for i in loss_values if isinstance(i, xla.ArrayImpl)]
            total_eq_cons_loss_list = [i.item() for i in eq_cons_loss_values if isinstance(i, xla.ArrayImpl)]
            kkt_residual_list = [i.item() for i in kkt_residual if isinstance(i, xla.ArrayImpl)]
            absolute_error_iter = [i.item() for i in sqp_optim.absolute_error_iter if isinstance(i, xla.ArrayImpl)]
            l2_relative_error_iter = [i.item() for i in sqp_optim.l2_relative_error_iter if isinstance(i, xla.ArrayImpl)]
            time_iter = sqp_optim.time_iter
            absolute_error, l2_relative_error, eval_u_theta = \
                sqp_optim.evaluation(params)
            
        else:
            if experiment == "l2^2_Penalty_experiment":                           # check
                loss = PINN(model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
                            N, nu, rho, alpha, system)
            # elif experiment == "l1_Penalty_experiment":
            #     loss = l1Penalty(model, data, pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
            #                 N)
            elif experiment == "l2_Penalty_experiment":
                loss = l2Penalty(model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
                            N, nu, rho, alpha, system)
            # elif experiment == "linfinity_Penalty_experiment":
            #     loss = linfinityPenalty(model, data, pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
            #                 N)
            elif experiment == "Augmented_Lag_experiment":
                loss = AugLag(model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
                            N, nu, rho, alpha, system)
            # elif experiment == "Pillo_Penalty_experiment":
            #     loss = PilloPenalty(model, data, pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
            #                 N, M)
            # elif experiment == "New_Augmented_Lag_experiment":
            #     loss = NewAugLag(model, data, pde_sample_data, IC_sample_data, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
            #                 N, M)
            # elif experiment == "Fletcher_Penalty_experiment":
            #     loss = FletcherPenalty(model, data, pde_sample_data, IC_sample_data, ui[0], beta, \
            #                 N)
            elif experiment == "Pillo_Aug_Lag_experiment":
                loss = PilloAugLag(model, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui[0], beta, \
                            N, nu, rho, alpha, system)
            
            total_loss_list, total_eq_cons_loss_list, total_l_k_loss_list, absolute_error_iter, l2_relative_error_iter, time_iter = [], [], [], [], [], []
            if experiment == "Augmented_Lag_experiment":
                def callback_func(params):
                    total_loss_list.append(loss.loss(params, mul, penalty_param).item())
                    total_l_k_loss_list.append(loss.l_k(params).item())
                    total_eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(loss.eq_cons(params), ord=2)).item())
                    u_theta = model.u_theta(params=params, data=eval_data)
                    absolute_error_iter.append(jnp.mean(np.abs(u_theta-eval_ui)))
                    l2_relative_error_iter.append(jnp.linalg.norm((u_theta-eval_ui[0]), ord = 2) / jnp.linalg.norm((eval_ui[0]), ord = 2))
                    time_iter.append(time.time() - start_time)
            elif experiment == "Pillo_Aug_Lag_experiment":
                def callback_func(params_mul):
                    params = params_mul['params']
                    total_loss_list.append(loss.loss(params_mul, penalty_param_mu, penalty_param_v).item())
                    total_l_k_loss_list.append(loss.l_k(params).item())
                    total_eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(loss.eq_cons(params), ord=2)).item())
                    u_theta = model.u_theta(params=params, data=eval_data)
                    absolute_error_iter.append(jnp.mean(np.abs(u_theta-eval_ui)))
                    l2_relative_error_iter.append(jnp.linalg.norm((u_theta-eval_ui[0]), ord = 2) / jnp.linalg.norm((eval_ui[0]), ord = 2))
                    time_iter.append(time.time() - start_time)
            else:
                def callback_func(params):
                    total_loss_list.append(loss.loss(params, penalty_param).item())
                    total_l_k_loss_list.append(loss.l_k(params).item())
                    total_eq_cons_loss_list.append(jnp.square(jnp.linalg.norm(loss.eq_cons(params), ord=2)).item())
                    u_theta = model.u_theta(params=params, data=eval_data)
                    absolute_error_iter.append(jnp.mean(np.abs(u_theta-eval_ui)))
                    l2_relative_error_iter.append(jnp.linalg.norm((u_theta-eval_ui[0]), ord = 2) / jnp.linalg.norm((eval_ui[0]), ord = 2))
                    time_iter.append(time.time() - start_time)

            LBFGS_opt = jaxopt.ScipyMinimize(method='L-BFGS-B', \
                            fun=loss.loss, \
                            maxiter=LBFGS_maxiter, \
                            options={'gtol': LBFGS_gtol, 'ftol': LBFGS_ftol}, \
                            callback=callback_func)

            optim = Optim(model, loss)
            total_loss_list_pernalty_change, total_eq_cons_loss_list_pernalty_change, \
                total_l_k_loss_list_pernalty_change, absolute_error_pernalty_change, \
                    l2_relative_error_pernalty_change = [], [], [], [], []
            start_time = time.time()
            penalty_param_list = []
            penalty_param_list.append(penalty_param)
            for _ in tqdm(range(max_iter_train)):
                params, params_mul, eq_cons = \
                    optim.update(params, penalty_param, experiment, \
                                        mul, params_mul, \
                                        penalty_param_mu, \
                                        penalty_param_v, LBFGS_opt)

                if experiment == "Augmented_Lag_experiment":
                    mul = mul + penalty_param  * 2 * eq_cons
                    total_loss_list_pernalty_change.append(loss.loss(params, mul, penalty_param).item())
                    total_l_k_loss_list_pernalty_change.append(loss.l_k(params).item())
                    total_eq_cons_loss_list_pernalty_change.append(jnp.square(jnp.linalg.norm(loss.eq_cons(params), ord=2)).item())
                    u_theta = model.u_theta(params=params, data=eval_data)
                    absolute_error_pernalty_change.append(jnp.mean(np.abs(u_theta-eval_ui)))
                    l2_relative_error_pernalty_change.append(jnp.linalg.norm((u_theta-eval_ui[0]), ord = 2) / jnp.linalg.norm((eval_ui[0]), ord = 2))
                if penalty_param <= panalty_param_upper_bound and experiment != "Pillo_Aug_Lag_experiment":
                    penalty_param = penalty_param_update_factor * penalty_param
                    total_loss_list_pernalty_change.append(loss.loss(params, penalty_param).item())
                    total_l_k_loss_list_pernalty_change.append(loss.l_k(params).item())
                    total_eq_cons_loss_list_pernalty_change.append(jnp.square(jnp.linalg.norm(loss.eq_cons(params), ord=2)).item())
                    u_theta = model.u_theta(params=params, data=eval_data)
                    absolute_error_pernalty_change.append(jnp.mean(np.abs(u_theta-eval_ui)))
                    l2_relative_error_pernalty_change.append(jnp.linalg.norm((u_theta-eval_ui[0]), ord = 2) / jnp.linalg.norm((eval_ui[0]), ord = 2))
                if experiment == "Pillo_Aug_Lag_experiment" and penalty_param_mu <= panalty_param_upper_bound:
                    penalty_param_mu = penalty_param_update_factor * penalty_param_mu
                if experiment == "Pillo_Aug_Lag_experiment" and penalty_param_v >= 1/(2**20):
                    penalty_param_v = (1/penalty_param_update_factor) * penalty_param_v
                if experiment == "Pillo_Aug_Lag_experiment":
                    print("penalty_param_mu: ", str(penalty_param_mu), 'penalty_param_v: ', str(penalty_param_v))
                else:
                    print("penalty_param: ", str(penalty_param))

                print("Number of iterations:", str(len(total_loss_list)))
                penalty_param_list.append(penalty_param)

                # pd.DataFrame(flatten_params(params)[0], columns=['params']).\
                # to_csv("{current_dir}/result/params_{experiment}_{test}.csv".format(experiment=experiment, \
                #                                     test="beta_"+str(beta), current_dir=current_dir), index=False)                        #check

            absolute_error, l2_relative_error, eval_u_theta = optim.evaluation(\
                                            params, eval_data, eval_ui[0])
            

        if experiment != "SQP_experiment":
            visual.line_graph(total_loss_list, "beta_test_{beta}".format(beta=beta), "Total_Loss", experiment=experiment)
            visual.line_graph(absolute_error_pernalty_change, "beta_test_{beta}".format(beta=beta), "absolute_error_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
            visual.line_graph(l2_relative_error_pernalty_change, "beta_test_{beta}".format(beta=beta), "l2_relative_error_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
            visual.line_graph(total_loss_list_pernalty_change, "beta_test_{beta}".format(beta=beta), "total_loss_list_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
            visual.line_graph(total_l_k_loss_list_pernalty_change, "beta_test_{beta}".format(beta=beta), "total_l_k_loss_list_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
            visual.line_graph(total_eq_cons_loss_list_pernalty_change, "beta_test_{beta}".format(beta=beta), "total_eq_cons_loss_list_pernalty_change", experiment=experiment, x=penalty_param_list[:-1])
        
        visual.line_graph(absolute_error_iter, "beta_test_{beta}".format(beta=beta), "absolute_error_iter_obj_loss", experiment=experiment, x=total_l_k_loss_list)
        visual.line_graph(l2_relative_error_iter, "beta_test_{beta}".format(beta=beta), "l2_relative_error_iter_obj_loss", experiment=experiment, x=total_l_k_loss_list)
        visual.line_graph(absolute_error_iter, "beta_test_{beta}".format(beta=beta), "absolute_error_iter_eq_loss", experiment=experiment, x=total_eq_cons_loss_list)
        visual.line_graph(l2_relative_error_iter, "beta_test_{beta}".format(beta=beta), "l2_relative_error_iter_eq_loss", experiment=experiment, x=total_eq_cons_loss_list)
        visual.line_graph(total_eq_cons_loss_list, "beta_test_{beta}".format(beta=beta),"Total_eq_cons_Loss", experiment=experiment)
        visual.line_graph(total_l_k_loss_list, "beta_test_{beta}".format(beta=beta),"Total_l_k_Loss", experiment=experiment)
        if experiment == "SQP_experiment":
            visual.line_graph(kkt_residual_list, "beta_test_{beta}".format(beta=beta),"KKT_residual", experiment=experiment)
        visual.line_graph(eval_ui[0], "beta_test_{beta}".format(beta=beta),"True_sol_line", experiment="")
        visual.line_graph(eval_u_theta, "beta_test_{beta}".format(beta=beta),"u_theta_line", experiment=experiment)
        visual.heatmap(eval_data, eval_ui[0], "beta_test_{beta}".format(beta=beta),"True_sol_heatmap", experiment="True_sol", nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)
        visual.heatmap(eval_data, eval_u_theta, "beta_test_{beta}".format(beta=beta),"u_theta_heatmap", experiment=experiment, nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)
        visual.line_graph(absolute_error_iter, "beta_test_{beta}".format(beta=beta),"absolute_error_per_iter", experiment=experiment)
        visual.line_graph(l2_relative_error_iter, "beta_test_{beta}".format(beta=beta),"l2_relative_error", experiment=experiment)
        absolute_error_list.append(absolute_error)
        l2_relative_error_list.append(l2_relative_error)
        if experiment != "SQP_experiment":
            print("last loss: "+str(total_loss_list[-1]))
        if experiment == "SQP_experiment":
            print("last KKT residual: " + str(kkt_residual_list[-1]))
        print("absolute_error: " + str(absolute_error))
        print("l2_relative_error: " + str(l2_relative_error))
        print("total_l_k_loss_list: " + str(total_l_k_loss_list[-1]))
        print("total_eq_cons_loss_list: " + str(total_eq_cons_loss_list[-1]))

        data_frame_path = "{current_dir}/result/{test}/{experiment}_dataframes".format(experiment=experiment, \
                                                        test="beta_test_"+str(beta), current_dir=current_dir)
        check_path(data_frame_path)        
        pd.DataFrame(flatten_params(params)[0], columns=['params']).\
        to_csv(data_frame_path+"/params_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta), current_dir=current_dir), index=False)
        pd.DataFrame(total_l_k_loss_list).to_csv(data_frame_path+"/total_l_k_loss_list_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
        pd.DataFrame(total_eq_cons_loss_list).to_csv(data_frame_path+"/total_eq_cons_loss_list_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
        if experiment == "SQP_experiment":
            pd.DataFrame(kkt_residual_list).to_csv(data_frame_path+"/kkt_residual_list_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
        pd.DataFrame(absolute_error_iter).to_csv(data_frame_path+"/absolute_error_iter_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
        pd.DataFrame(l2_relative_error_iter).to_csv(data_frame_path+"/l2_relative_error_iter_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
        pd.DataFrame(time_iter).to_csv(data_frame_path+"/time_iter_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
        if experiment != "SQP_experiment":
            pd.DataFrame(total_loss_list_pernalty_change).to_csv(data_frame_path+"/total_loss_list_pernalty_change_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
            pd.DataFrame(total_loss_list).to_csv(data_frame_path+"/total_loss_list{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
            pd.DataFrame(total_l_k_loss_list_pernalty_change).to_csv(data_frame_path+"/total_l_k_loss_list_pernalty_change_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
            pd.DataFrame(total_eq_cons_loss_list_pernalty_change).to_csv(data_frame_path+"/total_eq_cons_loss_list_pernalty_change_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
            pd.DataFrame(absolute_error_pernalty_change).to_csv(data_frame_path+"/absolute_error_pernalty_change_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
            pd.DataFrame(l2_relative_error_pernalty_change).to_csv(data_frame_path+"/l2_relative_error_pernalty_change_{experiment}_{test}.csv".format(experiment=experiment, \
                                                    test="beta_test_"+str(beta)), index=False)
    
        error_df = pd.DataFrame({
                                'experiment': [experiment], \
                                'absolute_error': [absolute_error], \
                                'l2_relative_error': [l2_relative_error], \
                                'Beta': [beta], \
                                'iterations': [len(time_iter)], \
                                'time_usage':[time_iter[-1]], \
                                'iteration_point_check': [iteration_point_check_convergence], \
                                'iteration_point_check_time': [np.array(time_iter)[np.array(iteration_point_check_convergence) - 1]], \
                                'iteration_point_check_absolute_error': [np.array(absolute_error_iter)[np.array(iteration_point_check_convergence) - 1]], \
                                'iteration_point_check_l2_relative_error': [np.array(l2_relative_error_iter)[np.array(iteration_point_check_convergence) - 1]], \
                                'iteration_point_check_objective_value': [np.array(total_l_k_loss_list)[np.array(iteration_point_check_convergence) - 1]], \
                                'iteration_point_check_constraints_violation': [np.array(total_eq_cons_loss_list)[np.array(iteration_point_check_convergence) - 1]]
                                })
        
        error_df_list.append(error_df)


folder_path = "{current_dir}/result/metrics_table".format(current_dir=current_dir)
pd.concat(error_df_list).to_csv(folder_path+".csv", index=False)
end_time = time.time()
print(f"Execution Time: {(end_time - full_start_time)/60} minutes")

