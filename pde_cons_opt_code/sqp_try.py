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
from jaxopt import BacktrackingLineSearch, HagerZhangLineSearch
import time
import jax


class SQP_Optim:
    def __init__(self, model, qp, feature, group_labels, hessian_param, M, params, beta, data, sample_data, IC_sample_data, ui, N, merit_func_penalty_param) -> None:
        self.model = model
        self.qp = qp
        self.feature = feature
        self.group_labels = group_labels
        self.hessian_param = hessian_param
        self.M = M
        self.layer_names = params["params"].keys()
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.ui = ui
        self.N = N
        self.merit_func_penalty_param = merit_func_penalty_param 


    def obj(self, params):
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
    

    def eq_cons_loss(self, params):
        # return jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))
        return jnp.linalg.norm(self.eq_cons(params), ord=1)


    def L(self, params, mul):
        return self.obj(params) + self.eq_cons(params) @ mul


    # def make_psd(self, mat):
    #     eigenvalues = jnp.linalg.eigvalsh(mat)
    #     min_eigenvalue = jnp.min(eigenvalues)
    #     if min_eigenvalue < 0.0:
    #         mat_psd = mat - 2 * min_eigenvalue * jnp.eye(mat.shape[0])
    #         return mat_psd
    #     else:
    #         return mat


    def is_psd(self, mat):
        eigenvalues = jnp.linalg.eigvalsh(mat)
        return jnp.all(eigenvalues > 0)
    

    def bfgs_hessian(self, paramsk, paramsk1, mulk1, step_size, p, Hk):
        mulk1 = mulk1.reshape(-1,1)
        p = p.reshape(-1,1)
        sk = step_size * p
        yk = self.flat_single_dict(jacfwd(self.L, 0)(paramsk1, mulk1)) - self.flat_single_dict(jacfwd(self.L, 0)(paramsk, mulk1))
        yk = yk.reshape(-1,1)
        skyk = sk.T @ yk
        skHksk = sk.T @ Hk @ sk
        if skyk >= 0.2 * skHksk:
            thetak = 1
        else:
            thetak = (0.8 * skHksk) / (skHksk - skyk)
        rk = thetak * yk + (1 - thetak) * Hk @ sk
        Hk1 = Hk - ((Hk @ sk @ sk.T @ Hk) / skHksk) + ((rk @ rk.T) / (sk.T @ rk))
        Hk1 = 0.5 * (Hk1 + Hk1.T)
        if self.is_psd(Hk1):
            return Hk1
        else:
            return Hk


    # def flat_single_dict(self, dicts):
    #     flat_params_list = pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
    #                     applymap(lambda x: x.flatten()).values.flatten()
    #     return jnp.concatenate([param.ravel() for param in flat_params_list])


    # def flat_multi_dict(self, dicts):
    #     flat_params_list = pd.DataFrame.from_dict(\
    #             unfreeze(dicts['params'])).\
    #                 apply(lambda x: x.explode()).set_index([self.group_labels]).\
    #                     sort_index().applymap(lambda x: x.flatten()).values.flatten()
    #     return jnp.concatenate([param.ravel() for param in flat_params_list])


    def flat_single_dict(self, dicts):
        return np.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
                        applymap(lambda x: x.flatten()).values.flatten())
    

    def flat_multi_dict(self, dicts, group_labels):
        return np.concatenate(pd.DataFrame.from_dict(\
                unfreeze(dicts['params'])).\
                    apply(lambda x: x.explode()).set_index([group_labels]).\
                        sort_index().applymap(lambda x: x.flatten()).values.flatten())


    def merit_func(self, params):
        return self.obj(params=params) + 0.5 * self.merit_func_penalty_param * self.eq_cons_loss(params)
        # flatted_gra_l_k = self.flat_single_dict(jacfwd(self.obj, 0)(params))
        # flatted_gra_eq_cons = jnp.array(jnp.split(self.flat_multi_dict(jacfwd(self.eq_cons, 0)(params)), 2*self.M))
        # lambdax = jnp.linalg.pinv(flatted_gra_eq_cons @ flatted_gra_eq_cons.T) @ flatted_gra_eq_cons @ flatted_gra_l_k
        # return self.obj(params) - lambdax.T @ self.eq_cons(params) + 0.5 * self.merit_func_penalty_param * self.eq_cons_loss(params)

    
    def get_li_in_cons_index(self, mat, qr_ind_tol):
        _, R = jnp.linalg.qr(mat)
        independent = jnp.where(jnp.abs(R.diagonal()) > qr_ind_tol)[0]
        return independent


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


    def SQP_optim(self, params, num_iter, maxiter, condition, decrease_factor, init_stepsize, line_search_tol, qr_ind_tol, init_mul):
        obj_list = []
        eq_con_list = []
        kkt_residual_list = []
        shapes = pd.DataFrame.from_dict(unfreeze(params["params"])).applymap(lambda x: x.shape).values.flatten()
        sizes = [np.prod(shape) for shape in shapes]
        updated_Hk = self.hessian_param * jnp.identity(sum(sizes))
        mulk = init_mul
        for _ in tqdm(range(num_iter)):
            gra_obj = jacfwd(self.obj, 0)(params)
            gra_eq_cons = jacfwd(self.eq_cons, 0)(params)
            eq_cons = self.eq_cons(params=params)
            flatted_gra_obj = self.flat_single_dict(gra_obj)
            flatted_current_params = self.flat_single_dict(params)
            flatted_gra_eq_cons = self.flat_multi_dict(gra_eq_cons, self.group_labels)

            c = flatted_gra_obj
            A = jnp.array(jnp.split(flatted_gra_eq_cons, 2*self.M))
            li_ind_index = self.get_li_in_cons_index(A, qr_ind_tol)
            A = A[li_ind_index, :]
            print(jnp.linalg.cond(A))
            b = -eq_cons[li_ind_index]
            li_d_index = jnp.sort(jnp.setdiff1d(jnp.arange(2*self.M), li_ind_index))

            try:
                Q = updated_Hk
                sol = self.qp.run(init_params=params, params_obj=(Q, c), params_eq=(A, b), params_ineq=None)
            except:
                Q = self.hessian_param * jnp.identity(sum(sizes))
                sol = self.qp.run(init_params=params, params_obj=(Q, c), params_eq=(A, b), params_ineq=None)
            else:
                Hk = updated_Hk

            flatted_delta_params = sol.params.primal
            kkt_residual = self.qp.l2_optimality_error(params=sol.params, params_obj=(Q, c), params_eq=(A, b), params_ineq=None)
            delta_params = self.get_recovered_dict(flatted_delta_params, shapes, sizes)









            # ls = BacktrackingLineSearch(fun=self.merit_func, maxiter=maxiter, condition=condition,
            #                             decrease_factor=decrease_factor, tol=line_search_tol)
            # stepsize, _ = ls.run(init_stepsize=init_stepsize, \
            #                      params=params,
            #                     descent_direction=delta_params)
            stepsize = 0.5
            # print(stepsize, flatted_delta_params.sum())
            flatted_updated_params = stepsize * flatted_delta_params + flatted_current_params
            updated_params = self.get_recovered_dict(flatted_updated_params, shapes, sizes)
            mul_lambda = sol.params.dual_eq
            mulk1 = mulk + stepsize * (mul_lambda - mulk)
            
            if len(li_d_index) != 0:
                for i, index in enumerate(li_d_index):
                    if i != 0:
                        mulk1 = jnp.insert(mulk1, index+1, 1.0)
                    else:
                        mulk1 = jnp.insert(mulk1, index, 1.0)


            # updated_Hk = self.bfgs_hessian(params, updated_params, mulk1, stepsize, flatted_delta_params, Hk)
            params = updated_params
            mulk = mulk1
            obj_list.append(self.obj(params))
            eq_con_list.append(self.eq_cons_loss(params))
            kkt_residual_list.append(kkt_residual)



        # pd.DataFrame(self.flat_single_dict(params)).to_csv("ddd.csv", index = False)
        
        return params, obj_list, eq_con_list, kkt_residual_list
        

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

from optim_PINN import PINN
from optim_l1_penalty import l1Penalty
from optim_l2_penalty import l2Penalty
from optim_linfinity_penalty import linfinityPenalty
from optim_aug_lag import AugLag
from optim_pillo_penalty import PilloPenalty
from optim_new_aug_lag import NewAugLag
from optim_fletcher_penalty import FletcherPenalty
from optim_bert_aug_lag import BertAugLag
# from optim_sqp import SQP_Optim

from data import Data
from NN import NN
from DataLoader import DataLoader
from Visualization import Visualization
from uncons_opt import Optim

from jax import random
import pandas as pd
from jax import numpy as jnp
from flax import linen as nn
from jaxopt import EqualityConstrainedQP, CvxpyQP, OSQP
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict, unfreeze
import numpy as np

from multiprocessing import Pool




#######################################config for data#######################################
# beta_list = [10**-4, 30]
beta_list = [10]
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
# features = [10, 10, 1] # 搭配 SQP_num_iter = 100， hessian_param = 0.6 # 0.6最好， init_stepsize = 1.0， line_search_tol = 0.001， line_search_max_iter = 30， line_search_condition = "strong-wolfe" ，line_search_decrease_factor = 0.8
features = [2, 3, 1]
####################################### config for NN #######################################


####################################### config for penalty param #######################################
penalty_param_update_factor = 2
init_penalty_param = 1
panalty_param_upper_bound = 150
uncons_optim_num_echos = 200
init_uncons_optim_learning_rate = 0.001
transition_steps = uncons_optim_num_echos
decay_rate = 0.9
end_value = 0.0001
transition_begin = 0
staircase = True
max_iter_train = 10
penalty_param_for_mul = 5
init_penalty_param_v = init_penalty_param
init_penalty_param_mu = init_penalty_param
####################################### config for penalty param #######################################


####################################### config for lagrange multiplier #######################################
init_mul = jnp.ones(2*M) # initial  for Pillo_Penalty_experiment, Augmented_Lag_experiment, New_Augmented_Lag_experiment
mul_num_echos = 10 # for Pillo_Penalty_experiment
alpha = 150 # for New_Augmented_Lag_experiment
####################################### config for lagrange multiplier #######################################


####################################### visualization #######################################
visual = Visualization(current_dir)
####################################### visualization #######################################


####################################### config for SQP #######################################
# qp = EqualityConstrainedQP(tol=0.001) # , refine_regularization=3, refine_maxiter=50
qp = CvxpyQP(solver='OSQP') # "OSQP", "ECOS", "SCS" , implicit_diff_solve=True
SQP_num_iter = 100
hessian_param = 0.6
init_stepsize = 1.0
line_search_tol = 0
line_search_max_iter = 100
line_search_condition = "armijo"  # armijo, goldstein, strong-wolfe or wolfe.
line_search_decrease_factor = 0.8
group_labels = list(range(1,2*M+1)) * 2
qr_ind_tol = 1e-5
merit_func_penalty_param = 2
####################################### config for SQP #######################################





# for experiment in ['PINN_experiment', \
#                     'l1_Penalty_experiment', \
#                     'l2_Penalty_experiment', \
#                     'linfinity_Penalty_experiment', \
#                     'Augmented_Lag_experiment', \    
#                     'Pillo_Penalty_experiment', \
#                     'New_Augmented_Lag_experiment',\
#                     'Fletcher_Penalty_experiment', \
#                     'Bert_Aug_Lag_experiment',\
#                     'SQP_experiment']:

error_df_list = []
for experiment in ['SQP_experiment']:

    # for activation_input in ['sin', \
    #                         'tanh', \
    #                         'cos']:
    for activation_input in ['sin']:

        if activation_input == "sin":
            activation = jnp.sin
        elif activation_input == "tanh":
            activation = nn.tanh
        elif activation_input == "cos":
            activation = jnp.cos
        elif activation_input == "identity":
            def identity(x):
                return x
            activation = identity

        activation_name = activation.__name__
        model = NN(features=features, activation=activation)
        absolute_error_list = []
        l2_relative_error_list = []

        lr_schedule = optax.exponential_decay(
        init_value=init_uncons_optim_learning_rate, 
        transition_steps = transition_steps, 
        decay_rate=decay_rate,
        end_value = end_value,
        transition_begin  = transition_begin,
        staircase = staircase)
        
        for beta in beta_list:
            data, sample_data, IC_sample_data, ui = dataloader.get_data(\
                xgrid, nt, x_data_min, x_data_max, t_data_min, t_data_max, \
                    x_sample_min, x_sample_max, t_sample_min, t_sample_max, \
                        beta, M, data_key_num, sample_data_key_num)
            

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
            
            params = model.init_params(key=key, data=data)
            # params = get_recovered_dict(jnp.array(pd.read_csv("params.csv").iloc[:,0].tolist())+0.1, shapes, sizes)

            params_mul = [params, init_mul]
            eval_data, eval_ui = dataloader.get_eval_data(xgrid, nt, x_data_min, x_data_max, t_data_min, t_data_max, beta)

            penalty_param = init_penalty_param
            penalty_param_v = init_penalty_param_v
            penalty_param_mu = init_penalty_param_mu
            uncons_optim_learning_rate = init_uncons_optim_learning_rate
            mul = init_mul
            
            if experiment == "SQP_experiment":
                sqp_optim = SQP_Optim(model, qp, features, group_labels, hessian_param, M, params, beta, data, sample_data, IC_sample_data, ui, N, merit_func_penalty_param)
                params, total_l_k_loss_list, total_eq_cons_loss_list, kkt_residual_list = sqp_optim.SQP_optim(params, SQP_num_iter, \
                                            line_search_max_iter, line_search_condition, \
                                            line_search_decrease_factor, init_stepsize, \
                                            line_search_tol, qr_ind_tol, mul)
                absolute_error, l2_relative_error, eval_u_theta = \
                    sqp_optim.evaluation(params, N, eval_data, eval_ui[0])
                
            else:
                if experiment == "PINN_experiment":
                    loss = PINN(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                elif experiment == "l1_Penalty_experiment":
                    loss = l1Penalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                elif experiment == "l2_Penalty_experiment":
                    loss = l2Penalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                elif experiment == "linfinity_Penalty_experiment":
                    loss = linfinityPenalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                elif experiment == "Augmented_Lag_experiment":
                    loss = AugLag(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                elif experiment == "Pillo_Penalty_experiment":
                    loss = PilloPenalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                elif experiment == "New_Augmented_Lag_experiment":
                    loss = NewAugLag(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                elif experiment == "Fletcher_Penalty_experiment":
                    loss = FletcherPenalty(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                elif experiment == "Bert_Aug_Lag_experiment":
                    loss = BertAugLag(model, data, sample_data, IC_sample_data, ui[0], beta, \
                                N, M)
                    
                
                optim = Optim(model, loss, panalty_param_upper_bound)
                total_loss_list = []
                total_eq_cons_loss_list = []
                total_l_k_loss_list = []
                iter_retrain = 1
                while iter_retrain <= max_iter_train:
                    params, params_mul, loss_list, uncons_optim_learning_rate, \
                    eq_cons_loss_list, l_k_loss_list, eq_cons = \
                        optim.adam_update(params, \
                                            uncons_optim_num_echos, \
                                            penalty_param, experiment, \
                                            mul, mul_num_echos, alpha, \
                                            lr_schedule, group_labels, \
                                            penalty_param_for_mul, \
                                            params_mul, \
                                            penalty_param_mu, \
                                            penalty_param_v)
                    iter_retrain+=1
                    uncons_optim_learning_rate = lr_schedule(uncons_optim_num_echos * iter_retrain)
                    if experiment == "Augmented_Lag_experiment":
                        mul = mul + penalty_param * 2 * eq_cons
                    if penalty_param < panalty_param_upper_bound:
                        penalty_param = penalty_param_update_factor * penalty_param
                    if experiment == "Bert_Aug_Lag_experiment" and penalty_param_mu < panalty_param_upper_bound:
                        penalty_param_mu = penalty_param_update_factor * penalty_param_mu
                    if experiment == "Bert_Aug_Lag_experiment" and penalty_param_v > 1/panalty_param_upper_bound:
                        penalty_param_v = (1/penalty_param_update_factor) * penalty_param_v

                    total_loss_list.append(loss_list)
                    total_eq_cons_loss_list.append(eq_cons_loss_list)
                    total_l_k_loss_list.append(l_k_loss_list)
                    if experiment != "Bert_Aug_Lag_experiment":
                        print("penalty param: ", str(penalty_param), "leanring rate: ", str(uncons_optim_learning_rate))
                    else:
                        print("penalty_param_mu: ", str(penalty_param_mu), ", ", "penalty_param_v: ", str(penalty_param_v))

                
                absolute_error, l2_relative_error, eval_u_theta = optim.evaluation(\
                                                params, N, eval_data, eval_ui[0])
                total_loss_list = jnp.concatenate(jnp.array(total_loss_list))
                total_eq_cons_loss_list = jnp.concatenate(jnp.array(total_eq_cons_loss_list))
                total_l_k_loss_list = jnp.concatenate(jnp.array(total_l_k_loss_list))

            if experiment != "SQP_experiment":
                visual.line_graph(total_loss_list, "Total_Loss", experiment=experiment, activation=activation_name, beta=beta)
            visual.line_graph(total_eq_cons_loss_list, "Total_eq_cons_Loss", experiment=experiment, activation=activation_name, beta=beta)
            visual.line_graph(total_l_k_loss_list, "Total_l_k_Loss", experiment=experiment, activation=activation_name, beta=beta)
            if experiment == "SQP_experiment":
                visual.line_graph(kkt_residual_list, "KKT_residual", experiment=experiment, activation=activation_name, beta=beta)

            
            visual.line_graph(eval_ui[0], "True_sol_line", experiment="", activation="", beta=beta)
            visual.line_graph(eval_u_theta, "u_theta_line", experiment=experiment, activation=activation_name, beta=beta)
            visual.heatmap(eval_data, eval_ui[0], "True_sol_heatmap", experiment="", beta=beta, activation="", nt=nt, xgrid=xgrid)
            visual.heatmap(eval_data, eval_u_theta, "u_theta_heatmap", experiment=experiment, activation=activation_name, beta=beta, nt=nt, xgrid=xgrid)

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

        error_df = pd.DataFrame({'Beta': beta_list, 'absolute_error': absolute_error_list, \
                                'l2_relative_error': l2_relative_error_list}).astype(float)
        error_df["activation"] = activation_name
        error_df["experiment"] = experiment
        error_df_list.append(error_df)
        folder_path = "{current_dir}/result/error".format(current_dir=current_dir)
        visual.error_graph(error_df, folder_path, experiment=experiment, activation=activation_name)

pd.concat(error_df_list).to_csv(folder_path+".csv")
end_time = time.time()
print(f"Execution Time: {end_time - start_time} seconds")



