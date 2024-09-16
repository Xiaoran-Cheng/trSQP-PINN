import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from System import Transport_eq, Reaction_Diffusion, Reaction, Burger

from jax import numpy as jnp
from jax import jacfwd, hessian
import numpy as np
import jax
import time
from scipy.linalg import cholesky
import numpy as np
from projection_methods import projections
import pandas as pd

from scipy.sparse import eye as speye
from projected_cg import modified_dogleg, projected_cg, box_intersections
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from Data import Data

from scipy.optimize import BFGS, SR1


class SQP_Optim:
    # def __init__(self, model, params, beta, data, pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, ui, N, eval_data, eval_ui, nu, rho, alpha, system, intermediate_data_frame_path) -> None:
    def __init__(self, model, params, beta, eval_data, eval_ui, nu, rho, alpha, system, intermediate_data_frame_path) -> None:
        self.model = model
        self.beta = beta
        # self.data = data
        # self.pde_sample_data = pde_sample_data
        # self.IC_sample_data = IC_sample_data
        # self.IC_sample_data_sol = IC_sample_data_sol
        # self.BC_sample_data_zero = BC_sample_data_zero
        # self.BC_sample_data_2pi = BC_sample_data_2pi
        # self.ui = ui
        # self.N = N
        shapes_and_sizes = [(p.shape, p.size) for p in jax.tree_util.tree_leaves(params)]
        self.shapes, self.sizes = zip(*shapes_and_sizes)
        self.indices = jnp.cumsum(jnp.array(self.sizes)[:-1])
        self.eval_data = eval_data
        self.eval_ui = eval_ui
        self.nu = nu
        self.rho = rho
        self.alpha = alpha
        self.system = system
        self.start_time = time.time()
        self.intermediate_data_frame_path = intermediate_data_frame_path
        self.total_l_k_loss_list = []
        self.total_eq_cons_loss_list = []
        self.kkt_residual = []
        self.absolute_error_iter, self.l2_relative_error_iter = [], []
        self.time_iter = []
        self.kkt_diff = []
        self.x_diff = []
        
        

    def obj(self, param_list, treedef, data, ui):
        params = self.unflatten_params(param_list, treedef)
        u_theta = self.model.u_theta(params=params, data=data)
        obj_value = jnp.mean(jnp.square(jnp.linalg.norm(u_theta - ui, ord=2)))
        return obj_value


    def grad_objective(self, param_list, treedef, data, ui):
        return jacfwd(self.obj, 0)(param_list, treedef, data, ui)


    def IC_cons(self, param_list, treedef, IC_sample_data, IC_sample_data_sol):
        params = self.unflatten_params(param_list, treedef)
        u_theta = self.model.u_theta(params=params, data=IC_sample_data)
        if self.system == "transport":
            return Transport_eq(beta=self.beta).solution(\
                IC_sample_data[:,0], IC_sample_data[:,1]) - u_theta
        elif self.system == "reaction_diffusion":
            # return Reaction_Diffusion(self.nu, self.rho).u0(self.IC_sample_data[:,0]) - u_theta
            return IC_sample_data_sol - u_theta
        elif self.system == "reaction":
            return Reaction(self.rho).u0(IC_sample_data[:,0]) - u_theta
        elif self.system == "burger":
            return Burger(self.alpha).u0(IC_sample_data[:,0]) - u_theta


    def BC_cons(self, param_list, treedef, BC_sample_data_zero, BC_sample_data_2pi):
        params = self.unflatten_params(param_list, treedef)
        u_theta_0 = self.model.u_theta(params=params, data=BC_sample_data_zero)
        u_theta_2pi = self.model.u_theta(params=params, data=BC_sample_data_2pi)
        return u_theta_2pi - u_theta_0
    
    
    def pde_cons(self, param_list, treedef, pde_sample_data):
        params = self.unflatten_params(param_list, treedef)
        if self.system == "transport":
            grad_x = jacfwd(self.model.u_theta, 1)(params, pde_sample_data)
            return Transport_eq(beta=self.beta).pde(jnp.diag(grad_x[:,:,0]),\
                jnp.diag(grad_x[:,:,1]))
        elif self.system == "reaction_diffusion":
            u_theta = self.model.u_theta(params=params, data=pde_sample_data)
            grad_x = jacfwd(self.model.u_theta, 1)(params, pde_sample_data)
            dudt = jnp.diag(grad_x[:,:,1])
            grad_xx = hessian(self.model.u_theta, 1)(params, pde_sample_data)
            du2dx2 = jnp.diag(jnp.diagonal(grad_xx[:, :, 0, :, 0], axis1=1, axis2=2))
            return Reaction_Diffusion(self.nu, self.rho).pde(dudt, du2dx2, u_theta)
        elif self.system == "reaction":
            u_theta = self.model.u_theta(params=params, data=pde_sample_data)
            grad_x = jacfwd(self.model.u_theta, 1)(params, pde_sample_data)
            dudt = jnp.diag(grad_x[:,:,1])
            return Reaction(self.rho).pde(dudt, u_theta)
        elif self.system == "burger":
            u_theta = self.model.u_theta(params=params, data=pde_sample_data)
            grad_x = jacfwd(self.model.u_theta, 1)(params, pde_sample_data)
            dudt = jnp.diag(grad_x[:,:,1])
            dudx = jnp.diag(grad_x[:,:,0])
            grad_xx = hessian(self.model.u_theta, 1)(params, pde_sample_data)
            du2dx2 = jnp.diag(jnp.diagonal(grad_xx[:, :, 0, :, 0], axis1=1, axis2=2))
            return Burger(self.alpha).pde(dudt, dudx, du2dx2, u_theta)
    

    
    def eq_cons(self, param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
        eq_cons = jnp.concatenate([self.IC_cons(param_list, treedef, IC_sample_data, IC_sample_data_sol), self.BC_cons(param_list, treedef, BC_sample_data_zero, BC_sample_data_2pi), self.pde_cons(param_list, treedef, pde_sample_data)])
        return eq_cons


    def grads_eq_cons(self, param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
        eq_cons_jac = jacfwd(self.eq_cons, 0)(param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
        return eq_cons_jac


    def flatten_params(self, params):
        flat_params_list, treedef = jax.tree_util.tree_flatten(params)
        return np.concatenate([param.ravel() for param in flat_params_list], axis=0), treedef


    def unflatten_params(self, param_list, treedef):
        param_groups = jnp.split(param_list, self.indices)
        reshaped_params = [group.reshape(shape) for group, shape in zip(param_groups, self.shapes)]
        return jax.tree_util.tree_unflatten(treedef, reshaped_params)
    

    def L(self, param_list, mul, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
        return self.obj(param_list, treedef, data, ui) + self.eq_cons(param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data) @ mul
    
    # def hessian_L(self, param_list, mul, treedef):
    #     return hessian(self.L, 0)(param_list, mul, treedef)
    
    def gradient_L(self, param_list, mul, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
        return jacfwd(self.L, 0)(param_list, mul, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data), \
            self.eq_cons(param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
            # jacfwd(self.L, 1)(param_list, mul, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)

    
    def SR1(self, xk, xk1, Bk, mulk1, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
    # def SR1(self, sk, yk, Bk):
        sk = xk1 - xk
        sk = sk.reshape(-1, 1)
        yk = jacfwd(self.L, 0)(xk1, mulk1, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data) \
            - jacfwd(self.L, 0)(xk, mulk1, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
        yk = yk.reshape(-1, 1)
        c1 = yk - Bk @ sk
        up = c1 @ c1.T
        do = c1.T @ sk
        Bk1 = Bk + jnp.divide(up, do)
        return Bk1


    def dBFGS(self, xk, xk1, Bk, mulk1, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
    # def dBFGS(self, sk, yk, Bk):
        sk = xk1 - xk
        sk = sk.reshape(-1, 1)
        yk = jacfwd(self.L, 0)(xk1, mulk1, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data) \
            - jacfwd(self.L, 0)(xk, mulk1, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
        yk = yk.reshape(-1, 1)
        syk = sk.T @ yk
        sBsk = sk.T @ Bk @ sk
        thetak = 1 if syk >= 0.2 * sBsk else jnp.divide(0.8 * sBsk, sBsk - syk)
        rk = thetak * yk + (1-thetak) * Bk @ sk
        Bk1 = Bk - jnp.divide(Bk @ sk @ sk.T @ Bk, sk.T @ Bk @ sk) + jnp.divide(rk @ rk.T, sk.T @ rk)
        return Bk1
    


    def Limited_hessian(self, length, S_k, Y_k, B0, method):
        L_k = jnp.zeros((length, length))
        D_k = jnp.zeros((length, length))
        for i in range(length):
            for j in range(i):
                if i > j:
                    # L_k[i, j] = jnp.dot(S_k[:,i], Y_k[:,j])
                    L_k = L_k.at[i, j].set(jnp.dot(S_k[:,i], Y_k[:,j]))
            # D_k[i, i] = jnp.dot(S_k[:,i].T, Y_k[:,i])
            D_k = D_k.at[i ,i].set(jnp.dot(S_k[:,i].T, Y_k[:,i]))
        if method == "LBFGS":
            middle_matrix = jnp.linalg.inv(jnp.block([[S_k.T @ B0 @ S_k, L_k.T],
                                                    [L_k, -D_k]]))
            H = B0 - jnp.block([B0 @ S_k, Y_k]).dot(middle_matrix).dot(jnp.block([[S_k.T @ B0], [Y_k.T]]))
        elif method == "LSR1":
            H = B0 + (Y_k - B0 @ S_k) @ jnp.linalg.inv(D_k + L_k + L_k.T - S_k.T @ B0 @ S_k) @ (Y_k - B0 @ S_k).T
        return H
        

    # def SQP_optim(self, params, mul, Bk, data, ui):
    #     param_list, treedef = self.flatten_params(params)
    #     c = self.grad_objective(param_list, treedef, data, ui)
    #     A = self.grads_eq_cons(param_list,treedef)
    #     b = self.eq_cons(param_list, treedef)
    #     H = Bk
    #     Z, _, Y = projections(A)
    #     x_delta = projected_cg(H, c, Z, Y, b)
    #     gradient_L = jnp.concatenate((self.gradient_L(param_list, mul, treedef)[0], self.gradient_L(param_list, mul, treedef)[1]), axis=0)
    #     return x_delta, jnp.linalg.norm(gradient_L, ord=2)


    def evaluation(self, params):
        u_theta = self.model.u_theta(params=params, data=self.eval_data)
        absolute_error = jnp.mean(jnp.abs(u_theta-self.eval_ui))
        l2_relative_error = jnp.power(jnp.power((u_theta-self.eval_ui), 2).sum(), 1/2) / jnp.power(jnp.power((self.eval_ui), 2).sum(), 1/2)
        return absolute_error, l2_relative_error, u_theta
    


    def default_scaling(self, x):
        n, = jnp.shape(x)
        return speye(n)


    def equality_constrained_sqp(self, B0,
                                x0, fun0, grad0, constr0,
                                jac0,
                                initial_penalty,
                                initial_trust_radius,
                                treedef,
                                Datas,
                                N,
                                data_key_num, 
                                sample_key_num,
                                X_star, 
                                eval_ui,
                                sqp_maxiter,
                                x_diff_tol,
                                kkt_tol,
                                hessian_method,
                                L_m,
                                init_data,
                                init_ui,
                                trust_lb=None,
                                trust_ub=None):
        PENALTY_FACTOR = 0.3
        LARGE_REDUCTION_RATIO = 0.9
        INTERMEDIARY_REDUCTION_RATIO = 0.3
        SUFFICIENT_REDUCTION_RATIO = 1e-8 
        TRUST_ENLARGEMENT_FACTOR_L = 7.0
        TRUST_ENLARGEMENT_FACTOR_S = 2.0
        MAX_TRUST_REDUCTION = 0.5
        MIN_TRUST_REDUCTION = 0.1
        SOC_THRESHOLD = 0.1
        TR_FACTOR = 0.8
        BOX_FACTOR = 0.5
        
        n, = jnp.shape(x0)

        if hessian_method == "dBFGS":
            bfgs = BFGS("damp_update")
            bfgs.initialize(n, 'hess')
        elif hessian_method == "SR1":
            sr1 = SR1()
            sr1.initialize(n, 'hess')

        

        if trust_lb is None:
            trust_lb = jnp.full(n, -jnp.inf)
        if trust_ub is None:
            trust_ub = jnp.full(n, jnp.inf)
        x = jnp.copy(x0)
        trust_radius = initial_trust_radius
        penalty = initial_penalty
        f = fun0
        c = grad0
        b = constr0
        A = jac0
        S = self.default_scaling(x)
        Z, LS, Y = projections(A)
        v = -LS.dot(c)
        H = B0
        start_time = time.time()


        S_k = jnp.empty((n, 0))
        Y_k = jnp.empty((n, 0))
        hessain_counter = 1
        last_iteration_failed = False
        gradient_average_counter = 0
        data, ui = init_data, init_ui
        x_diff_list = []
        kkt_list = []
        pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num, X_star, eval_ui)
        for iter in tqdm(range(sqp_maxiter)):
            # new_data, new_ui = Datas.generate_data(N, data_key_num, X_star, eval_ui)
            # pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num, X_star, eval_ui)
            dn = modified_dogleg(A, Y, b,
                                TR_FACTOR*trust_radius,
                                BOX_FACTOR*trust_lb,
                                BOX_FACTOR*trust_ub)

            c_t = H.dot(dn) + c
            b_t = jnp.zeros_like(b)
            trust_radius_t = jnp.sqrt(trust_radius**2 - jnp.linalg.norm(dn)**2)
            lb_t = trust_lb - dn
            ub_t = trust_ub - dn
            dt, _ = projected_cg(H, c_t, Z, Y, b_t,
                                    trust_radius_t,
                                    lb_t, ub_t)
            d = dn + dt
            quadratic_model = 1/2*(H.dot(d)).dot(d) + c.T.dot(d)
            linearized_constr = A.dot(d)+b
            vpred = norm(b) - norm(linearized_constr)
            vpred = max(1e-16, vpred)
            previous_penalty = penalty
            if quadratic_model > 0:
                new_penalty = quadratic_model / ((1-PENALTY_FACTOR)*vpred)
                penalty = max(penalty, new_penalty)
            predicted_reduction = -quadratic_model + penalty*vpred
            merit_function = f + penalty*norm(b)
            x_next = x + S.dot(d)
            f_next, b_next = self.obj(x_next, treedef, data, ui), self.eq_cons(x_next, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
            
            merit_function_next = f_next + penalty*norm(b_next)
            actual_reduction = merit_function - merit_function_next
            reduction_ratio = actual_reduction / predicted_reduction

            # if reduction_ratio < SUFFICIENT_REDUCTION_RATIO and \
            # norm(dn) <= SOC_THRESHOLD * norm(dt):

            #     y = -Y.dot(b_next)

            #     x_soc = x + S.dot(d + y)
            #     f_soc, b_soc = self.obj(x_soc, treedef, data, ui), self.eq_cons(x_soc, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)

            #     merit_function_soc = f_soc + penalty*norm(b_soc)
            #     actual_reduction_soc = merit_function - merit_function_soc

            #     reduction_ratio_soc = actual_reduction_soc / predicted_reduction
            #     if reduction_ratio_soc >= SUFFICIENT_REDUCTION_RATIO:
            #         x_next = x_soc
            #         f_next = f_soc
            #         b_next = b_soc
            #         reduction_ratio = reduction_ratio_soc
            
            if reduction_ratio >= LARGE_REDUCTION_RATIO:
                trust_radius = max(TRUST_ENLARGEMENT_FACTOR_L * norm(d),
                                  trust_radius)
            elif reduction_ratio >= INTERMEDIARY_REDUCTION_RATIO:
                trust_radius = max(TRUST_ENLARGEMENT_FACTOR_S * norm(d),
                                  trust_radius)
            elif reduction_ratio < SUFFICIENT_REDUCTION_RATIO:
                trust_reduction = ((1-SUFFICIENT_REDUCTION_RATIO) /
                                  (1-reduction_ratio))
                new_trust_radius = trust_reduction * norm(d)
                if new_trust_radius >= MAX_TRUST_REDUCTION * trust_radius:
                    trust_radius *= MAX_TRUST_REDUCTION
                elif new_trust_radius >= MIN_TRUST_REDUCTION * trust_radius:
                    trust_radius = new_trust_radius
                else:
                    trust_radius *= MIN_TRUST_REDUCTION


            prev_x = jnp.copy(x)
            if reduction_ratio >= SUFFICIENT_REDUCTION_RATIO:
                x = x_next
                S = self.default_scaling(x)
                last_iteration_failed = False
            else:
                penalty = previous_penalty
                last_iteration_failed = True

            
            params = self.unflatten_params(prev_x, treedef)
            absolute_error, l2_relative_error, _ = self.evaluation(params)
            self.absolute_error_iter.append(absolute_error)
            self.l2_relative_error_iter.append(l2_relative_error)
            self.total_l_k_loss_list.append(f)
            self.total_eq_cons_loss_list.append(jnp.linalg.norm(b, ord=2))
            self.time_iter.append(time.time() - start_time)


            f, b = self.obj(x, treedef, data, ui), self.eq_cons(x, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
            c, A = self.grad_objective(x, treedef, data, ui), self.grads_eq_cons(x,treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
            Z, LS, Y = projections(A)
            v = -LS.dot(c)

            gradient_L1 = self.gradient_L(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                                        BC_sample_data_2pi, pde_sample_data)[0]
            kkt1 = jnp.linalg.norm(gradient_L1, ord=jnp.inf)
            self.kkt_residual.append(kkt1)
            x_diff = jnp.linalg.norm(x - prev_x, ord=2)

            if last_iteration_failed == False:
                if x_diff <= x_diff_tol or kkt1 <= kkt_tol:
                    break
            else:
                if trust_radius <= x_diff_tol:
                    break

            self.x_diff.append(x_diff.item())

            



            if iter % 1 == 0:
                print("norm d: ", str(norm(d)))
                print("trust radius: ", str(trust_radius))
                print("condition number: ", str(jnp.linalg.cond(A)))
                print("obj value: ", str(f))
                print("eq_cons value: ", str(jnp.linalg.norm(b, ord = 2)))
                print("x_diff: ", str(x_diff))
                # print("kkt diff: ", str(kkt_diff))
                print("kkt: ", str(kkt1))
                print("penalty: ", str(penalty))
                print("reduction ratio", str(reduction_ratio))
                print("absolute_error: " + str(absolute_error))
                print("l2_relative_error: " + str(l2_relative_error))
                df_param = pd.DataFrame(x, columns=['params'])
                df_param.to_pickle(self.intermediate_data_frame_path+"SQP_params.pkl")

            # data, ui = jnp.copy(new_data), jnp.copy(new_ui)
            if last_iteration_failed == False:
              if hessian_method == "LBFGS" or hessian_method == "LSR1":
                  if hessain_counter <= L_m:
                      S_k = jnp.hstack([S_k, jnp.array(x-prev_x).reshape(n,1)])
                      Y_k = jnp.hstack([Y_k, jnp.array(self.gradient_L(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                                          BC_sample_data_2pi, pde_sample_data)[0] - self.gradient_L(prev_x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                                          BC_sample_data_2pi, pde_sample_data)[0]).reshape(n,1)])
                      B0 = jnp.identity(n)
                      H = self.Limited_hessian(hessain_counter, S_k, Y_k, B0, hessian_method)
                      hessain_counter += 1
                  
                  else:
                      S_k = jnp.hstack([S_k, jnp.array(x-prev_x).reshape(n,1)])
                      Y_k = jnp.hstack([Y_k, jnp.array(self.gradient_L(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                                          BC_sample_data_2pi, pde_sample_data)[0] - self.gradient_L(prev_x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                                          BC_sample_data_2pi, pde_sample_data)[0]).reshape(n,1)])
                      S_k = S_k[:, 1:]
                      Y_k = Y_k[:, 1:]

                      B0 = ((S_k[:,-1].T @ Y_k[:,-1]) / (Y_k[:,-1].T @ Y_k[:,-1])) * jnp.identity(n)
                      H = self.Limited_hessian(L_m, S_k, Y_k, B0, hessian_method)
                      hessain_counter += 1
              elif hessian_method == "dBFGS":
                  # if iter != 0:
                  #     print("averaging")
                  #     S_k = jnp.hstack([S_k, jnp.array(x-prev_x).reshape(n,1)])
                  #     Y_k = jnp.hstack([Y_k, jnp.array(self.gradient_L(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                  #                         BC_sample_data_2pi, pde_sample_data)[0] - self.gradient_L(prev_x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                  #                         BC_sample_data_2pi, pde_sample_data)[0]).reshape(n,1)])
                  # else:
                  #     print("hessian producing")
                  #     S_k = jnp.hstack([S_k, jnp.array(x-prev_x).reshape(n,1)])
                  #     Y_k = jnp.hstack([Y_k, jnp.array(self.gradient_L(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                  #                         BC_sample_data_2pi, pde_sample_data)[0] - self.gradient_L(prev_x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                  #                         BC_sample_data_2pi, pde_sample_data)[0]).reshape(n,1)])

                  #     H = self.dBFGS(S_k.mean(axis=1).reshape(-1, 1), Y_k.mean(axis=1).reshape(-1, 1), H)
                  #     S_k = jnp.empty((n, 0))
                  #     Y_k = jnp.empty((n, 0))
                  # gradient_average_counter += 1

                  H = self.dBFGS(prev_x, x, H, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
                  # delta_grad = jacfwd(self.L, 0)(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data) \
                  #     - jacfwd(self.L, 0)(prev_x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
                  # delta_x = x - prev_x
                  # H = bfgs.update(delta_x.reshape(-1, 1), delta_grad.reshape(-1, 1))
                  
              elif hessian_method == "SR1":
                  # if gradient_average_counter % L_m != 0 or gradient_average_counter == 0:
                  #     print("averaging")
                  #     S_k = jnp.hstack([S_k, jnp.array(x-prev_x).reshape(n,1)])
                  #     Y_k = jnp.hstack([Y_k, jnp.array(self.gradient_L(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                  #                         BC_sample_data_2pi, pde_sample_data)[0] - self.gradient_L(prev_x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                  #                         BC_sample_data_2pi, pde_sample_data)[0]).reshape(n,1)])
                  # else:
                  #     print("hessian producing")
                  #     S_k = jnp.hstack([S_k, jnp.array(x-prev_x).reshape(n,1)])
                  #     Y_k = jnp.hstack([Y_k, jnp.array(self.gradient_L(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                  #                         BC_sample_data_2pi, pde_sample_data)[0] - self.gradient_L(prev_x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, \
                  #                         BC_sample_data_2pi, pde_sample_data)[0]).reshape(n,1)])
                  #     H = self.SR1(S_k.mean(axis=1).reshape(-1, 1), Y_k.mean(axis=1).reshape(-1, 1), H)
                  #     S_k = jnp.empty((n, 0))
                  #     Y_k = jnp.empty((n, 0))
                  # gradient_average_counter += 1

                  H = self.SR1(prev_x, x, H, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
                  # delta_grad = jacfwd(self.L, 0)(x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data) \
                  #     - jacfwd(self.L, 0)(prev_x, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
                  # delta_x = x - prev_x
                  # H = sr1.update(delta_x.reshape(-1, 1), delta_grad.reshape(-1, 1))

              elif hessian_method == None:
                  pass
        return x
    
