import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from System import Transport_eq, Reaction_Diffusion, Reaction

from jax import numpy as jnp
from jax import jacfwd, hessian
import numpy as np
import jax
import time
import numpy as np
from projection_methods import projections
import pandas as pd

from scipy.sparse import eye as speye
from projected_cg import modified_dogleg, projected_cg
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm


class SQP_Optim:
    def __init__(self, model, params, beta, eval_data, eval_ui, nu, rho, alpha, system, intermediate_data_frame_path) -> None:
        ''' Define PDE system coefficients, and initialize optimization information holder '''
        self.model = model
        self.beta = beta
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
        ''' Convert the flatten Pytree NN parameters to Pytree format and define the empirical loss which is our objective function '''
        params = self.unflatten_params(param_list, treedef)
        u_theta = self.model.u_theta(params=params, data=data)
        obj_value = jnp.mean(jnp.square(jnp.linalg.norm(u_theta - ui, ord=2)))
        return obj_value


    def grad_objective(self, param_list, treedef, data, ui):
        ''' Compute objective function gradient '''
        return jacfwd(self.obj, 0)(param_list, treedef, data, ui)


    def IC_cons(self, param_list, treedef, IC_sample_data, IC_sample_data_sol):
        ''' Get initial condition constraints '''
        params = self.unflatten_params(param_list, treedef)
        u_theta = self.model.u_theta(params=params, data=IC_sample_data)
        if self.system == "transport":
            return Transport_eq(beta=self.beta).solution(\
                IC_sample_data[:,0], IC_sample_data[:,1]) - u_theta
        elif self.system == "reaction_diffusion":
            return IC_sample_data_sol - u_theta
        elif self.system == "reaction":
            return Reaction(self.rho).u0(IC_sample_data[:,0]) - u_theta


    def BC_cons(self, param_list, treedef, BC_sample_data_zero, BC_sample_data_2pi):
        ''' Get boundary condition constraints '''
        params = self.unflatten_params(param_list, treedef)
        u_theta_0 = self.model.u_theta(params=params, data=BC_sample_data_zero)
        u_theta_2pi = self.model.u_theta(params=params, data=BC_sample_data_2pi)
        return u_theta_2pi - u_theta_0
    
    
    def pde_cons(self, param_list, treedef, pde_sample_data):
        ''' Get PDE constraints '''
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
    
    
    def eq_cons(self, param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
        ''' Stack IC, BC, PDE constraints together into an vector '''
        eq_cons = jnp.concatenate([self.IC_cons(param_list, treedef, IC_sample_data, IC_sample_data_sol), self.BC_cons(param_list, treedef, BC_sample_data_zero, BC_sample_data_2pi), self.pde_cons(param_list, treedef, pde_sample_data)])
        return eq_cons


    def grads_eq_cons(self, param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
        ''' Compute IC, BC, PDE constraints gradient '''
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
        ''' Define Lagrangian function '''
        return self.obj(param_list, treedef, data, ui) + self.eq_cons(param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data) @ mul
    
    
    def gradient_L(self, param_list, mul, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
        ''' Compute Lagrangian function gradient '''
        return jacfwd(self.L, 0)(param_list, mul, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data), \
            self.eq_cons(param_list, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)

    
    def SR1(self, xk, xk1, Bk, mulk1, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data):
        ''' SR1 method for quasi-Newton Hessian information updating '''
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
        ''' Damped BFGS method for quasi-Newton Hessian information updating '''
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
        ''' This function is to perform Trsut Region Sequential Quadratic Programming. 
        We modifed the TrSQP method used in SciPy. See Scipy for more information '''

        # penalty increment controling factor
        PENALTY_FACTOR = 0.3
        # trust region and NN parameters updating factors
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
        
        # Dimention of NN parameters
        n, = jnp.shape(x0)
    
        # Set default lower and upper bounds.
        if trust_lb is None:
            trust_lb = jnp.full(n, -jnp.inf)
        if trust_ub is None:
            trust_ub = jnp.full(n, jnp.inf)
        # initialization
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

        last_iteration_failed = False
        data, ui = init_data, init_ui

        # Get the constraints data
        pde_sample_data, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi = Datas.sample_data(sample_key_num, X_star, eval_ui)
        for iter in tqdm(range(sqp_maxiter)):
            # Compute the Cauchy reduction within the shrunk trust region
            dn = modified_dogleg(A, Y, b,
                                TR_FACTOR*trust_radius,
                                BOX_FACTOR*trust_lb,
                                BOX_FACTOR*trust_ub)

            # Perform projected conjugate gradient to compute direction of descent desides the cuachy reduction
            c_t = H.dot(dn) + c
            b_t = jnp.zeros_like(b)
            trust_radius_t = jnp.sqrt(trust_radius**2 - jnp.linalg.norm(dn)**2)
            lb_t = trust_lb - dn
            ub_t = trust_ub - dn
            dt, _ = projected_cg(H, c_t, Z, Y, b_t,
                                    trust_radius_t,
                                    lb_t, ub_t)
            # Combine cuachy direction and projected conjugate gradient direction
            d = dn + dt
            # Compute the quadratic model for deciding whether the trail of step is accepted or not
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
            
            # Update trust region based on different levels of confidence
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

            # Update NN parameters
            prev_x = jnp.copy(x)
            if reduction_ratio >= SUFFICIENT_REDUCTION_RATIO:
                x = x_next
                S = self.default_scaling(x)
                last_iteration_failed = False
            else:
                penalty = previous_penalty
                last_iteration_failed = True

            # Compute and record optimization information
            params = self.unflatten_params(prev_x, treedef)
            absolute_error, l2_relative_error, _ = self.evaluation(params)
            self.absolute_error_iter.append(absolute_error)
            self.l2_relative_error_iter.append(l2_relative_error)
            self.total_l_k_loss_list.append(f)
            self.total_eq_cons_loss_list.append(jnp.linalg.norm(b, ord=2))
            self.time_iter.append(time.time() - start_time)

            # Computing new function and gradients for next optimization round 
            f, b = self.obj(x, treedef, data, ui), self.eq_cons(x, treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
            c, A = self.grad_objective(x, treedef, data, ui), self.grads_eq_cons(x,treedef, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
            Z, LS, Y = projections(A)
            v = -LS.dot(c)
            
            # Termination checking
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
                print("kkt: ", str(kkt1))
                print("penalty: ", str(penalty))
                print("reduction ratio", str(reduction_ratio))
                print("absolute_error: " + str(absolute_error))
                print("l2_relative_error: " + str(l2_relative_error))
                df_param = pd.DataFrame(x, columns=['params'])
                df_param.to_pickle(self.intermediate_data_frame_path+"SQP_params.pkl")

            # Perform SR1 or dBFGS for Hessian information updates
            if last_iteration_failed == False:
              if hessian_method == "dBFGS":
                  H = self.dBFGS(prev_x, x, H, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
              elif hessian_method == "SR1":
                  H = self.SR1(prev_x, x, H, v, treedef, data, ui, IC_sample_data, IC_sample_data_sol, BC_sample_data_zero, BC_sample_data_2pi, pde_sample_data)
              elif hessian_method == None:
                  pass
        return x
    
