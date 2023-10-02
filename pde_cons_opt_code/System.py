import jax.numpy as jnp
from scipy.integrate import odeint
import numpy as np



class Transport_eq:
    def __init__(self, beta:float) -> None:
        self.beta = beta


    def solution(self, x, t):
        u = jnp.sin(x - self.beta*t)
        return u
    

    def pde(self, dudx, dudt):
        return dudt + self.beta * dudx
    
    

class Reaction_Diffusion:
    def __init__(self, nu, rho) -> None:
        self.nu = nu
        self.rho = rho


    def pde(self, dudt, du2dx2, u):
        return dudt - self.nu * du2dx2 - self.rho * u * (1 - u)
    

    def u0(self, x):
        x0 = jnp.pi
        sigma = 0.1
        return jnp.exp(-jnp.power((x - x0)/sigma, 2.)/2.)


    def reaction(self, u, dt):
        factor_1 = u * jnp.exp(self.rho * dt)
        factor_2 = (1 - u)
        u = factor_1 / (factor_2 + factor_1)
        return u


    def diffusion(self, u, dt, IKX2):
        factor = jnp.exp(self.nu * IKX2 * dt)
        u_hat = jnp.fft.fft(u)
        u_hat *= factor
        u = jnp.real(jnp.fft.ifft(u_hat))
        return u


    def solution(self, x, t):
        nt = len(t)
        xgrid = len(x)
        dt = 1/nt
        u = jnp.zeros((xgrid, nt))

        IKX_pos = 1j * jnp.arange(0, xgrid/2+1, 1)
        IKX_neg = 1j * jnp.arange(-xgrid/2+1, 0, 1)
        IKX = jnp.concatenate((IKX_pos, IKX_neg))
        IKX2 = IKX * IKX
        u0 = self.u0(x)
        u = u.at[:,0].set(u0)
        u_ = u0
        for i in range(nt-1):
            u_ = self.reaction(u_, dt)
            u_ = self.diffusion(u_, dt, IKX2)
            u = u.at[:,i+1].set(u_)

        u = u.T
        u = u.flatten()
        return u
        


class Reaction:
    def __init__(self, rho) -> None:
        self.rho = rho


    def pde(self, dudt, u):
        return dudt  - self.rho * u * (1 - u)
    

    def u0(self, x):
        x0 = jnp.pi
        # sigma = jnp.pi/4
        sigma = 0.5
        return jnp.exp(-jnp.power((x - x0)/sigma, 2.)/2.)
        # return jnp.sin(x) + 1


    def solution(self, u0, t):
        factor_1 = u0 * jnp.exp(self.rho * t)
        factor_2 = (1 - u0)
        return factor_1 / (factor_2 + factor_1)
    

    


class Burger:
    def __init__(self, alpha):
        self.alpha = alpha

    def pde(self, dudt, dudx, du2dx2, u):
        return dudt + u * dudx - self.alpha * du2dx2

    def u0(self, x):
        x0 = jnp.pi
        # sigma = jnp.pi/4
        sigma = 0.5
        return jnp.exp(-jnp.power((x - x0)/sigma, 2.)/2.)
        # return jnp.sin(x) + 1

    def Burgers_fft(self, u, t, kappa):
        uhat = np.fft.fft(u)
        d_uhat = (1j) * kappa * uhat
        dd_uhat = -jnp.power(kappa, 2) * uhat
        d_u = np.fft.ifft(d_uhat)
        dd_u = np.fft.ifft(dd_uhat)
        du_dt = -u * d_u + self.alpha * dd_u
        return du_dt.real

    def solution(self, kappa, x, t):
        u0 = self.u0(x)
        return odeint(self.Burgers_fft, u0, t, args=(kappa,)).flatten()
    

# alpha = 1

# xgrid = 256
# nt = 1000

# x_min = 0
# x_max = 2*jnp.pi
# t_min = 0
# t_max = 1

# x = jnp.arange(x_min, x_max, x_max/xgrid)
# t = jnp.linspace(t_min, t_max, nt)
# X, T = np.meshgrid(x, t)
# X_star = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# # Usage example:
# kappa = 2 * jnp.pi * jnp.fft.fftfreq(xgrid, d=x_max / xgrid)
# solver = Burger(alpha)
# eval_ui = solver.solution(kappa, x, t)


# from Visualization import Visualization
# import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# current_dir = os.getcwd().replace("\\", "/")
# visual = Visualization(current_dir)



# color_bar_bounds = [eval_ui.min(), eval_ui.max()]
# visual.heatmap(X_star, eval_ui, "True_sol", experiment='Pre_Train', nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)



