import jax.numpy as jnp

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


    def pde(self, dudt, du2dx2, u_theta):
        return dudt - self.nu * du2dx2 - self.rho * u_theta * (1 - u_theta)
    

    def u0(self, x):
        return jnp.exp(-jnp.power((x - jnp.pi)/(jnp.pi/4), 2.)/2.)


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
        # u0 = jnp.exp(-jnp.power((x - jnp.pi)/(jnp.pi/4), 2.)/2.)
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
        
# from Visualization import Visualization
# import sys
# import os
# import numpy as np
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# current_dir = os.getcwd().replace("\\", "/")
# sys.path.append(parent_dir)
# visual = Visualization(current_dir)

# x = jnp.arange(0, 2*jnp.pi, 2*jnp.pi/256)
# x.shape
# t = jnp.linspace(0, 1, 100).reshape(-1, 1)
# X, T = np.meshgrid(x, t)
# X_star = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))


# xgrid = x.shape[0]

# sol= Reaction_Diffusion(5,5).solution(x, t)

# sol.shape




# visual.heatmap(X_star, sol, "", "", "", 1, 100, xgrid)




