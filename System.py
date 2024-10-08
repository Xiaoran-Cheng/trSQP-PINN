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
    ''' Define IC, PDE and solution of reaction-diffusion equation '''
    def __init__(self, nu, rho) -> None:
        self.nu = nu
        self.rho = rho


    def pde(self, dudt, du2dx2, u):
        return dudt - self.nu * du2dx2 - self.rho * u * (1 - u)
    

    def u0(self, x):
        ''' IC condition '''
        x0 = jnp.pi
        sigma = 0.5
        return jnp.exp(-jnp.power((x - x0)/sigma, 2.)/2.)


    def reaction(self, u, dt):
        ''' Use reaction component for iteratively get diffusion term solution'''
        factor_1 = u * jnp.exp(self.rho * dt)
        factor_2 = (1 - u)
        u = factor_1 / (factor_2 + factor_1)
        return u


    def diffusion(self, u, dt, IKX2):
        ''' get diffusion term solution '''
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
    ''' Define IC, PDE and solution of reaction equation '''
    def __init__(self, rho) -> None:
        self.rho = rho


    def pde(self, dudt, u):
        return dudt  - self.rho * u * (1 - u)
    

    def u0(self, x):
        x0 = jnp.pi
        sigma = 0.5
        return jnp.exp(-jnp.power((x - x0)/sigma, 2.)/2.)


    def solution(self, u0, t):
        factor_1 = u0 * jnp.exp(self.rho * t)
        factor_2 = (1 - u0)
        return factor_1 / (factor_2 + factor_1)
    

