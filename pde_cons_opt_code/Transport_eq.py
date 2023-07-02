import jax.numpy as jnp

class Transport_eq:
    def __init__(self, beta:float) -> None:
        self.beta = beta


    def solution(self, x, t):
        u = jnp.sin(x - self.beta*t)
        # u = x - self.beta*t
        return u
    

    def pde(self, dudx, dudt):
        return dudt + self.beta * dudx