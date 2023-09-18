# import numpy as np
# from scipy.integrate import odeint
# from jax import numpy as jnp
# nu = 1

# xgrid = 256
# nt = 1000

# x_min = 0
# x_max = 2*jnp.pi
# t_min = 0
# t_max = 1
   
# x = jnp.arange(x_min, x_max, x_max/xgrid)
# t = jnp.linspace(t_min, t_max, nt)
# X, T = np.meshgrid(x, t)
# eval_data = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# u0 = jnp.exp(-jnp.power((x - jnp.pi)/(0.5), 2.)/2.)
# # u0 = jnp.sin(x) + 10
# kappa = 2*jnp.pi * jnp.fft.fftfreq(xgrid, d = x_max/xgrid)
# def rhsBurgers(u,t,kappa):
#     uhat = np.fft.fft(u)
#     d_uhat = (1j)*kappa*uhat
#     dd_uhat = -np.power(kappa,2)*uhat
#     d_u = np.fft.ifft(d_uhat)
#     dd_u = np.fft.ifft(dd_uhat)
#     du_dt = -u * d_u + nu*dd_u
#     return du_dt.real


# eval_ui = odeint(rhsBurgers, u0, t, args=(kappa,))

# from Visualization import Visualization
# import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# current_dir = os.getcwd().replace("\\", "/")
# visual = Visualization(current_dir)

# color_bar_bounds = [eval_ui.min(), eval_ui.max()]
# visual.heatmap(eval_data, eval_ui, "True_sol", experiment='Pre_Train', nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)




import numpy as np
from scipy.integrate import odeint
from jax import numpy as jnp

class BurgersSolver:
    def __init__(self, nu=1):
        self.nu = nu

    def initial_condition(self, x):
        return jnp.exp(-jnp.power((x - jnp.pi) / 0.5, 2.) / 2.)

    def rhs_burgers(self, u, t, kappa):
        uhat = np.fft.fft(u)
        d_uhat = (1j) * kappa * uhat
        dd_uhat = -jnp.power(kappa, 2) * uhat
        d_u = np.fft.ifft(d_uhat)
        dd_u = np.fft.ifft(dd_uhat)
        du_dt = -u * d_u + self.nu * dd_u
        return du_dt.real

    def solve(self, kappa, x, t):
        u0 = self.initial_condition(x)
        eval_ui = odeint(self.rhs_burgers, u0, t, args=(kappa,))
        return eval_ui


nu = 1

xgrid = 256
nt = 1000

x_min = 0
x_max = 2*jnp.pi
t_min = 0
t_max = 1
   
x = jnp.arange(x_min, x_max, x_max/xgrid)
t = jnp.linspace(t_min, t_max, nt)
X, T = np.meshgrid(x, t)
eval_data = jnp.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# Usage example:
kappa = 2 * jnp.pi * jnp.fft.fftfreq(xgrid, d=x_max / xgrid)
solver = BurgersSolver()
eval_ui = solver.solve(kappa, x, t)



from Visualization import Visualization
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
current_dir = os.getcwd().replace("\\", "/")
visual = Visualization(current_dir)



color_bar_bounds = [eval_ui.min(), eval_ui.max()]
visual.heatmap(eval_data, eval_ui, "True_sol", experiment='Pre_Train', nt=nt, xgrid=xgrid, color_bar_bounds=color_bar_bounds)
