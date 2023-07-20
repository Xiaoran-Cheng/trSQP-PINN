
import jax.numpy as jnp
from jaxopt import LBFGS
from jax import value_and_grad

def loss_fun(params):
  dd = jnp.sum(params**4 - params**2)
  ddd.append(dd)
  return dd


ddd = []
init_params = jnp.array([1.0, 1.0], dtype=jnp.float32)
solver = LBFGS(loss_fun, \
               maxiter=1, \
                tol=1e-10, 
                stop_if_linesearch_fails=True,\
                linesearch='backtracking',
                  jit = False)

params_solution, state = solver.run(init_params)




for i in ddd:
  print(i)
  print("????????????????????????????????????????")



# hager-zhang









# solver = LBFGS(loss_fun, maxiter=1, tol=1e-10, stop_if_linesearch_fails=False, linesearch='backtracking')
# params = jnp.array([1.0, 1.0], dtype=jnp.float32)
# for i in range(100):
#   state = solver.init_state(params)
#   params, _ = solver.update(params, state)
#   print(loss_fun(params))

# params

# len(ddd)





# hager-zhang


