from jax import numpy as jnp

class DataLoader:
    def __init__(self, Data) -> None:
        self.data_generator = Data


    def get_data(self, x_data_min, x_data_max, t_data_min, t_data_max, \
        x_sample_min, x_sample_max, t_sample_min, t_sample_max, beta, M):
        generated_data = self.data_generator.generate_data(x_min=x_data_min,\
                x_max=x_data_max,t_min=t_data_min,t_max=t_data_max,beta=beta)
        sample_data = self.data_generator.sample_data(x_min=x_sample_min,\
                    x_max=x_sample_max,t_min=t_sample_min,t_max=t_sample_max)
        xi, ti, ui = generated_data
        xj, tj = sample_data
        data = jnp.concatenate((xi.T, ti.T), axis=1)
        sample_data = jnp.concatenate((xj.T, tj.T), axis=1)
        zeros = jnp.zeros((M,1))
        IC_sample_data = jnp.concatenate((xj.T, zeros), axis=1)
        return data, sample_data, IC_sample_data, ui