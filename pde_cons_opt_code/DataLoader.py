from jax import numpy as jnp
from Transport_eq import Transport_eq



class DataLoader:
    def __init__(self, Data) -> None:
        self.data_generator = Data


    def get_data(self, xgrid, nt, x_data_min, x_data_max, t_data_min, t_data_max, \
        x_sample_min, x_sample_max, t_sample_min, t_sample_max, beta, M, data_key_num, sample_data_key_num):

        generated_data = self.data_generator.generate_data(xgrid=xgrid,nt=nt,x_min=x_data_min,\
                x_max=x_data_max,t_min=t_data_min,t_max=t_data_max,beta=beta,key_num=data_key_num)
        
        sample_data = self.data_generator.sample_data(xgrid=xgrid,nt=nt,x_min=x_sample_min,\
                    x_max=x_sample_max,t_min=t_sample_min,t_max=t_sample_max,key_num=sample_data_key_num)
        
        xi, ti, ui = generated_data
        xj, tj = sample_data
        data = jnp.concatenate((xi.T, ti.T), axis=1)
        sample_data = jnp.concatenate((xj.T[:int(M/2),:], tj.T[:int(M/2),:]), axis=1)
        zeros = jnp.zeros((int(M/2),1))
        IC_sample_data = jnp.concatenate((xj.T[int(M/2):,:], zeros), axis=1)
        return data, sample_data, IC_sample_data, ui
    

    def get_eval_data(self, xgrid, nt, x_min, x_max, t_min, t_max, beta):
        X_star = self.data_generator.data_grid(xgrid, nt, x_min, x_max, t_min, t_max)
        data_grid_len = xgrid*nt
        xi = X_star[:,0].reshape(1,data_grid_len)
        ti = X_star[:,1].reshape(1,data_grid_len)
        ui = Transport_eq(beta=beta).solution(xi, ti)
        return X_star, ui


