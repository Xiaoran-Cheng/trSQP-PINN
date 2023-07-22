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

        sample_data_x, IC_sample_data_x, _ = jnp.split(xj.T, 3)
        sample_data_t, _, BC_sample_data_t = jnp.split(tj.T, 3)
        sample_data = jnp.concatenate((sample_data_x, sample_data_t), axis = 1)
        zeros = jnp.zeros((int(M/3),1))
        twopi = jnp.ones((int(M/3),1)) * 2 * jnp.pi
        IC_sample_data = jnp.concatenate((IC_sample_data_x, zeros), axis=1)
        BC_sample_data = jnp.concatenate((twopi, BC_sample_data_t), axis=1)
        
        return data, sample_data, IC_sample_data, BC_sample_data, ui
    

    def get_eval_data(self, xgrid, nt, x_min, x_max, t_min, t_max, beta):
        X_star = self.data_generator.data_grid(xgrid, nt, x_min, x_max, t_min, t_max)
        data_grid_len = xgrid*nt
        xi = X_star[:,0].reshape(1,data_grid_len)
        ti = X_star[:,1].reshape(1,data_grid_len)
        ui = Transport_eq(beta=beta).solution(xi, ti)
        return X_star, ui


# from data import Data
# #######################################config for data#######################################
# beta_list = [0.0001]
# xgrid = 256
# nt = 100
# N=100
# M=12
# data_key_num, sample_data_key_num = 100, 256
# eval_data_key_num, eval_sample_data_key_num = 300, 756
# dim = 2
# Datas = Data(N=N, M=M, dim=dim)
# dataloader = DataLoader(Data=Datas)

# x_data_min = 0
# x_data_max = 2*jnp.pi
# t_data_min = 0
# t_data_max = 1
# x_sample_min = 0
# x_sample_max = 2*jnp.pi
# t_sample_min = 0
# t_sample_max = 1


# data, sample_data, IC_sample_data, BC_sample_data, ui = dataloader.get_data(xgrid, nt, x_data_min, x_data_max, t_data_min, t_data_max, \
#                     x_sample_min, x_sample_max, t_sample_min, t_sample_max, \
#                         1, M, data_key_num, sample_data_key_num)


# IC_sample_data
# sample_data

# BC_sample_data
