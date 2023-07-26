import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from Transport_eq import Transport_eq

from jax import numpy as jnp
from jax import jacfwd
import pandas as pd
from flax.core.frozen_dict import unfreeze



class FletcherPenalty:
    def __init__(self, model, data, sample_data, IC_sample_data, BC_sample_data, ui, beta, N, M):
        self.model = model
        self.beta = beta
        self.data = data
        self.sample_data = sample_data
        self.IC_sample_data = IC_sample_data
        self.BC_sample_data = BC_sample_data
        self.ui = ui
        self.N = N
        self.M = M


    def l_k(self, params):
        u_theta = self.model.u_theta(params=params, data=self.data)
        return 1 / self.N * jnp.square(jnp.linalg.norm(u_theta - self.ui, ord=2))
    

    def IC_cons(self, params):
        u_theta = self.model.u_theta(params=params, data=self.IC_sample_data)
        return Transport_eq(beta=self.beta).solution(\
            self.IC_sample_data[:,0], self.IC_sample_data[:,1]) - u_theta
    
    
    def BC_cons(self, params):
        u_theta = self.model.u_theta(params=params, data=self.BC_sample_data)
        return Transport_eq(beta=self.beta).solution(\
            self.BC_sample_data[:,0], self.BC_sample_data[:,1]) - u_theta
    
    
    def pde_cons(self, params):
        grad_x = jacfwd(self.model.u_theta, 1)(params, self.sample_data)
        return Transport_eq(beta=self.beta).pde(jnp.diag(grad_x[:,:,0]),\
            jnp.diag(grad_x[:,:,1]))
    

    def eq_cons(self, params):
        return jnp.concatenate([self.IC_cons(params), self.BC_cons(params), self.pde_cons(params)])
    

    def eq_cons_loss(self, params):
        return jnp.square(jnp.linalg.norm(self.eq_cons(params), ord=2))
    

    def flat_single_dict(self, dicts):
        return jnp.concatenate(pd.DataFrame.from_dict(unfreeze(dicts["params"])).\
                        applymap(lambda x: x.flatten()).values.flatten().tolist())
    

    # def flat_multi_dict(self, dicts, group_labels):
    #     return jnp.concatenate(pd.DataFrame.from_dict(\
    #             unfreeze(dicts['params'])).\
    #                 apply(lambda x: x.explode()).set_index([group_labels]).\
    #                     sort_index().applymap(lambda x: x.flatten()).values.flatten().tolist())

    
    def flat_multi_dict(self, dicts):
        df = pd.DataFrame.from_dict(\
                unfreeze(dicts['params'])).\
                    apply(lambda x: x.explode())
        bias = df.loc['bias']
        kernel = df.loc['kernel']
        dd = []
        for i in range(self.M):
            dd.append(pd.concat([bias.iloc[i,:], kernel.iloc[i,:]], axis=1).T)
        return jnp.concatenate(pd.concat(dd, axis=0).applymap(lambda x: x.flatten()).values.flatten().tolist())
    

    # def loss(self, params, penalty_param, group_labels):
    #     df = pd.DataFrame.from_dict(\
    #             unfreeze(jacfwd(self.eq_cons, 0)(params)['params'])).\
    #                 apply(lambda x: x.explode())
    #     bias = df.loc['bias']
    #     kernel = df.loc['kernel']
    #     dd = []
    #     for i in range(self.M):
    #         dd.append(pd.concat([bias.iloc[i,:], kernel.iloc[i,:]], axis=1).T)
    #     return jnp.concatenate(pd.concat(dd, axis=0).applymap(lambda x: x.flatten()).values.flatten().tolist())
        




        # df['index'] = group_labels
        # df.sort_values("index", inplace=True)
        # df.drop(columns="index", inplace=True)
        # return jnp.concatenate(df.applymap(lambda x: x.flatten()).values.flatten().tolist())
    

    
    def loss(self, params, penalty_param):
        flatted_gra_l_k = self.flat_single_dict(jacfwd(self.l_k, 0)(params))
        flatted_gra_eq_cons = jnp.array(jnp.split(self.flat_multi_dict(jacfwd(self.eq_cons, 0)(params)), self.M))
        lambdax = jnp.linalg.pinv(flatted_gra_eq_cons.T) @ flatted_gra_l_k
        return self.l_k(params) - self.eq_cons(params) @ lambdax + 0.5 * penalty_param * self.eq_cons_loss(params)
        


