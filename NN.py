from flax import linen as nn
from typing import Sequence, Callable
from jax import random



class NN(nn.Module):
    features: Sequence[int]
    activation: Callable

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
            # if i == len(self.features) - 2:
                x = self.activation(x)
        return x
    

    def init_params(self, NN_key_num, data):
        return self.init(random.PRNGKey(NN_key_num), data)
    
    
    def u_theta(self, params, data):
        return self.apply(params, data).T[0]