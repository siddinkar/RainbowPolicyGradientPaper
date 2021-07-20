import torch
import torch.nn as nn
import numpy as np

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation;
        layers += [nn.Linear(sizes[i], sizes[i+1]), act]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super(MLPActor, self).__init__()
        self.pi = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.pi(obs)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPCritic, self).__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)



class MLPActorCritic(nn.Module):

    def __init__(self, env, hidden_sizes=(256,256), activation=nn.ReLU):
        super(MLPActorCritic, self).__init__()

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0]

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

