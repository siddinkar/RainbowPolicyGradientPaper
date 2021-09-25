import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for i in range(len(sizes) - 1):
        if i == 0:
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
        else:
            layers.append(nn.Linear(sizes[i] + sizes[0], sizes[i+1]))
    return layers

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super(MLPActor, self).__init__()
        assert len(hidden_sizes) == 2
        self.pi_layers = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh())
        self.l1, self.l2, self.l3 = self.pi_layers
        self.act_limit = torch.as_tensor(act_limit).cuda(device)

    def forward(self, obs):
        a = F.relu(self.l1(obs))

        a = torch.cat([a, obs], -1)
        a = F.relu(self.l2(a))

        a = torch.cat([a, obs], -1)

        return self.act_limit * torch.tanh(self.l3(a))


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPCritic, self).__init__()
        assert len(hidden_sizes) == 2
        self.q_layers = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.l1, self.l2, self.l3 = self.q_layers


    def forward(self, obs, act):
        sa = torch.cat([obs, act], dim=-1)

        q = F.relu(self.l1(sa))

        q = torch.cat([q, sa], -1)
        q = F.relu(self.l2(q))

        q = torch.cat([q, sa], -1)

        q = self.l3(q)

        return torch.squeeze(q, -1)



class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256), activation=nn.ReLU()):
        super(MLPActorCritic, self).__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        self.q1 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

