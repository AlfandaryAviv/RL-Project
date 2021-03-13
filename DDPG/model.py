import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import initialize


class Actor(nn.Module):
    """ Actor Policy Model - maps the state into an action """
    def __init__(self, state_dim, action_dim, linear1_dim=128, linear2_dim=128, pre_w=3e-3):
        super(Actor, self).__init__()
        # 3 linear layers
        self.linear1 = nn.Linear(state_dim, linear1_dim)
        self.linear2 = nn.Linear(linear1_dim, linear2_dim)
        self.linear3 = nn.Linear(linear2_dim, action_dim)
        # for initialized weights and bias of the third linear layer
        self.w = pre_w

        # pre initialization of the weights and bias of the third linear layer
        initialize(self.linear3, self.w)

        # all layers together with relu and tanh activation functions
        self.all = nn.Sequential(self.linear1, nn.ReLU(), self.linear2, nn.ReLU(), self.linear3, nn.Tanh())

    def forward(self, state):
        """Given a state pass it through the network and return the predicted q-values for each action"""
        return self.all(state)


class Critic(nn.Module):
    """ Critic Model - maps the concatenation of state and action into Q-values """
    def __init__(self,state_dim, action_dim, linear1_dim=128, linear2_dim=128, pre_w=3e-3):
        super(Critic, self).__init__()
        # 3 linear layers
        self.linear1 = nn.Linear(state_dim, linear1_dim)
        self.linear2 = nn.Linear(linear1_dim+action_dim, linear2_dim)
        self.linear3 = nn.Linear(linear2_dim, 1)
        # for initialized weights and bias of the third linear layer
        self.w = pre_w

        # pre initialization of the weights and bias of the third linear layer
        initialize(self.linear3, self.w)

    def forward(self, state, action):
        """
        Given a state and an action, concatenate them and pass them through the network
        """
        x = F.relu(self.linear1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
