import numpy as np
import copy


class OUNoise:
    """The Ornstein-Uhlenbeck Process (adding noise to the action)"""
    def __init__(self, action_dim, mu=0., theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        # mu theta and sigma are parameters of the OU noise process
        self.mu = np.ones(self.action_dim)*mu
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        # reset the state
        self.state = copy.copy(self.mu)

    def evolve_state(self):
        # add noise to the state
        x = self.state
        # according to the OU noise process
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
