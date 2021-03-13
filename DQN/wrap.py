import numpy as np
import itertools
from gym import ActionWrapper, spaces
from collections import namedtuple
import gym

ActionChange = namedtuple('AC', ['target', 'convert_from'])


class FlattenAS(ActionWrapper):
    """ Flattens the action space of an environment to a single value vector action."""
    def __init__(self, env):
        super(FlattenAS, self).__init__(env)
        self.env = env
        new = self.flatten_space()
        # the new Discrete action space
        self.action_space = new.target
        # converts the action using the lambda function
        self.action = new.convert_from

    def flatten_space(self):
        """Flattens the space. returns a transform object of the flattened action space."""
        action_space = self.env.action_space
        # pairs of all the combinations from the MD space
        pairs = list(itertools.product(*[range(0, k) for k in action_space.nvec]))
        return ActionChange(target=spaces.Discrete(len(pairs)), convert_from=lambda x: pairs[x])


class DiscretizeAS(ActionWrapper):
    """Discretization of the action space"""
    def __init__(self, env, div):
        super(DiscretizeAS, self).__init__(env)
        self.env = env
        self.div = div
        new = self.discretize()
        # the action space is now a MultiDiscrete
        self.action_space = new.target
        # the actions are converted using the lambda function
        self.action = new.convert_from

    def discretize(self):
        """
        Creates a discrete version of the continuous space. uses the old minimum and maximum value. Return
        a transform to the discrete space.
        """
        action_space = self.env.action_space
        # the lowest and highest actions values , e.g: [-1,-1], [1, 1]
        low, high = action_space.low, action_space.high
        div = np.ones(action_space.shape, dtype=int) * self.div
        # converts steps to a MD space
        discrete_space = spaces.MultiDiscrete(div)
        return ActionChange(target=discrete_space, convert_from=lambda x: low + (high - low) / (div - 1.0) * x)


if __name__ == '__main__':
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)

    print('Observation space: ', env.observation_space, '\nExample state: ', env.observation_space.sample(),
          'Type of state: ', type(env.observation_space.sample()))
    print('Action space: ', env.action_space, '\nExample action space: ', env.action_space.sample())
    print("   ")
    env = DiscretizeAS(env, 3)

    print('Observation space: ', env.observation_space, '\nExample state: ', env.observation_space.sample(),
          'Type of state: ', type(env.observation_space.sample()))
    print('Action space: ', env.action_space, '\nExample action space: ', env.action_space.sample())
    print("   ")

    env = FlattenAS(env)
    # this is now a single integer
    print('Observation space: ', env.observation_space, '\nExample state: ', env.observation_space.sample(),
          'Type of state: ', type(env.observation_space.sample()))
    print('Action space: ', env.action_space, '\nExample action space: ', env.action_space.sample())
