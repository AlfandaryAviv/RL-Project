import torch
import torch.nn as nn
from torch.distributions import Normal
from utils import initialize


class SoftCritic(nn.Module):
    """Soft Q-Network"""
    def __init__(self, input_size, action_size, hid_size=256, pre_w=3e-3):
        super(SoftCritic, self).__init__()
        # 3 linear layers
        self.fc1 = nn.Linear(input_size + action_size, hid_size)
        self.fc2 = nn.Linear(hid_size, hid_size)
        self.fc3 = nn.Linear(hid_size, 1)
        self.w = pre_w

        # pre initialization of the weights and bias of the third linear layer
        initialize(self.fc3, self.w)

        # combine all layers with a ReLU activation function
        self.all = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3)

    def forward(self, state, action):
        """
        Given a state and an action, concatenate them and pass them through the network and return the predicted
        q-values for each action
        """
        x = torch.cat([state, action], 1)
        x = self.all(x)
        return x


class Actor(nn.Module):
    """Policy Network"""
    def __init__(self, action_range, input_size, action_size, device, hid_size=256, pre_w=3e-3, log_std_min=-20,
                 log_std_max=2):
        super(Actor, self).__init__()
        self.device = device

        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.w = pre_w
        self.action_range = action_range

        # 2 linear layers
        self.fc1 = nn.Linear(input_size, hid_size)
        self.fc2 = nn.Linear(hid_size, hid_size)
        self.all_fc = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU())

        # one network head that for every action outputs the mean of the Gaussian distribution
        self.mean_fc = nn.Linear(hid_size, action_size)
        # initialize weights and biases uniformly
        initialize(self.mean_fc, self.w)

        # another network head that for every action output s the log of the Gaussian distribution
        self.log_std_fc = nn.Linear(hid_size, action_size)
        # initialize weights and biases uniformly
        initialize(self.log_std_fc, self.w)

    def forward(self, state):
        """Given a state pass it through the network and return the mean and log std of the Gaussian distribution"""
        x = self.all_fc(state)
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # the neural network outputs the mean and the log std of the Gaussian distribution
        return mean, log_std

    def sample(self, state, epsilon=1e-6, only_action=False):
        """
        Given a state pass it through the network to find the parameters of the Gaussian distribution, sample from it
        and apply tanh to find the action and compute the log of the policy (return them both).
        """
        # mean and log std of the Gaussian distribution, thus exp() needs to be performed
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # normal is this Gaussian distribution
        normal_dist = Normal(mean, std)
        # sample of z from this Gaussian distribution
        z = normal_dist.rsample()
        # action is represented as the tanh applied to z-values sampled from the mean and covariance of the
        # Gaussian distribution that is the output of the actor network
        action = torch.tanh(z)

        if not only_action:
            # computation is modified to address action bounds on the unbounded Gaussian distribution
            log_pi = normal_dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
            log_pi = log_pi.sum(1, keepdim=True)
        else:
            log_pi = None

        return action, log_pi

    def get_action(self, state):
        """
        Given a state pass it through the network to find the parameters of the Gaussian distribution, sample from it
        and apply tanh to find the action and compute the final action by the formula.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # mean and log std of the Gaussian distribution, thus exp() needs to be performed
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # normal is this Gaussian distribution
        normal_dist = Normal(mean, std)
        # sample of z from this Gaussian distribution
        z = normal_dist.sample()
        # action is represented as the tanh applied to z-values sampled from the mean and covariance of the
        # Gaussian distribution that is the output of the actor network
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        # rescale
        action = action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] +
                                                                                 self.action_range[0]) / 2.0
        return action
