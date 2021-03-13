import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    DQN is a fully connected network with 3 linear layers and relu activation function.
    """
    def __init__(self, input_dim, output_dim, hidden1=64, hidden2=64):
        super(DQN, self).__init__()

        # 3 linear layers
        self.linear1 = nn.Linear(input_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, output_dim)
        # concatenation of all with relu activation function
        self.all = nn.Sequential(self.linear1, nn.ReLU(), self.linear2, nn.ReLU(), self.linear3)

    def forward(self, state):
        """Given a state, pass it through the network and return the predicted q values for each action."""
        return self.all(state)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture. One linear layer, two linear layers for advantage and two linear layers for values.
    """
    def __init__(self, input_dim, output_dim, linear_hid_size=128, adv_hid_size=128, val_hid_size=128):
        super(DuelingDQN, self).__init__()

        # common linear layer
        self.linear1 = nn.Linear(input_dim, linear_hid_size)

        # 2 linear layers for advantage calculation
        self.linear_adv_1 = nn.Linear(linear_hid_size, adv_hid_size)
        self.linear_adv_2 = nn.Linear(adv_hid_size, output_dim)

        # 2 linear layers for value calculation
        self.linear_val_1 = nn.Linear(linear_hid_size, val_hid_size)
        self.linear_val_2 = nn.Linear(val_hid_size, 1)

    def forward(self, state):
        """
        Given a state, pass it through the network and return the predicted q values for each action computed by the
        formula that combines the advantage and value calculated.
        """
        x = F.relu(self.linear1(state))
        adv = self.linear_adv_2(F.relu(self.linear_adv_1(x)))
        val = self.linear_val_2(F.relu(self.linear_val_1(x)))

        return val + (adv - adv.mean())
