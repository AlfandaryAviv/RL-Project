from collections import deque, namedtuple
import random
import numpy as np
import torch


class ReplayMemory:
    """Replay memory class - for sampling experiences from memory"""
    def __init__(self, max_length, device):
        self.max_length = max_length
        # cretae buffer with maximum given length
        self.buffer = deque(maxlen=max_length)
        self.device = device
        # define named tuple of experience
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def append(self, state, action, reward, next_state, done):
        """Add a new experience to the replay buffer"""
        experience = self.experience(state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size, model_name):
        """
        Sample items from the replay buffer according to the batch size value, and return the experience (already
        divided to states, actions, rewards, next states and dones.
        """
        experiences = random.sample(self.buffer, k=batch_size)   # sample random experiences
        my_type = torch.long if 'DQN' in model_name else torch.float

        state_batch = torch.tensor(np.vstack([exp.state for exp in experiences]), device=self.device,
                                   dtype=torch.float)
        action_batch = torch.tensor(np.vstack([exp.action for exp in experiences]), device=self.device,
                                    dtype=my_type)
        reward_batch = torch.tensor(np.vstack([exp.reward for exp in experiences]), device=self.device,
                                    dtype=torch.float)
        next_state_batch = torch.tensor(np.vstack([exp.next_state for exp in experiences]), device=self.device,
                                        dtype=torch.float)
        done_batch = torch.tensor(np.vstack([[exp.done] for exp in experiences]), device=self.device,
                                  dtype=torch.float)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
