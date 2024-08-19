import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftQNetwork(nn.Module):
    """
    Soft Q Network class used as an actor in SAC,
    :ivar fc1: First fully connected layer
    :ivar fc2: Second fully connected layer
    :ivar fc3: Third fully connected layer
    """

    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
