import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, obs):
        # Convert obs to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.tanh(self.layer1(obs))
        activation2 = F.tanh(self.layer2(activation1))
        output = self.layer3(activation2)

        return output

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, obs):
        # Convert obs to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.tanh(self.layer1(obs))
        activation2 = F.tanh(self.layer2(activation1))
        output = self.layer3(activation2)

        return output