import torch
from torch import nn
from torch.functional import F
from gym import spaces
import torch.nn as nn
import torch 

class DQN(nn.Module):

    def __init__(self, observation_space, action_space: spaces.Discrete):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, action_space.n)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x