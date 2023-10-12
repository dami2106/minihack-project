import torch.nn as nn
from gym import spaces

class Policy(nn.Module):
    def __init__(self, action_space: spaces.Discrete):
        super(Policy, self).__init__()

        input_size = 21 * 79

        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, action_space.n)
        self.softmax = nn.Softmax(dim=-1)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x