from gym import spaces
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network using a transformer from the PyTorch Library
    """

    def __init__(self, observation_space, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param action_space: the action space of the environment
        """
        super().__init__()
        n_observations = observation_space.shape[0] * observation_space.shape[1]

        #Linear layer input is the number of observations going into 64 neurons
        self.fc = nn.Linear(n_observations, 64)

        #MLP feeds into the transformer layer with 2 heads and 64 neurons
        self.Tlayer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        #The transformer layer feeds into the transformer encoder with 3 layers
        self.transformerE = nn.TransformerEncoder(self.Tlayer, num_layers=3)

        #The output of the transformer encoder is fed into a linear layer with 32 neurons
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, action_space.n)

        #Initialise the weights of the network
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        Returns the values of a forward pass of the network
        :param x: The input to feed into the network 
        """
        #Get linear output 
        x = self.fc(x)
        #Get transformer output
        out = self.transformerE(x)
        #Get MLP output
        out = F.relu(self.fc1(out))
        #Get Q values
        qvalue = self.fc2(out)

        return qvalue

 