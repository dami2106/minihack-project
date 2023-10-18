from gym import spaces
import torch.nn as nn
from torch.nn import (BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Linear,
                      MaxPool2d, Module, ReLU, Sequential, Softmax)
import torch.nn.functional as F
import torch 
class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network.
    """

    def __init__(self, observation_space, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param action_space: the action space of the environment
        """
        super().__init__()
        n_observations = observation_space.shape[0] * observation_space.shape[1]
        self.fc = nn.Linear(n_observations, 64)

        self.Tlayer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        self.transformerE = nn.TransformerEncoder(self.Tlayer, num_layers=3)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, action_space.n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    # n_observations = observation_space.shape[0] * observation_space.shape[1]
    #     self.layer1 = nn.Linear(n_observations, 512)
    #     self.layer2 = nn.Linear(512, 128)
    #     self.layer3 = nn.Linear(128, action_space.n)
        
        # self.conv1 = Conv2d(in_channels=1, out_channels=20,
        #     kernel_size=(5, 5))
        # self.relu1 = ReLU()
        # self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # # initialize second set of CONV => RELU => POOL layers
        # self.conv2 = Conv2d(in_channels=20, out_channels=50,
        #     kernel_size=(5, 5))
        # self.relu2 = ReLU()
        # self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # # initialize first (and only) set of FC => RELU layers
        # self.fc1 = Linear(in_features=1600, out_features=500)
        # self.relu3 = ReLU()
        # self.fc2 = Linear(in_features=500, out_features=action_space.n)



    def forward(self, x):
        """
        Returns the values of a forward pass of the network
        :param x: The input to feed into the network 
        """
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # return self.layer3(x)

        x = self.fc(x)
        out = self.transformerE(x)
        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out)

        return qvalue

        # # define first conv layer with max pooling
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.maxpool1(x)
        # # define second conv layer with max pooling
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        # # Define fully connected layers
        # x = x.reshape(x.shape[0], -1)
        # x = self.fc1(x)
        # x = self.relu3(x)
        # x = self.fc2(x)
        # return x