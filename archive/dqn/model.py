from gym import spaces
import torch.nn as nn
from torch.nn import (BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Linear,
                      MaxPool2d, Module, ReLU, Sequential, Softmax)

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    #Nature config - 3 conv layers, 2 linear layers
    #The conv layers need to have decreasing kernel sizes 8 then 4 then 3
    #The conv layers need to have increasing channels 32 then 64 then 64
    #The conv layers need to have decreasing strides 4 then 2 then 1
    #Use relu for the conv and linear layers

    #Make a feed forward function that does a feed forward of all layers
    #make sure to use the nn from the func parameters otherwise it wont work


    def __init__(self, observation_space, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        observation_space.shape[0] 21
        action_space.n 12
        """
        super().__init__()

        # out :  torch.Size([1, 64, 6, 35])
        # batch :  1
        # out :  torch.Size([1, 13440])
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=81 , out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256 , out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256 , out_features=128),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=action_space.n)
        # )
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

        self.conv1 = Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=1, stride=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=1, stride=1)
        self.fc1 = Linear(in_features=32 * 5 * 5, out_features=128)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=128, out_features=action_space.n)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    # def forward(self, x):
    #     """
    #     Returns the values of a forward pass of the network
    #     :param x: The input to feed into the network
    #     """
    #     # define first conv layer with max pooling
    #     #reshape x into 81 from 9x9
    #     # x = x.reshape(x.shape[0], -1)
        

    #     return self.fc(x)
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
