from gym import spaces
import torch.nn as nn
from torch.nn import (BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Linear,
                      MaxPool2d, Module, ReLU, Sequential, Softmax)
import torch

device = torch.device("cuda")

from a2c.helper import get_observation

class AdvantageActorCritic(nn.Module):
    """
    A basic implementation of a Deep Q-Network - using 2 networks and combiniing them into one 
    to estimate the action probabilities and the state value function
    Using the network proposed by https://github.com/BrentonBudler/deep-rl-minihack-the-planet/blob/main/A2C.ipynb 
    """

    def __init__(self, observation_space, action_space: spaces.Discrete):
        super().__init__()

        #Standard Convolutional and pooling layers for glyphs input
        self.conv1 = Conv2d(in_channels=1, out_channels=20,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Initialize second set of of convolutional and pooling layers with a ReLU activation function 
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Initialize fully connected layers for glyph output after convolutional and pooling layers
        self.fc1 = Linear(in_features=1600, out_features=500)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500, out_features=128)
        self.relu4 = ReLU()

        #Was a normal DQN until here 
        
        # Initialize fully connected for message input 
        self.fc3 = Linear(in_features=256, out_features=128)
        self.relu5 = ReLU()
        
        # Initialize fully connected for combination of glyphs and message 
        self.fc4 = Linear(in_features=256, out_features=128)
        self.relu6 = ReLU()

        # To estimate the value function of the state 
        self.value_layer = nn.Linear(128, 1)

        # To calculate the probability of taking each action in the given state
        self.action_layer = nn.Linear(128, action_space.n)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        """
        Returns the values of a forward pass of the network
        :param x: The input to feed into the network 
        """
 
        raw_glyphs = x["glyphs"]
        raw_message = x["message"]

        #Get tensors for the arrays
        glyphs_tensor  = torch.from_numpy(raw_glyphs).float().to(device)
        message_tensor  = torch.from_numpy(raw_message).float().to(device)

        # Pass the 2D glyphs input through our convolutional and pooling layers 
        glyphs_tensor = self.conv1(glyphs_tensor)
        glyphs_tensor = self.relu1(glyphs_tensor)
        glyphs_tensor = self.maxpool1(glyphs_tensor)
        glyphs_tensor = self.conv2(glyphs_tensor)
        glyphs_tensor = self.relu2(glyphs_tensor)
        glyphs_tensor = self.maxpool2(glyphs_tensor)
        
        # Platten the output from the final pooling layer and pass it through the fully connected layers 
        glyphs_tensor = glyphs_tensor.reshape(glyphs_tensor.shape[0], -1)
        glyphs_tensor = self.fc1(glyphs_tensor)
        glyphs_tensor = self.relu3(glyphs_tensor)
        glyphs_tensor = self.fc2(glyphs_tensor)
        glyphs_tensor = self.relu4(glyphs_tensor)
        
        # Pass the message input through a fully connected layer
        message_tensor = self.fc3(message_tensor)
        message_tensor = self.relu5(message_tensor)

        #Combine the two outputs for the final layers in the network
        combined = torch.concat([glyphs_tensor,message_tensor],1)
        combined = self.fc4(combined)
        combined = self.relu6(combined)
        state_value = self.value_layer(glyphs_tensor)
        action_probs = self.softmax(self.action_layer(glyphs_tensor))

        
        return action_probs, state_value