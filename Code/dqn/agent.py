from gym import spaces
import numpy as np

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.betas = (0.9, 0.999)

        #Need to create two networks, one for the policy and one for the target network (for double DQN)
        self.policy_network = DQN(self.observation_space, self.action_space).to(device) #online
        self.target_network = DQN(self.observation_space, self.action_space).to(device)

        #Need to initialise the target network with the same weights as the policy network (backup / copy the weights)
        self.update_target_network()
        
        #Need to set the target network to eval mode so that it doesn't train
        self.target_network.eval()
        
        #Need to use Adam optimiser (Benji said so)
        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr, betas=self.betas) 
        


    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """

        state_set, action_set, reward_set, next_state_set, done_set = self.replay_buffer.sample(self.batch_size)

        #Need to convert the numpy arrays to tensors
        state_set = torch.from_numpy(state_set).float().to(device)
        next_state_set = torch.from_numpy(next_state_set).float().to(device)

        reward_set = torch.from_numpy(reward_set).float().to(device)
        action_set = torch.from_numpy(action_set).long().to(device)
        done_set = torch.from_numpy(done_set).float().to(device)
        
        
        #Need to calculate TD error 
        with torch.no_grad(): #Need to use no grad because we don't want to train the target network
            if self.use_double_dqn:
                #Need to use the policy network to select the best action
                next_state_q_values = self.policy_network(next_state_set)
                next_state_best_actions = torch.argmax(next_state_q_values, dim=1)
                #Need to use the target network to get the q values for the best actions
                next_state_q_values = self.target_network(next_state_set)
                next_state_q_values = next_state_q_values.gather(1, next_state_best_actions.unsqueeze(1)).squeeze(1)
            else:
                #Need to use the target network to get the q values for the best actions
                next_state_q_values = self.target_network(next_state_set)
                next_state_q_values = torch.max(next_state_q_values, dim=1).values

            final_q = reward_set + (self.gamma * next_state_q_values * (1 - done_set))

        #Need to get the q values for the actions taken
        q_values = self.policy_network(state_set)
        q_values = q_values.gather(1, action_set.unsqueeze(1)).squeeze(1)

        # loss = F.smooth_l1_loss(input_q_values, target_q_values)
        loss = F.mse_loss(q_values, final_q)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        #Need to copy the weights from the policy network to the target network (backup)
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, observation):
        """Select action base on network inference"""

        #Need to convert the tensor to floats onto the gpu
        state = observation.float().unsqueeze(0).to(device)

        #Need to get the q values for the state and select the best action (argmax)
        #Need to use no grad because we don't want to train the policy network
        with torch.no_grad():
            result = self.policy_network.forward(state)
            return torch.argmax(result).item()

    def save_network(self, fname):
        #Save the polciy network to a file
        torch.save(self.policy_network, fname)