from gym import spaces
import numpy as np

from a2c.model import AdvantageActorCritic
# from dqn.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F

from a2c.helper import get_observation

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ACAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        lr,
        gamma,
        max_episode_length,
        max_episodes,
        env
    ):

        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.model = AdvantageActorCritic(env.observation_space, env.action_space)
        self.model.to(device)
        self.max_episode_length = max_episode_length
        self.max_episodes = max_episodes
        self.env = env

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.score = []

        #Training the network
        for ep in range(max_episodes):
            #Reset the environment and get the initial state
            state = self.env.reset()
            state = get_observation(state)

            #Initialise the episode metrics
            episode_metrics = {
                "reward" : [],
                "actor_values" : [],
                "critic_values" : []
            }

            #Generate an episode 
            for _ in range(max_episode_length):

                #Get the best action for the current state 
                act, crit = self.model.forward(state)
                best_action = torch.distributions.Categorical(act).sample()

                #Take the action and get the next state, reward and done flag
                next_state, reward, done, _ = self.env.step(best_action.item())
                next_state = get_observation(next_state)

                #Append the reward and values to the episode metrics
                episode_metrics["reward"].append(reward)
                episode_metrics["actor_values"].append(
                    torch.distributions.Categorical(act).log_prob(best_action)
                    ) #act
                episode_metrics["critic_values"].append(crit) #state
                
                state = next_state

                if done:
                    break
            
            #Calculate the discounted return for the episode
            episode_return = self.calc_return(episode_metrics["reward"])

            #Append the score for the episode
            self.score.append(np.sum(episode_metrics["reward"]))

            #Calculate the loss for the episode and update the network
            loss = self.get_loss(episode_metrics["actor_values"], episode_metrics["critic_values"], episode_return)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            
            #Print the episode metrics
            print(f"Episode : {ep}, Reward : {np.round(np.sum(episode_metrics['reward']), 3)}, Avg Reward : {np.round(np.mean(self.score[-50:]), 3)}, Loss : {np.round(loss.item(), 3)}")

    #Get the loss for the episode
    def get_loss(self, actor_values, critic_values, rewards):
        loss = 0
        for a, c, r in zip(actor_values, critic_values, rewards):
            actor_loss = -a *  r - c.item()
            critic_loss = F.smooth_l1_loss(c, r.resize(1,1))
            loss += (actor_loss + critic_loss)
        return loss
    
    #Calculate the discounted return for the episode
    def calc_return(self, rewards):
        r = rewards.copy()
        r.reverse()
        G = 0
        returns = [] 
        for rew in r:
            G *= self.gamma 
            G += rew
            returns.append(G)

        returns.reverse()

        return torch.from_numpy(np.array(returns)).float().to(device)
  
    #Get the best action for the current state 
    def act(self, observation):
        state = get_observation(observation)
        act, _ = self.model.forward(state)
        best_action = torch.distributions.Categorical(act).sample()
        return best_action.item()
    
    #Save the network to a file
    def save_network(self, path):
        torch.save(self.model, path)