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

        for ep in range(max_episodes):
            

            state = self.env.reset()
            #TODO Need to format the state here 
            state = get_observation(state)

            episode_metrics = {
                "reward" : [],
                "actor_values" : [],
                "critic_values" : []
            }



            #Generate an episode 
            for _ in range(max_episode_length):


                act, crit = self.model.forward(state)
                best_action = torch.distributions.Categorical(act).sample()

                next_state, reward, done, _ = self.env.step(best_action.item())
                #TODO Format next_state here
                next_state = get_observation(next_state)

                # self.replay_buffer.add(state, best_action, reward, next_state, done)

                
                episode_metrics["reward"].append(reward)
                episode_metrics["actor_values"].append(
                    torch.distributions.Categorical(act).log_prob(best_action)
                    ) #act
                episode_metrics["critic_values"].append(crit) #state
                
                state = next_state

                if done:
                    break

            episode_return = self.calc_return(episode_metrics["reward"])

            self.score.append(np.sum(episode_metrics["reward"]))
            
            print(f"Episode : {ep}, Reward : {np.round(np.sum(episode_metrics['reward']), 3)}, Avg Reward : {np.round(np.mean(self.score[-50:]), 3)}")


            loss = self.get_loss(episode_metrics["actor_values"], episode_metrics["critic_values"], episode_return)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()


    def get_loss(self, actor_values, critic_values, rewards):
        loss = 0
        for a, c, r in zip(actor_values, critic_values, rewards):
            actor_loss = -a *  r - c.item()
            critic_loss = F.smooth_l1_loss(c, r.resize(1,1))
            loss += (actor_loss + critic_loss)
        return loss
    
    def calc_return(self, rewards):
        """
        Calculate the return G_t from the rewards
        :param rewards: the rewards
        :return: the return
        """
        r = rewards.copy()
        r.reverse()
        G = 0
        returns = [] 
        # print(r)
        for rew in r:
            G *= self.gamma 
            G += rew
            returns.append(G)

        returns.reverse()

        return torch.from_numpy(np.array(returns)).float().to(device)
  
    def act(self, observation):
        state = get_observation(observation)
        act, _ = self.model.forward(state)
        best_action = torch.distributions.Categorical(act).sample()
        return best_action.item()
    
    def save_network(self, path):
        torch.save(self.model, path)