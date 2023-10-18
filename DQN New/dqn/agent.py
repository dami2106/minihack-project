from dqn.model import DQN
from dqn.replay_memory import ReplayMemory, Record
import torch 
import numpy as np 
from dqn.symbols import Symbols
import random
import math
from torch import optim, nn

class Agent:
    def __init__(
                 self,
                 eps_start, 
                 eps_end,
                 eps_decay,
                 env
                ):
        self.env = env
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.policy = DQN(self.observation_space, self.action_space)
        
        


    def get_observation(self, obs):
        print(obs)
        chs = obs["chars"]
        colors = obs["colors"]



        # pixels = obs["pixels"]
        # glyphs = obs["glyphs"]

        obs_set = [chs, colors]

        return torch.Tensor(
            np.array([obs_set])
            )

    def train_loop(
              self,
              replay_memory: ReplayMemory,
              batch_size: int,
              gamma: float,
              print_msg : bool
              ):
        
        state = self.env.reset()
        steps, t_r, t_l = 0, 0, 0
        next_state = None
        done = False

        while not done:
            state = self.get_observation(state)
            action = self.act(state, steps)

            # reward = self.env.step(action)
            next_state, reward, done, _ = self.env.step(action)
            steps += 1
            t_r += reward

            if next_state is not None:
                ns = next_state if next_state is not None else torch.zeros(state.shape)
                replay_memory.push(Record(state, action, reward, ns))
            
            next_state = state

            if len(replay_memory.memory) > batch_size:
                loss = self.optimize_td_loss(replay_memory.sample(batch_size), gamma)
                t_l += loss

            if print_msg:
                print("*"*8 + "Training" + "*"*8)
                print("Steps: {}, Reward: {}, Loss: {}".format(steps, t_r, t_l))
                # print(f"Status: {'Win' if self.env.over_hero_symbol == Symbols.STAIR_UP_CHAR else 'Lost'}") #If were in a staircase based ennv 
    
    def train_dqn(
                  self,
                  n_episodes = 1000,
                  reply_buffer_size = 10000,
                  batch_size = 128,
                  gamma = 0.99,
                  print_msg = True
                  ):
        replay_memory = ReplayMemory(reply_buffer_size)

        for i in range(n_episodes):
            print(f"Episode {i}")
            self.train_loop(replay_memory, batch_size, gamma, print_msg)

    
    def act(self, state, steps):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy(state).argmax(1)[0]
        else:
            return torch.tensor(random.sample(range(self.action_space.n), 1)[0])

    def optimize_td_loss(self, batch, gamma):
        loss_func = nn.SmoothL1Loss()
        optimizer = optim.AdamW(self.policy.parameters())

        # batches
        state_batch = torch.cat([torch.tensor(s.state) for s in batch])
        next_state_batch = torch.cat([torch.tensor(s.next_state) for s in batch])
        reward_batch = torch.cat([torch.tensor([s.reward]) for s in batch])

        # expected action values
        prediction = self.policy(state_batch)
        with torch.no_grad():
            next_state_values = self.policy(next_state_batch).max(1)[0]
        target = reward_batch + gamma * next_state_values

        # update weights
        loss = loss_func(prediction, target.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        optimizer.step()
        return loss.item()