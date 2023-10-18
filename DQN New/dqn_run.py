from dqn.agent import Agent
import random
import numpy as np
import gym

from nle import nethack
from minihack import RewardManager
import torch

import os 


env = gym.make("MiniHack-Room-Monster-5x5-v0",
                observation_keys = ['pixel', 'message', 'glyphs', 'blstats', 'chars', 'colors'],
                # penalty_time=-0.1,
                # penalty_step=-0.1,
                # reward_lose=-1,
                # reward_win=5,
                # seeds = hyper_params["seed"],
                # actions = MOVE_ACTIONS,
                # reward_manager=reward_manager
                )

agent = Agent(
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=1000,
    env=env
)

agent.train_dqn()


