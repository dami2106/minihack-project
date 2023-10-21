import random
import numpy as np
import gym

# from dqn.agent import DQNAgent
# from dqn.replay_buffer import ReplayBuffer
# from dqn.wrappers import *
# from helper import make_video, normalize_glyphs, distance_to_object
from nle import nethack
from minihack import RewardManager
import torch


from a2c.agent import ACAgent
from a2c.model import AdvantageActorCritic
from a2c.helper import get_observation, normalize_glyphs, normalize_messages, make_video

from environment_manager import setup_environment

device = torch.device("cuda")


hyper_params = {
        'replay-buffer-size': int(5e6),
        'learning-rate': 1e-4,
        'discount-factor': 0.99,  # discount factor
        'num-steps': int(30000),  # Steps to run for, max episodes should be hit before this
        'batch-size': 32,  
        'learning-starts': 500,  # set learning to start after 1000 steps of exploration
        'learning-freq': 1,  # Optimize after each step
        'use-double-dqn': True,
        'target-update-freq': 100, # number of iterations between every target network update
        'eps-start': 1.0,  # e-greedy start threshold 
        'eps-end': 0.1,  # e-greedy end threshold 
        'eps-fraction': 0.7,  # Percentage of the time that epsilon is annealed
        'print-freq': 10,
        'seed' : 102,
        'env' : "MiniHack-Room-5x5-v0",
        'extra-info' : "config"  #config or plain
    }

env = setup_environment(hyper_params["env"], hyper_params["extra-info"])
env.seed(hyper_params["seed"])  

ac = ACAgent(
    observation_space = env.observation_space,
    action_space = env.action_space,
    use_double_dqn = None,
    lr = 0.01,
    batch_size = None,
    gamma = 0.99,
    max_episode_length = 100,
    max_episodes = 100,
    env = env
)


make_video(
    env, 
    ac,
    30,
    30,
    "test",
    100,
    "a2c.mp4"
)

