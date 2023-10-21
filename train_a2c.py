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

device = torch.device("cuda")


env = gym.make(
    "MiniHack-Room-5x5-v0",
    observation_keys = ['pixel', 'message', 'glyphs'],
)


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

