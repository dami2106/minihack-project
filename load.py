import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from dqn.visualize import make_video
from nle import nethack

import torch

mod = torch.load("bestOne")    