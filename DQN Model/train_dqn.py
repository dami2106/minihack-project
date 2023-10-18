import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from helper import make_video, normalize_glyphs, distance_to_object, explore_cave
from nle import nethack
from minihack import RewardManager
import torch

import os 

from torch.utils.tensorboard import SummaryWriter




hyper_params = {
        'replay-buffer-size': int(1e6),
        'learning-rate': 1e-4,
        'discount-factor': 0.99,  # discount factor
        'num-steps': int(50000),  # Steps to run for, max episodes should be hit before this
        'batch-size': 32,  
        'learning-starts': 1000,  # set learning to start after 1000 steps of exploration
        'learning-freq': 1,  # Optimize after each step
        'use-double-dqn': True,
        'target-update-freq': 500, # number of iterations between every target network update
        'eps-start': 1.0,  # e-greedy start threshold 
        'eps-end': 0.1,  # e-greedy end threshold 
        'eps-fraction': 0.3,  # Percentage of the time that epsilon is annealed
        'print-freq': 10,
        'seed' : 69,
        'env' : "MiniHack-Room-5x5-v0",
        'extra-info' : "NothingExtra"
    }

#Create a folder using the hyperparameters
os.makedirs(f"Agents/{hyper_params['env']}", exist_ok=True)
os.makedirs(f"Agents/{hyper_params['env']}/logs", exist_ok=True)
writer = SummaryWriter(log_dir=f"Agents/{hyper_params['env']}/logs")

np.random.seed(hyper_params["seed"])
random.seed(hyper_params["seed"])
os.environ['PYTHONHASHSEED'] = str(hyper_params["seed"])
torch.manual_seed(hyper_params["seed"])
torch.cuda.manual_seed(hyper_params["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

reward_manager = RewardManager()

reward_manager.add_eat_event("apple", reward = 1.0)
# reward_manager.add_message_event(["key", "Key"], reward = 1.0, terminal_sufficient=True)
# reward_manager.add_message_event(["fixed", "wall", "stone", "Stone", "solid"], reward = -0.5, terminal_required=False, terminal_sufficient=False)
reward_manager.add_custom_reward_fn(distance_to_object)
reward_manager.add_location_event("staircase down", 1.0, terminal_sufficient=True)
# reward_manager.add_custom_reward_fn(explore_cave)

MOVE_ACTIONS = tuple(nethack.CompassDirection) + (
    nethack.Command.QUAFF,
    nethack.Command.FIRE,
)

env = gym.make(hyper_params["env"],
                observation_keys = ['pixel', 'message', 'glyphs'],
                # penalty_time=-0.1,
                # penalty_step=-0.1,
                # reward_lose=-1,
                # reward_win=5,
                # seeds = hyper_params["seed"],
                actions = MOVE_ACTIONS,
                reward_manager=reward_manager
                )

env.seed(hyper_params["seed"])  

replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

agent = DQNAgent(
    env.observation_space["glyphs"],
    env.action_space,
    replay_buffer,
    hyper_params["use-double-dqn"],
    hyper_params["learning-rate"],
    hyper_params["batch-size"],
    hyper_params["discount-factor"]
)

eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
episode_rewards = [0.0]

state = env.reset() 


for t in range(hyper_params["num-steps"]):
    fraction = min(1.0, float(t) / eps_timesteps)
    eps_threshold = hyper_params["eps-start"] + fraction * (
        hyper_params["eps-end"] - hyper_params["eps-start"]
    )

    sample = random.random()
    if sample <= eps_threshold:
        action = env.action_space.sample()
    else:
        action = agent.act(normalize_glyphs(state))


    next_state, reward, done, _ = env.step(action)
    writer.add_scalar(f'DQN/{hyper_params["env"]}/Reward', reward, t)

    # next_state = normalize_glyphs(next_state)
    agent.replay_buffer.add(normalize_glyphs(state), action, reward, normalize_glyphs(next_state), float(done))
    state = next_state

    episode_rewards[-1] += reward
    if done:
        state = env.reset()
        episode_rewards.append(0.0)
    if (
        t > hyper_params["learning-starts"]
        and t % hyper_params["learning-freq"] == 0
    ):
        agent.optimise_td_loss()

    if (
        t > hyper_params["learning-starts"]
        and t % hyper_params["target-update-freq"] == 0
    ):
        agent.update_target_network()

    num_episodes = len(episode_rewards)
        
    if (
        done
        and hyper_params["print-freq"] is not None
        and len(episode_rewards) % hyper_params["print-freq"] == 0
    ):
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        writer.add_scalar(f'DQN/{hyper_params["env"]}/Average 100ep Reward', mean_100ep_reward, t)
        writer.add_scalar(f'DQN/{hyper_params["env"]}/Epsilon', eps_threshold, t)
        writer.add_scalar(f'DQN{hyper_params["env"]}//% time spent exploring', int(100 * eps_threshold), t)

        print("********************************************************")
        print("steps: {}".format(t))
        print("episodes: {}".format(num_episodes))
        print("mean 100 episode reward: {}".format(mean_100ep_reward))
        print("% time spent exploring: {}".format(int(100 * eps_threshold)))
        print("********************************************************")

        
writer.flush()
writer.close()
agent.save_network(f"Agents/{hyper_params['env']}/model_{hyper_params['extra-info']}.pt")

for r in range(3):
    env.reset()
    make_video(env, agent, 30, 30, f"Agents/{hyper_params['env']}/Videos", 1000, f"video_{r}_{hyper_params['extra-info']}.mp4")

env = gym.make("MiniHack-LavaCross-Levitate-Ring-Inv-Full-v0",
                observation_keys = ['pixel', 'message', 'glyphs'],
                # penalty_time=-0.1,
                # penalty_step=-0.1,
                # reward_lose=-1,
                # reward_win=5,
                # seeds = hyper_params["seed"],
                actions = MOVE_ACTIONS,
                # reward_manager=reward_manager
                )

for r in range(3):
    env.reset()
    make_video(env, agent, 30, 30, f"Agents/{hyper_params['env']}/Videos", 1000, f"videofire_{r}_{hyper_params['extra-info']}.mp4")