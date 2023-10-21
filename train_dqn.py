import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from dqn.helper import make_video, normalize_glyphs, distance_to_object, explore_cave, get_msg, discover_maze, discover_staircase, discover_quest_hard, discover_door
from nle import nethack
from minihack import RewardManager
import torch

from environment_manager import setup_environment

import os 

from minihack import LevelGenerator


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

#Create a folder using the hyperparameters
os.makedirs(f"Agents/{hyper_params['env']}", exist_ok=True)
os.makedirs(f"Agents/{hyper_params['env']}/logs", exist_ok=True)


np.random.seed(hyper_params["seed"])
random.seed(hyper_params["seed"])

env = setup_environment(hyper_params["env"], hyper_params["extra-info"])

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
info = ""
prev_mean_reward = np.inf

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

    next_state, reward, done, info = env.step(action)
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
        loss = agent.optimise_td_loss()


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
        print("********************************************************")
        print("Message: ", get_msg(state))
        print("Info: ", info)
        print("steps: {}".format(t))
        print("episodes: {}".format(num_episodes))
        print("mean 100 episode reward: {}".format(mean_100ep_reward))
        print("Best reward: ", np.max(episode_rewards[-101:-1]))
        print("episode reward: {}".format(episode_rewards[-1]))
        print("% time spent exploring: {}".format(int(100 * eps_threshold)))
        print("********************************************************")

        if prev_mean_reward > mean_100ep_reward:
            agent.save_network(f"Agents/{hyper_params['env']}/model_{hyper_params['extra-info']}_{mean_100ep_reward}.pt")
            print("Mean reward decreased ", prev_mean_reward, " -> ", mean_100ep_reward)
            # if t >= 0.4 * hyper_params["num-steps"]:
            #     print("Training complete.")
            #     break
        prev_mean_reward = mean_100ep_reward

agent.save_network(f"Agents/{hyper_params['env']}/model_{hyper_params['extra-info']}.pt")

for r in range(4):
    env.reset()
    make_video(env, agent, 30, 30, f"Agents/{hyper_params['env']}/Videos", 5000, f"video_{r}_{hyper_params['extra-info']}.mp4")
