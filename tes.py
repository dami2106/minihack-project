# %%
# from dqn.agent import Agent
import random
import numpy as np
import gym

from nle import nethack
from minihack import RewardManager
import torch
from helper import view_image, distance_to_object, get_msg, msg_rew
# from minihack import StepStatus

import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# %%

reward_manager = RewardManager()
reward_manager.add_message_event(["float"], reward = 100.0, terminal_sufficient = True)
reward_manager.add_message_event(["drink"], reward = 25.0)
reward_manager.add_custom_reward_fn(distance_to_object)
# reward_manager.add_custom_reward_fn(msg_rew)
reward_manager.add_eat_event("apple", reward = 1.0)
# reward_manager.add_location_event("staircase down", reward = 1.0)
MOVE_ACTIONS = tuple(nethack.CompassDirection) + (
    nethack.Command.QUAFF,
    nethack.Command.FIRE,
)

env = gym.make("MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0",
                observation_keys = ['pixel', 'message', 'glyphs', 'blstats', 'chars', 'pixel_crop'],
                obs_crop_h = 15,
                obs_crop_w = 15,
                # penalty_time=-0.1,
                # penalty_step=-0.1,
                # reward_lose=-1,
                # reward_win=5,
                # seeds = hyper_params["seed"],
                actions = MOVE_ACTIONS,
                reward_manager=reward_manager
                )

state = env.reset()
# print(state["pixel_crop"].shape)

# view_image(env, "test.png")

# env.step(1)

actions = [8, 9]

for i in range(2):
    #Generate a random int between 0 and 4
    # action = random.randint(0, env.action_space.n)
    action = actions[i]
    state, reward, done, info = env.step(action)
    print("Action: ", action)
    print("Reward: ", reward)
    print("Done: ", done)
    print("Info: ", info)
    print(info["end_status"]) #<StepStatus.TASK_SUCCESSFUL: 2>
    print(info["end_status"] == 2)
    print("Message:", get_msg(state))
    view_image(env, state, f"test{i}.png")