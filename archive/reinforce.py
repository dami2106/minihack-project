# @title Imports
from pathlib import Path
import gym
import pygame
import numpy as np
import cv2  # OpenCV for video creation
import os
import minihack
from pygame.locals import *
import random
from tqdm import tqdm

from gym import spaces
import numpy as np


import torch

from gym import spaces
import torch.nn as nn

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)

import torch.optim as optim

from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from minihack import RewardManager

# from reinforce.rewards import staircase_reward

from rf.model import Policy

from rf.rewards import staircase_reward, keep_moving, discourage_stone


# @title Provided Visualization Code
def get_glyphs(state):
    glyphs = state["glyphs"]
    glyphs = glyphs / glyphs.max()
    return glyphs


# Function to scale an observation to a new size using Pygame
def scale_observation(observation, new_size):
    """
    Scale an observation (image) to a new size using Pygame.
    Args:
        observation (pygame.Surface): The input Pygame observation.
        new_size (tuple): The new size (width, height) for scaling.
    Returns:
        pygame.Surface: The scaled observation.
    """
    return pygame.transform.scale(observation, new_size)


# Function to render the game observation
def render(obs, screen, font, text_color):
    """
    Render the game observation on the Pygame screen.
    Args:
        obs (dict): Observation dictionary containing "pixel" and "message" keys.
        screen (pygame.Surface): The Pygame screen to render on.
        font (pygame.Font): The Pygame font for rendering text.
        text_color (tuple): The color for rendering text.
    """
    img = obs["pixel"]
    msg = obs["message"]
    msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")
    rotated_array = np.rot90(img, k=-1)

    window_size = screen.get_size()
    image_surface = pygame.surfarray.make_surface(rotated_array)
    image_surface = scale_observation(image_surface, window_size)

    screen.fill((0, 0, 0))
    screen.blit(image_surface, (0, 0))

    text_surface = font.render(msg, True, text_color)
    text_position = (
        window_size[0] // 2 - text_surface.get_width() // 2,
        window_size[1] - text_surface.get_height() - 20,
    )
    screen.blit(text_surface, text_position)
    pygame.display.flip()


# Function to record a video of agent gameplay
def record_video(
    env, agent, video_filepath, pygame_frame_rate, video_frame_rate, max_timesteps
):
    """
    Record a video of agent's gameplay and save it as an MP4 file.
    Args:
        env (gym.Env): The environment in which the agent plays.
        agent (object): The agent that interacts with the environment.
        video_filepath (Path): The file path where the video will be saved.
        pygame_frame_rate (int): Frame rate for rendering the video.
        video_frame_rate (int): Frame rate for the output video.
        max_timesteps (int): Maximum number of timesteps to record in the video.
    """
    frame_width = env.observation_space["pixel"].shape[1]
    frame_height = env.observation_space["pixel"].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(video_filepath), fourcc, video_frame_rate, (frame_width, frame_height)
    )

    pygame.init()
    screen = pygame.display.set_mode((frame_width, frame_height))
    font = pygame.font.Font(None, 36)
    text_color = (255, 255, 255)

    done = False
    obs = env.reset()
    clock = pygame.time.Clock()

    steps = 1

    while not done and steps < max_timesteps:
        action = agent.act(get_glyphs(obs))
        obs, _, done, _ = env.step(action)
        # obs = get_glyphs(obs)
        render(obs, screen, font, text_color)

        # Capture the current frame and save it to the video
        pygame.image.save(screen, "temp_frame.png")
        frame = cv2.imread("temp_frame.png")
        out.write(frame)

        clock.tick(pygame_frame_rate)
        steps += 1

    out.release()  # Release the video writer
    cv2.destroyAllWindows()  # Close any OpenCV windows
    os.remove("temp_frame.png")  # Remove the temporary frame file


# Function to visualize agent's gameplay and save it as a video
def visualize(env, agent, pygame_frame_rate, video_frame_rate, save_dir, max_timesteps):
    """
    Visualize agent's gameplay and save it as a video.
    Args:
        env (gym.Env): The environment in which the agent plays.
        agent (object): The agent that interacts with the environment.
        pygame_frame_rate (int): Frame rate for rendering on the pygame screen.
        video_frame_rate (int): Frame rate for the output video.
        save_dir (str): Directory where the video will be saved.
        max_timesteps (int): Maximum number of timesteps to record in the video.
    """
    os.makedirs(save_dir, exist_ok=True)
    video_filepath = Path(save_dir) / "video.mp4"

    record_video(
        env, agent, video_filepath, pygame_frame_rate, video_frame_rate, max_timesteps
    )


hyper_params = {
    "seed": 42,  # which seed to use
    "env": "MiniHack-Room-5x5-v0",  # name of the game
    "replay-buffer-size": int(1e6),  # replay buffer size
    "learning-rate": 0.01,  # learning rate for Adam optimizer
    "discount-factor": 0.99,  # discount factor
    "num-steps": int(1e4),  # total number of steps to run the environment for
    "batch-size": 32,  # number of transitions to optimize at the same time
    "learning-starts": 1000,  # number of steps before learning starts
    "learning-freq": 1,  # number of iterations between every optimization step
    "use-double-dqn": True,  # use double deep Q-learning
    "target-update-freq": 1000,  # number of iterations between every target network update
    "eps-start": 1.0,  # e-greedy start threshold
    "eps-end": 0.1,  # e-greedy end threshold
    "eps-fraction": 0.1,  # fraction of num-steps
    "print-freq": 10,
}


rm = RewardManager()
# rm.add_custom_reward_fn(staircase_reward)
rm.add_location_event("stairs", reward=-1, terminal_required=False)
# reward_manager.add_custom_reward_fn(keep_moving)
rm.add_custom_reward_fn(discourage_stone)

env = gym.make(hyper_params["env"], observation_keys=("glyphs", "pixel", "message"))

# reward_manager.add_message_event(["It's solid stone."], reward=-10000, repeatable=True, terminal_required=False, terminal_sufficient=False)

np.random.seed(hyper_params["seed"])
random.seed(hyper_params["seed"])

policy = Policy(env.action_space)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + 0.9 * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def select_action(state):
    state = get_glyphs(state)
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    print(probs)
    m = Categorical(probs)
    print(m)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()



state = env.reset()

print(env._observation_keys)

# print(state["message"])


# st = ""
# for fuck in state["message"]:
#     st += chr(fuck)

# print(st)

# action = select_action(state)
# state, reward, done, _ = env.step(action)
# print("NEW MESSAGE:")
# st = ""
# for fuck in state["message"]:
#     st += chr(fuck)

# print("*" + st + "*")
# print(reward)
# action = select_action(state)
# state, reward, done, _ = env.step(action)
# print("NEW MESSAGE:")
# st = ""
# for fuck in state["message"]:
#     st += chr(fuck)

# print("*" + st + "*")
# print(reward)

# print(st)
# running_reward = 10
# for i_episode in range(10):
#     state = env.reset()

#     action = select_action(state)

#     ep_reward = 0

#     for t in range(1, 10):  # Don't infinite loop while learning
#         action = select_action(state)
#         state, reward, done, _ = env.step(action)
#         # if args.render:
#         #     env.render()
#         policy.rewards.append(reward)
#         ep_reward += reward
#         if done:
#             break

#     running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
#     finish_episode()
#     if i_episode % 2 == 0:
#         print(
#             "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
#                 i_episode, ep_reward, running_reward
#             )
#         )
#         st = ""
#         for fuck in state["message"]:
#             st += chr(fuck)
#         print(st)


