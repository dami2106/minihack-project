from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

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
import torch
from minihack import RewardManager

from dqn.rewards import staircase_reward, pickup_key, apple_reward
from nle import nethack

from minihack.level_generator import LevelGenerator

def get_glyphs_crop(state):
    glyphs_crop = state["glyphs_crop"]
    glyphs_crop = glyphs_crop / glyphs_crop.max()  #put in the  range 0-1
    # return torch.from_numpy(glyphs_crop.reshape((1,1,21,79))).squeeze(0)
    return torch.from_numpy(glyphs_crop.reshape((1,1,9,9))).squeeze(0)

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
    text_position = (window_size[0] // 2 - text_surface.get_width() // 2, window_size[1] - text_surface.get_height() - 20)
    screen.blit(text_surface, text_position)
    pygame.display.flip()

# Function to record a video of agent gameplay
def record_video(env, agent, video_filepath, pygame_frame_rate, video_frame_rate, max_timesteps):
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_filepath), fourcc, video_frame_rate, (frame_width, frame_height))

    pygame.init()
    screen = pygame.display.set_mode((frame_width, frame_height))
    font = pygame.font.Font(None, 36)
    text_color = (255, 255, 255)

    done = False
    obs = env.reset()
    clock = pygame.time.Clock()

    steps = 1

    while not done and steps < max_timesteps:
        action = agent.act(get_glyphs_crop(obs))
        obs, _, done, _ = env.step(action)
        # obs = get_glyphs_crop(obs)
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
def visualize(env, agent, pygame_frame_rate, video_frame_rate, save_dir, max_timesteps, num):
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
    video_filepath = Path(save_dir) / f"video_{num}.mp4"

    record_video(
        env,
        agent,
        video_filepath,
        pygame_frame_rate,
        video_frame_rate,
        max_timesteps
    )




hyper_params = {
        "seed": 42,  # which seed to use
        # "env": "MiniHack-Eat-v0",  # name of the game
        "env": "MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-3,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(30000),  # total number of steps to run the environment for
        "batch-size": 64,  # number of transitions to optimize at the same time
        "learning-starts": 2000,  # number of steps before learning starts
        "learning-freq": 2,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.3,  # fraction of num-steps
        "print-freq": 10,
}


np.random.seed(hyper_params["seed"])
random.seed(hyper_params["seed"])
torch.manual_seed(42)

ACTIONS = (
        nethack.CompassDirection.N,
        nethack.CompassDirection.E,
        nethack.CompassDirection.S,
        nethack.CompassDirection.W,
        nethack.Command.PICKUP,
        nethack.Command.ZAP,
        nethack.Command.FIRE)

env = gym.make('MiniHack-LavaCross-Full-v0',
                    observation_keys = ['pixel_crop'],
                    penalty_time=-0.1,
                    penalty_step=-0.1,
                    reward_lose=-1,
                    reward_win=5,
                    seeds = [0],
                    actions = ACTIONS)


# env = WarpFrame(env)
# env = PyTorchFrame(env)
# env = FrameStack(env, 4)
state = np.array(env.reset())
replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])


model = torch.load('model')
visualize(env, model, 30, 30, "mike", 1000, 10)