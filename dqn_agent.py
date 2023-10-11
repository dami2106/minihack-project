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

reward_manager = RewardManager()
# # # reward_manager.add_custom_reward_fn(staircase_reward)
# # # reward_manager.add_custom_reward_fn(apple_reward)
# # reward_manager.add_custom_reward_fn(pickup_key)
reward_manager.add_eat_event("apple", reward = 1.0)

# ""
# # reward_manager.add_message_event(["The door opens."], reward=1.5, terminal_required=True)
reward_manager.add_message_event(["The door opens.", "You start to float in the air!", "What do you want to drink? [f or ?*]"], reward=1.0, terminal_required=False)
reward_manager.add_message_event(["It's a wall.", "The stairs are solidly fixed to the floor.",
                                  "What a strange direction! Never mind.",
                                  "You stop at the edge of the lava."], reward=-0.1, terminal_required=False, repeatable=True)

EAT_ACTIONS = tuple(nethack.CompassDirection) + (nethack.Command.EAT,) + (nethack.Command.PICKUP,)# Eat is to complete an episode by confirmation
MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS =   (
    # nethack.Command.APPLY,
    # nethack.Command.OPEN,
    # nethack.Command.PICKUP,
    # nethack.Command.WEAR,
    # nethack.Command.WIELD,
    nethack.Command.QUAFF,
    nethack.Command.FIRE,
    # nethack.Command.INVOKE,
    # nethack.Command.ZAP,
    # nethack.Command.SWAP,
    # nethack.Command.EAT
    # nethack.Command.AUTOPICKUP,
    # nethack.Command.KICK
    # nethack.Command.FIRE
)


# lvl_gen = LevelGenerator(w=8, h=8)
# lvl_gen.add_object("apple", "%")

# env = gym.make(
#     "MiniHack-Skill-Custom-v0",
#     des_file=lvl_gen.get_des(),
#     observation_keys=("glyphs_crop", "pixel", "message", "chars"),
#     reward_manager = reward_manager,
#                reward_lose=-1.0,
#                penalty_time=-0.005,
#                penalty_step=-0.1,
#     actions = EAT_ACTIONS
# )

# VISUALIZATION HERE ...
env = gym.make(hyper_params["env"],
               observation_keys=("glyphs_crop", "pixel", "message", "chars"),
               reward_manager = reward_manager,
            #    reward_lose=-1.0,
            #    penalty_time=-0.005,
            #    penalty_step=-0.1,
               actions = NAVIGATE_ACTIONS
               )


np.random.seed(hyper_params["seed"])
random.seed(hyper_params["seed"])
torch.manual_seed(42)

replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

agent = DQNAgent(
        env.observation_space["glyphs_crop"],
        env.action_space,
        replay_buffer,
        hyper_params["use-double-dqn"],
        hyper_params["learning-rate"],
        hyper_params["batch-size"],
        hyper_params["discount-factor"]
)



eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
episode_rewards = [0.0]
episode_loss = []

state = get_glyphs_crop(env.reset())



for t in tqdm(range(hyper_params["num-steps"])):
    fraction = min(1.0, float(t) / eps_timesteps)
    eps_threshold = hyper_params["eps-start"] + fraction * (
        hyper_params["eps-end"] - hyper_params["eps-start"]
    )
    sample = random.random()


    if sample <= eps_threshold:
        # Exploring
        action = env.action_space.sample()
    else:
        # Exploiting
        action = agent.act(state)

    next_state, reward, done, _ = env.step(action)
    next_state = get_glyphs_crop(next_state)
    agent.replay_buffer.add(state, action, reward, next_state, float(done))
    state = next_state

    episode_rewards[-1] += reward
    if done:
        state = get_glyphs_crop(env.reset())
        episode_rewards.append(0.0)

    if (
        t > hyper_params["learning-starts"]
        and t % hyper_params["learning-freq"] == 0
    ):
        loss_ = agent.optimise_td_loss()
        episode_loss.append(loss_)

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
        print("\n********************************************************")
        print("steps: {}".format(t))
        print("episodes: {}".format(num_episodes))
        print("mean 100 episode reward: {}".format(mean_100ep_reward))
        print("% time spent exploring: {}".format(int(100 * eps_threshold)))
        print("********************************************************\n")

# env = gym.make("MiniHack-Room-15x15-v0",
#                observation_keys=("glyphs_crop", "pixel", "message", "chars"),
#             #    reward_manager = reward_manager,
#             #    reward_lose=-1.0,
#             #    penalty_time=-0.005,
#             #    penalty_step=-0.1,
#                actions = NAVIGATE_ACTIONS
#                )

for vid in range(3):
    env.reset()
    visualize(
        env,
        agent,
        pygame_frame_rate=60,
        video_frame_rate=5,
        save_dir="new_videos",
        max_timesteps=1000,
        num = vid
    )
