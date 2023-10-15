from pathlib import Path
import gym
import pygame
import numpy as np
import cv2  # OpenCV for video creation
import os
import minihack
from pygame.locals import *
import torch
# from train_dqn import normalize_glyphs


def door_opened(env, prev, action, curr):
    glyphs = curr[env._observation_keys.index("chars")]
    cur_pos = glyph_pos(glyphs, ord(">"))
    if cur_pos is None:
        return 0.0
    print("Door Opened!!")
    return 1.0
    

def go_right_bonus(env, prev, action, curr):
    # Get the x coord of the @
    glyphs = curr[env._observation_keys.index("chars")]
    cur_pos = glyph_pos(glyphs, ord("@"))
    prev_pos = glyph_pos(glyphs, ord("@"))

    if cur_pos is None or prev_pos is None:
        return 0

    # Get the x coord of cur_pos
    cur_x = cur_pos[1]
    prev_x = prev_pos[1]

    # return the reward for moving to the right
    return 0.001 * (cur_x - prev_x)

def glyph_pos(glyphs, glyph):
    glyph_positions = np.where(np.asarray(glyphs) == glyph)
    assert len(glyph_positions) == 2
    if glyph_positions[0].shape[0] == 0:
        return None
    return np.array([glyph_positions[0][0], glyph_positions[1][0]], dtype=np.float32)


def distance_to_object(env, prev_obs, action, current_obs):
    glyphs = current_obs[env._observation_keys.index("chars")]
    cur_pos = glyph_pos(glyphs, ord("@"))
    staircase_pos = glyph_pos(glyphs, ord(">"))
    if staircase_pos is None:
        # Staircase has been reached
        return 0.0
    distance = np.linalg.norm(cur_pos - staircase_pos)
    distance /= np.max(glyphs.shape)
    return -distance  

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
        action, _state = agent.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
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
def make_video(env, agent, pygame_frame_rate, video_frame_rate, save_dir, max_timesteps, fname):
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
    video_filepath = Path(save_dir) / fname

    record_video(
        env, 
        agent, 
        video_filepath, 
        pygame_frame_rate, 
        video_frame_rate,
        max_timesteps
    )


# Function to record a video of agent gameplay
def view_image(env, fname):

    frame_width = env.observation_space["pixel"].shape[1]
    frame_height = env.observation_space["pixel"].shape[0]

    pygame.init()
    screen = pygame.display.set_mode((frame_width, frame_height))
    font = pygame.font.Font(None, 36)
    text_color = (255, 255, 255)

    done = False
    obs = env.reset()
    clock = pygame.time.Clock()
    
    render(obs, screen, font, text_color)

    # Capture the current frame and save it to the video
    pygame.image.save(screen, fname)

    pygame.quit()
