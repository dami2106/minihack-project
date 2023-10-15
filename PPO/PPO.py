import stable_baselines3
import gymnasium as gym
import numpy as np
import random
import gym
from nle import nethack
from minihack import RewardManager
import torch
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy



from helper import make_video, distance_to_object, go_right_bonus, door_opened

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    nethack.Command.EAT,
)
# reward_manager = RewardManager()
# reward_manager.add_eat_event("apple", reward = 1.0)
# reward_manager.add_custom_reward_fn(door_opened)
# # reward_manager.add_custom_reward_fn(go_right_bonus)
# reward_manager.add_location_event("staircase down", 2.0, terminal_sufficient=True)
# reward_manager.add_message_event(["fixed", "wall", "stone", "Stone"], reward = -0.5, terminal_required=False, terminal_sufficient=False)
# reward_manager.add_message_event(["g - a key"], reward = 0.5, terminal_required=False, terminal_sufficient=False)

# ["g - a key"]
# ["wall"]

env_name = "MiniHack-Eat-v0"



env = gym.make(env_name,
                observation_keys = ['pixel', 'message', 'glyphs', 'specials', 'screen_descriptions'],
                actions = NAVIGATE_ACTIONS,
                # reward_manager=reward_manager
                )

model = PPO(
    "MultiInputPolicy", 
    env, 
    verbose=1,
    learning_rate = 1e-5,
    batch_size = 128,
    seed = SEED,
    ent_coef = 0.0001,
    vf_coef = 0.5,
    n_steps = 256,
    n_epochs = 20)


model.learn(total_timesteps=15000)

for k in range(3):
    env.reset()
    make_video(
        env,
        model,
        pygame_frame_rate=10,
        video_frame_rate=10,
        save_dir=env_name,
        fname=f"video_{k}.mp4",
        max_timesteps=10000,
    )