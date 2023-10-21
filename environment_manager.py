import random
import numpy as np
import gym

from dqn.helper import make_video, normalize_glyphs, distance_to_object, explore_cave, get_msg, discover_maze, discover_staircase, discover_quest_hard, discover_door
from nle import nethack
from minihack import RewardManager
import torch
from minihack import LevelGenerator


def setup_environment(env_name, config):
    if config == "plain" and env_name != "MiniHack-Skill-Custom-v0":
        return  gym.make(env_name, observation_keys = ['pixel', 'message', 'glyphs'])
    
    else:
        reward_manager = RewardManager()
        ACTIONS = tuple(nethack.CompassDirection) 

        if env_name == "MiniHack-Room-5x5-v0":
            reward_manager.add_eat_event("apple", reward = 1.0)
            reward_manager.add_location_event("staircase down", 2.0)
            reward_manager.add_custom_reward_fn(distance_to_object)

            return gym.make(env_name, 
                            observation_keys = ['pixel', 'message', 'glyphs'], 
                            actions = ACTIONS, 
                            reward_manager=reward_manager)
        
        elif env_name == "MiniHack-MazeWalk-9x9-v0":
            reward_manager.add_eat_event("apple", reward = 1.0)
            reward_manager.add_location_event("staircase down", 2.0)
            reward_manager.add_custom_reward_fn(discover_maze)
            reward_manager.add_custom_reward_fn(discover_staircase)

            return gym.make(env_name, 
                            observation_keys = ['pixel', 'message', 'glyphs'], 
                            actions = ACTIONS, 
                            reward_manager=reward_manager)

        elif env_name == "MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0":
            reward_manager.add_eat_event("apple", reward = 1.0)
            reward_manager.add_location_event("staircase down", 2.0)
            reward_manager.add_message_event(["drink"], reward = 0.2, terminal_sufficient = False)
            reward_manager.add_message_event(["float"], reward = 0.5, terminal_sufficient = False)
            reward_manager.add_message_event(["stone", "wall"], reward = -0.3, terminal_sufficient = False)

            ACTIONS += (nethack.Command.QUAFF, nethack.Command.FIRE)

            return gym.make(env_name,
                            observation_keys = ['pixel', 'message', 'glyphs'], 
                            actions = ACTIONS, 
                            reward_manager=reward_manager)
        
        elif env_name == "MiniHack-Quest-Hard-v0":
            reward_manager.add_eat_event("apple", reward = 1.0)
            reward_manager.add_location_event("staircase down", 2.0)
            reward_manager.add_custom_reward_fn(discover_quest_hard)
            reward_manager.add_custom_reward_fn(discover_door)

            return gym.make(env_name, 
                            observation_keys = ['pixel', 'message', 'glyphs'], 
                            actions = ACTIONS, 
                            reward_manager=reward_manager)
        
        elif env_name == "MiniHack-Skill-Custom-v0":
            lvl_gen = LevelGenerator(w=4, h=4)
            lvl_gen.add_object("apple", "%", place=(3, 3))
            lvl_gen.set_start_pos((0, 1))

            reward_manager.add_eat_event("apple", 1.0, terminal_sufficient = True)

            if config == "config":
                reward_manager.add_message_event(["apple", "[ynq]"], reward = 0.5, repeatable = False)

                ACTIONS += (
                    nethack.Command.EAT,
                )
                
                return gym.make(
                    "MiniHack-Skill-Custom-v0",
                    observation_keys=("glyphs", "pixel", "message"),
                    des_file=lvl_gen.get_des(),
                    reward_manager=reward_manager,
                    actions=ACTIONS,
                )
            
            else:
                return gym.make(
                    "MiniHack-Skill-Custom-v0",
                    observation_keys=("glyphs", "pixel", "message"),
                    des_file=lvl_gen.get_des(),
                    reward_manager=reward_manager,
                )
