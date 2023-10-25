import numpy as np
import gym
from nle import nethack
from minihack import RewardManager
from minihack import LevelGenerator

#A function to get a string of the message of the current observation
def get_msg(obs):
    msg = obs["message"]
    msg = msg.tobytes().decode("utf-8")
    return msg

#A function that gets the coordinates of a given glyph item in the observation
def glyph_pos(glyphs, glyph):
    glyph_positions = np.where(np.asarray(glyphs) == glyph)
    assert len(glyph_positions) == 2
    if glyph_positions[0].shape[0] == 0:
        return None
    return np.array([glyph_positions[0][0], glyph_positions[1][0]], dtype=np.float32)

#A function that gets the inverse distance to the downard staircase 
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

#A function that gives a reward for expploring corridors of a cave 
#(characterised by the number of full stops uncovered)
def discover_maze(env, prev_obs, action, current_obs):
    curr_chars = current_obs[env._observation_keys.index("chars")]
    prev_chars = prev_obs[env._observation_keys.index("chars")]

    curr_dots = 0
    prev_dots = 0

    for row in curr_chars:
        for char in row:
            if char == ord("."):
                curr_dots += 1

    for row in prev_chars:
        for char in row:
            if char == ord("."):
                prev_dots += 1

    if curr_dots > prev_dots:
        return 0.1

    return 0.0

#A function that gives a reward for expploring corridors of a cave 
#(characterised by the number of hastags uncovered in quest hard)
def discover_quest_hard(env, prev_obs, action, current_obs):
    curr_chars = current_obs[env._observation_keys.index("chars")]
    prev_chars = prev_obs[env._observation_keys.index("chars")]

    curr_dots = 0
    prev_dots = 0

    for row in curr_chars:
        for char in row:
            if char == ord("#"):
                curr_dots += 1

    for row in prev_chars:
        for char in row:
            if char == ord("#"):
                prev_dots += 1

    if curr_dots > prev_dots:
        return 0.1

    return 0.0

#A function that gives a reward for expploring corridors of a cave and discovering the closed door
def discover_door(env, prev_obs, action, current_obs):
    curr_chars = current_obs[env._observation_keys.index("chars")]
    prev_chars = prev_obs[env._observation_keys.index("chars")]

    curr_dots = 0
    prev_dots = 0

    for row in curr_chars:
        for char in row:
            if char == ord("+"):
                curr_dots += 1

    for row in prev_chars:
        for char in row:
            if char == ord("+"):
                prev_dots += 1

    if curr_dots > prev_dots:
        return 0.1

    return 0.0

#A function that gives a reward for expploring corridors of a cave and discovering the downward staircase
def discover_staircase(env, prev_obs, action, current_obs):
    curr_chars = current_obs[env._observation_keys.index("chars")]
    prev_chars = prev_obs[env._observation_keys.index("chars")]

    curr_staircase = 0
    prev_staircase = 0

    for row in curr_chars:
        for char in row:
            if char == ord(">"):
                curr_staircase += 1

    for row in prev_chars:
        for char in row:
            if char == ord(">"):
                prev_staircase += 1

    if curr_staircase > prev_staircase:
        return 0.5

    return 0.0

"""
A function to setup the given environment with the given configuration
"""
def setup_environment(env_name, config):
    #Plain config, no rewards or custom actions or custom envs
    if config == "plain" and env_name != "MiniHack-Skill-Custom-v0":
        return  gym.make(env_name, observation_keys = ['pixel', 'message', 'glyphs'])
    
    #Custom environemnt rewards, action
    else:
        reward_manager = RewardManager()
        ACTIONS = tuple(nethack.CompassDirection) 

        #Custom room 5x5 configuration
        if env_name == "MiniHack-Room-5x5-v0":
            reward_manager.add_eat_event("apple", reward = 1.0)
            reward_manager.add_location_event("staircase down", 2.0)
            reward_manager.add_custom_reward_fn(distance_to_object)

            return gym.make(env_name, 
                            observation_keys = ['pixel', 'message', 'glyphs'], 
                            actions = ACTIONS, 
                            reward_manager=reward_manager)
        
        #Custom mazewalk configuration
        elif env_name == "MiniHack-MazeWalk-9x9-v0":
            reward_manager.add_eat_event("apple", reward = 1.0)
            reward_manager.add_location_event("staircase down", 2.0)
            reward_manager.add_custom_reward_fn(discover_maze)
            reward_manager.add_custom_reward_fn(discover_staircase)

            return gym.make(env_name, 
                            observation_keys = ['pixel', 'message', 'glyphs'], 
                            actions = ACTIONS, 
                            reward_manager=reward_manager)

        #Custom lava cross configuration
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
        
        #Custom quest hard configuration
        elif env_name == "MiniHack-Quest-Hard-v0":
            reward_manager.add_eat_event("apple", reward = 1.0)
            reward_manager.add_location_event("staircase down", 2.0)
            reward_manager.add_custom_reward_fn(discover_quest_hard)
            reward_manager.add_custom_reward_fn(discover_door)

            return gym.make(env_name, 
                            observation_keys = ['pixel', 'message', 'glyphs'], 
                            actions = ACTIONS, 
                            reward_manager=reward_manager)
        
        #Custom Eating Apples configuration with a custom environment
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
