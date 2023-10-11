import numpy as np


"""
Obs is full observation, obj is a character representing object 
"""
def get_object_position(obs, obj):
    occurances = np.where(np.array(obs[1]) == ord(obj))

    if occurances[0].shape[0] == 0:
        return np.array([-69, -69])
    
    return np.array([occurances[0][0], occurances[1][0]])




""""
Obs is full observation, obj is a character representing object 
"""
def staircase_reward(env, prev_obs, action, current_obs):
    player_pos = get_object_position(current_obs,  '@')
    staircase_down_pos = get_object_position(current_obs,  '>')
    
    if player_pos[0] == -69 or staircase_down_pos[0] == -69:
        return 0.0
    
    distance_to_stairs = -np.linalg.norm(player_pos - staircase_down_pos) / np.max(current_obs[0].shape)

    return distance_to_stairs

""""
Obs is full observation, obj is a character representing object 
"""
def apple_reward(env, prev_obs, action, current_obs):
    player_pos = get_object_position(current_obs,  '@')
    staircase_down_pos = get_object_position(current_obs,  '%')
    
    if player_pos[0] == -69 or staircase_down_pos[0] == -69:
        return 0.0
    
    distance_to_stairs = -np.linalg.norm(player_pos - staircase_down_pos) / np.max(current_obs[0].shape)

    return distance_to_stairs


"""
Change this 
# """
def pickup_key(env, prev_obs, action, current_obs):
    #Key is '('200
    key_current = np.where(np.array(current_obs[1]) == ord('('))
    key_prev = np.where(np.array(prev_obs[1]) == ord('('))

    # if key_current[0].shape[0] == 0 and key_prev[0].shape[0] != 0:
    #     return 1.2

    # #GET REWARD FOR PICKING UP KEY
    # if env.key_in_inventory("key") != None:
    #     print("<<<<<<<<<<<<<<<<<<KEY IN INVENTORY>>>>>>>>>>>>>>>>>!@##$$%^&*")
    #     return 1.1
    # return -0.5
    
