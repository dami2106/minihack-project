import numpy as np


"""
Obs is full observation, obj is a character representing object 
"""
def get_object_position(obs, obj):
    occurances = np.where(np.array(obs["chars"]) == ord(obj))
    if occurances == False:
        return None
    return np.array([occurances[0][0], occurances[1][0]])


""""
Obs is full observation, obj is a character representing object 
"""
def staircase_reward(env, prev_obs, action, current_obs):
    player_pos = get_object_position(current_obs,  '@')
    staircase_down_pos = get_object_position(current_obs,  '>')
    
    if player_pos == None or staircase_down_pos == None:
        return 0.0
    
    distance_to_stairs = - np.linalg.norm(player_pos - staircase_down_pos) / np.max(current_obs.shape)

    return distance_to_stairs
