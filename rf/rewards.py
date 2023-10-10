import numpy as np


"""
Obs is full observation, obj is a character representing object 
"""
def get_object_position(obs, obj):
    occurances = np.where(np.array(obs[1]) == ord(obj))

    if occurances == False:
        return None
    
    return np.array([occurances[0][0], occurances[1][0]])


""""
Obs is full observation, obj is a character representing object 
"""
def staircase_reward(env, prev_obs, action, current_obs):
    player_pos = get_object_position(current_obs,  '@')
    staircase_down_pos = get_object_position(current_obs,  '>')
    
    if player_pos.all() == None or staircase_down_pos.all() == None:
        return 0.0

    distance_to_stairs = -np.linalg.norm(player_pos - staircase_down_pos) / 71

    return distance_to_stairs 

def keep_moving(env, prev_obs, action, current_obs):
    player_pos_now = get_object_position(current_obs,  '@')
    player_pos_before = get_object_position(prev_obs,  '@')

    if player_pos_now.all() == player_pos_before.all():
        return -1.0
    else:
        return 1.0


def discourage_stone(env, prev_obs, action, current_obs):
    message = current_obs["message"]
    #Convert messgae to string
    message = ''.join(chr(i) for i in message)
    print("messahe : "  + message)
    if "It's solid stone." in message:
        print("FUCK")
        return -1000
        
    return 0.5
        

    #add_message_event(msgs: List[str], reward=1, repeatable=False, terminal_required=True, terminal_sufficient=False)
