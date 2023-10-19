#Load the model in 
import torch
import gym 
from helper import normalize_glyphs
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def act(observation, model):
    """Select action base on network inference"""
    if not torch.cuda.is_available():
        observation = observation.type(torch.FloatTensor) 
    else:
        observation = observation.type(torch.cuda.FloatTensor) 
    state = torch.unsqueeze(observation, 0).to(device)
    result = model.forward(state)
    action = torch.argmax(result).item()
    return action


hyper_params = {
        'env' : "MiniHack-Room-15x15-v0",
        'extra-info' : "NothingExtra",
        'runs' : 10
    }





env = gym.make(hyper_params["env"],
                observation_keys = ['pixel', 'message', 'glyphs'],
                )

# agent = DQN()
agent = torch.load(f"Agents/MiniHack-Room-5x5-v0/model_NothingExtra_mlp.pt")


def run_episodes(env, agent, episodes):
    

runs = []
#Run 1000 times 
for run in range(hyper_params["runs"]):
    state = env.reset()
    done = False
    episode = []
    #Each episode 
    while True:
        action = act(normalize_glyphs(state), agent)
        state, reward, done, info = env.step(action)
        episode.append(reward)
        # print(info)
        if done:
            break
    runs.append(np.sum(episode))

runs = np.array(runs)


#runs is runxepisode
print(runs)



#List of returns for n episodes 
# Then have 100 lists of these returns 





