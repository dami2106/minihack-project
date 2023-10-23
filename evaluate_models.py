#Load the model in 
import torch
import gym 
# from helper import flatten_glyphs, get_observation
from dqn.helper import normalize_glyphs
from a2c.helper import get_observation
import numpy as np
from nle import nethack
from minihack import RewardManager
from tqdm import tqdm

from environment_manager import setup_environment

device = "cuda" if torch.cuda.is_available() else "cpu"


#DQN class to manage the DQN model and agent
class DQN:
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        observation = normalize_glyphs(observation)
        """Select action base on network inference"""
        if not torch.cuda.is_available():
            observation = observation.type(torch.FloatTensor) 
        else:
            observation = observation.type(torch.cuda.FloatTensor) 
        state = torch.unsqueeze(observation, 0).to(device)
        result = self.model.forward(state)
        action = torch.argmax(result).item()
        return action
    
#A2C class to manage the A2C model and agent
class A2C:
    def __init__(self, model):
        self.model = model
    
    def act(self, observation):
        state = get_observation(observation)
        act, _ = self.model.forward(state)
        best_action = torch.distributions.Categorical(act).sample()
        return best_action.item()

#A function that collects the average return over N episodes for max time steps 1000
def run_episodes(env, agent, episodes, max_steps = 1000):
    returns = []
    steps = []

    for _ in range(episodes):
        actions = []
        r = 0
        step_ep = 0
        state = env.reset()
        done = False
        while step_ep <= max_steps:
            action = agent.act(state)
            actions.append(action)
            state, reward, done, info = env.step(action)
    
            r += reward
            step_ep += 1
            if done:
                break

        steps.append(step_ep)
        returns.append(r)

    return returns, steps

#Setup the testing env
hyper_params = {
        'env' : "MiniHack-MazeWalk-9x9-v0", #Name of folder in  runs folder
        'type' : "config",   #config or plain 
        'runs' : 10,    #Number of times to run set of episodes
        'episodes' : 100 #Number of episodes to run
    }
env = setup_environment(hyper_params["env"], hyper_params["type"])

#Load the saved models in
dqn_agent = DQN(torch.load(f"Saved_Runs/{hyper_params['env']}/DQN/DQN_{hyper_params['type']}.pt"))
a2c_agent = A2C(torch.load(f"Saved_Runs/{hyper_params['env']}/A2C/A2C_{hyper_params['type']}.pt"))


#A a loop to evaluate both models and collect the steps, returns etc for each run as well as the variance
for i in range(2):
    agent = dqn_agent if i == 0 else a2c_agent

    runs = []
    step = []
    for run in tqdm(range(hyper_params["runs"])):
        returns, steps = run_episodes(env, agent, hyper_params["episodes"])
        runs.append(returns)
        step.append(steps)

    runs = np.array(runs)
    step = np.array(step)

    mean_runs = np.mean(runs, axis=0)
    var_runs = np.var(runs, axis=0)

    mean_steps = np.mean(step, axis=0)
    var_steps = np.var(step, axis=0)

    agent_data = [
        mean_runs,
        var_runs,
        mean_steps,
        var_steps
    ]

    agent_data = np.array(agent_data)

    if i == 0:
        np.save(f"Saved_Runs/{hyper_params['env']}/{hyper_params['type']}_DQN.npy", agent_data)
    else:
        np.save(f"Saved_Runs/{hyper_params['env']}/{hyper_params['type']}_A2C.npy", agent_data)








