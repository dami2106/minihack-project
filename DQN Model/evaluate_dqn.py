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
        'env' : "MiniHack-Room-Random-5x5-v0",
        'extra-info' : "NothingExtra",
        'runs' : 10,
        'episodes' : 10
    }





env = gym.make(hyper_params["env"],
                observation_keys = ['pixel', 'message', 'glyphs'],
                # penalty_time=-0.1,
                # penalty_step=-0.1,
                )

# agent = DQN()
agent = torch.load(f"Agents/MiniHack-Room-Random-5x5-v0/model_NothingExtra.pt")


def run_episodes(env, agent, episodes, max_steps = 1000):
    returns = []
    steps = []

    for _ in range(episodes):
        r = 0
        step_ep = 0
        state = env.reset()
        done = False
        while step_ep <= max_steps:
            action = act(normalize_glyphs(state), agent)
            state, reward, done, info = env.step(action)
            r += reward
            step_ep += 1
            if done:
                break

        steps.append(step_ep)
        returns.append(r)

    return returns, steps

runs = []
step = []
#Run 1000 times 
for run in range(hyper_params["runs"]):
    returns, steps = run_episodes(env, agent, hyper_params["episodes"])
    runs.append(returns)
    step.append(steps)

runs = np.array(runs)


#runs is runxepisode

mean = np.mean(runs, axis=0)
std = np.std(runs, axis=0)

#plot the mean and error 
import matplotlib.pyplot as plt
plt.plot(mean, label="DQN MLP")
plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.ylim(-1, 1)
plt.title(f"Mean return of {hyper_params['runs']} runs of {hyper_params['episodes']} episodes")
plt.savefig(f"mean_return.png")





