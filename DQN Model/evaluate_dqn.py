#Load the model in 
import torch
import gym 
from helper import normalize_glyphs, discover_maze, discover_staircase, make_video
import numpy as np
from nle import nethack
from minihack import RewardManager
device = "cuda" if torch.cuda.is_available() else "cpu"


class DQN:
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        """Select action base on network inference"""
        if not torch.cuda.is_available():
            observation = observation.type(torch.FloatTensor) 
        else:
            observation = observation.type(torch.cuda.FloatTensor) 
        state = torch.unsqueeze(observation, 0).to(device)
        result = self.model.forward(state)
        action = torch.argmax(result).item()
        return action

hyper_params = {
        'env' : "MiniHack-MazeWalk-9x9-v0",
        'extra-info' : "NothingExtra",
        'runs' : 100,
        'episodes' : 1
    }


reward_manager = RewardManager()
reward_manager.add_eat_event("apple", reward = 1.0)
# reward_manager.add_message_event(["key", "Key"], reward = 1.0, terminal_sufficient=True)
# reward_manager.add_message_event(["fixed", "wall", "stone", "Stone", "solid"], reward = -0.4, terminal_required=False, terminal_sufficient=False)
# reward_manager.add_custom_reward_fn(distance_to_object)
reward_manager.add_custom_reward_fn(discover_staircase)
reward_manager.add_custom_reward_fn(discover_maze)
reward_manager.add_location_event("staircase down", 2.0)
# reward_manager = RewardManager()

# reward_manager.add_eat_event("apple", reward = 1.0)
# reward_manager.add_location_event("staircase down", 1.0)
# reward_manager.add_message_event(["drink"], reward = 0.2, terminal_sufficient = False)
# reward_manager.add_message_event(["float"], reward = 0.5, terminal_sufficient = False)
# reward_manager.add_message_event(["stone", "wall"], reward = -0.3, terminal_sufficient = False)

MOVE_ACTIONS =  tuple(nethack.CompassDirection) +(
    nethack.Command.QUAFF,
    nethack.Command.FIRE,
)

env = gym.make(hyper_params["env"],
                observation_keys = ['pixel', 'message', 'glyphs'],
                # penalty_time=-0.1,
                # penalty_step=-0.1,
                reward_manager = reward_manager,
                actions =  tuple(nethack.CompassDirection),
                )




agent = DQN(torch.load(f"Agents/MiniHack-MazeWalk-9x9-v0/model_plain_2.2.pt"))

make_video(env, agent, 30, 30, "video", 6000, "test_eval.mp4")

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
            action = agent.act(normalize_glyphs(state), agent)
            actions.append(action)
            state, reward, done, info = env.step(action)
            if(info["end_status"] == 2):
                print(actions)
                actions  = []
            r += reward
            step_ep += 1
            if done:
                break

        steps.append(step_ep)
        returns.append(r)

    return returns, steps





# runs = []
# step = []
# #Run 1000 times 
# for run in range(hyper_params["runs"]):
#     returns, steps = run_episodes(env, agent, hyper_params["episodes"])
#     runs.append(returns)
#     step.append(steps)

# runs = np.array(runs)


#runs is runxepisode

# mean = np.mean(runs, axis=0)
# std = np.std(runs, axis=0)

# #plot the mean and error 
# import matplotlib.pyplot as plt
# plt.plot(mean, label="DQN MLP")
# plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
# plt.xlabel("Episode")
# plt.ylabel("Return")
# plt.ylim(-1, 1)
# plt.title(f"Mean return of {hyper_params['runs']} runs of {hyper_params['episodes']} episodes")
# plt.savefig(f"mean_return.png")





