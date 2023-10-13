import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from helper import make_video, normalize_glyphs, distance_to_object
from nle import nethack
from minihack import RewardManager
import torch

hyper_params = {
        'replay-buffer-size': int(1e6),
        'learning-rate': 0.01,
        'discount-factor': 0.99,  # discount factor
        'num-steps': int(2e5//2),  # Steps to run for, max episodes should be hit before this
        'batch-size': 32,  
        'learning-starts': 1000,  # set learning to start after 1000 steps of exploration
        'learning-freq': 1,  # Optimize after each step
        'use-double-dqn': True,
        'target-update-freq': 1000, # number of iterations between every target network update
        'eps-start': 1.0,  # e-greedy start threshold 
        'eps-end': 0.1,  # e-greedy end threshold 
        'eps-fraction': 0.3,  # Percentage of the time that epsilon is annealed
        'print-freq': 10,
        'seed' : 42,
        'env' : "MiniHack-Room-5x5-v0"

    }

np.random.seed(hyper_params["seed"])
random.seed(hyper_params["seed"])

ACTIONS = tuple(nethack.CompassDirection)
# ACTIONS = (
#     nethack.CompassDirection.N,
#     nethack.CompassDirection.E,
#     nethack.CompassDirection.S,
#     nethack.CompassDirection.W,
#     # nethack.Command.QUAFF,
#     # nethack.Command.PICKUP,
#     # nethack.Command.EAT,
#     # nethack.Command.FIRE
#     # nethack.Command.APPLY
#     )

reward_manager = RewardManager()

reward_manager.add_eat_event("apple", reward = 1.0)
# reward_manager.add_message_event(["key", "Key"], reward = 1.0, terminal_sufficient=True)
# reward_manager.add_message_event(["fixed", "wall", "stone", "Stone"], reward = -0.5, terminal_required=False, terminal_sufficient=False)
reward_manager.add_custom_reward_fn(distance_to_object)


env = gym.make(hyper_params["env"],
                observation_keys = ['pixel', 'message', 'glyphs'],
                # penalty_time=-0.1,
                # penalty_step=-0.1,
                # reward_lose=-1,
                # reward_win=5,
                # seeds = hyper_params["seed"],
                actions = ACTIONS,
                # reward_manager=reward_manager
                )

replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

agent = DQNAgent(
    env.observation_space["glyphs"],
    env.action_space,
    replay_buffer,
    hyper_params["use-double-dqn"],
    hyper_params["learning-rate"],
    hyper_params["batch-size"],
    hyper_params["discount-factor"]
)

eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
episode_rewards = [0.0]

state = env.reset() #must be pixels only 



#Mike code
lava_count = 0
key_count = 0
last_action = None
prev_action = None

old_best=-10000
new_best=-10000

best_model = None

# You move over some lava.

for t in range(hyper_params["num-steps"]):
    fraction = min(1.0, float(t) / eps_timesteps)
    eps_threshold = hyper_params["eps-start"] + fraction * (
        hyper_params["eps-end"] - hyper_params["eps-start"]
    )

    sample = random.random()
    if sample <= eps_threshold:
        action = env.action_space.sample()
    else:
        action = agent.act(normalize_glyphs(state))

    # st  = ""
    # for n in state["message"]:
    #     st += chr(n)

    # if "What do you want to drink? [f or ?*]" in st:
    #     # print("Set action to 5")
    #     action = 5
    
    # prev_action = last_action
    # last_action = action

    # if prev_action == 4:
    #     action = 5

    next_state, reward, done, _ = env.step(action)

    # st  = ""
    # for n in next_state["message"]:
    #     st += chr(n)
    # print(st)

    # next_state = normalize_glyphs(next_state)
    agent.replay_buffer.add(normalize_glyphs(state), action, reward, normalize_glyphs(next_state), float(done))
    state = next_state

    episode_rewards[-1] += reward
    if done:
        state = env.reset()

        new_best=episode_rewards[-1]
        episode_rewards.append(0.0)
        lava_count = 0
        key_count = 0
        last_action = None
        prev_action = None

    if (
        t > hyper_params["learning-starts"]
        and t % hyper_params["learning-freq"] == 0
    ):
        agent.optimise_td_loss()

    if (
        t > hyper_params["learning-starts"]
        and t % hyper_params["target-update-freq"] == 0
    ):
        agent.update_target_network()

    num_episodes = len(episode_rewards)

    if(
        new_best>old_best
    ):
        new_best=old_best
        # make_video(env, agent, 30, 30, hyper_params["env"], 1000, f"video_{t}_{new_best}.mp4")
        best_model = agent
        torch.save(agent, 'bestOne')
        

    if (
        done
        and hyper_params["print-freq"] is not None
        and len(episode_rewards) % hyper_params["print-freq"] == 0
    ):
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        print("********************************************************")
        print("steps: {}".format(t))
        print("episodes: {}".format(num_episodes))
        print("mean 100 episode reward: {}".format(mean_100ep_reward))
        print("% time spent exploring: {}".format(int(100 * eps_threshold)))
        print("********************************************************")

        
# # torch.save(agent, 'best2')
# #Load the model in 
# agent = torch.load('best2')
for r in range(3):
    make_video(env, best_model, 30, 30, hyper_params["env"], 1000, f"video_{r}_best.mp4")
    make_video(env, agent, 30, 30, hyper_params["env"], 1000, f"video_{r}_agent.mp4")