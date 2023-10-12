import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from dqn.visualize import make_video
from nle import nethack
from minihack import RewardManager
import torch

def normalize_glyphs(state):
    glyphs = state["glyphs"]
    glyphs = glyphs/glyphs.max()
    return torch.from_numpy(glyphs.reshape((1,1,21,79))).squeeze(0)

hyper_params = {
    # "seed": 42,  # which seed to use
    "env": "MiniHack-MazeWalk-9x9-v0",  # name of the game
    "replay-buffer-size": int(5e3),  # replay buffer size
    "learning-rate": 1e-4,  # learning rate for Adam optimizer
    "discount-factor": 1,  # discount factor
    "num-steps": int(50000),  # total number of steps to run the environment for
    "batch-size": 256,  # number of transitions to optimize at the same time
    "learning-starts": 5000,  # number of steps before learning starts
    "learning-freq": 5,  # number of iterations between every optimization step
    "use-double-dqn": True,  # use double deep Q-learning
    "target-update-freq": 1000,  # number of iterations between every target network update
    "eps-start": 1.0,  # e-greedy start threshold
    "eps-end": 0.01,  # e-greedy end threshold
    "eps-fraction": 0.6,  # fraction of num-steps
    "print-freq": 10,
}


# np.random.seed(hyper_params["seed"])
# random.seed(hyper_params["seed"])

# ACTIONS = nethack.CompassDirection.all_directions() + nethack.Command.all_directions() 
ACTIONS = (
    nethack.CompassDirection.N,
    nethack.CompassDirection.E,
    nethack.CompassDirection.S,
    nethack.CompassDirection.W,
    # nethack.Command.PICKUP,
    # nethack.Command.ZAP,
    # nethack.Command.FIRE
    )
reward_manager = RewardManager()
# # # reward_manager.add_custom_reward_fn(staircase_reward)
# # # reward_manager.add_custom_reward_fn(apple_reward)
# # reward_manager.add_custom_reward_fn(pickup_key)
reward_manager.add_eat_event("apple", reward = 1.0)
reward_manager.add_message_event(["It's solid stone."], reward=-0.2, terminal_required=False)


env = gym.make(hyper_params["env"],
                observation_keys = ['pixel', 'message', 'glyphs'],
                # penalty_time=-0.1,
                # penalty_step=-0.1,
                # reward_lose=-1,
                # reward_win=5,
                # seeds = hyper_params["seed"],
                # actions = ACTIONS
                reward_manager=reward_manager
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

state = normalize_glyphs(env.reset()) #must be pixels only 

#Mike code
lava_count = 0
key_count = 0
last_action = None
prev_action = None

old_best=-10000
new_best=-10000

best_model = None

for t in range(hyper_params["num-steps"]):
    fraction = min(1.0, float(t) / eps_timesteps)
    eps_threshold = hyper_params["eps-start"] + fraction * (
        hyper_params["eps-end"] - hyper_params["eps-start"]
    )
    sample = random.random()
    if sample <= eps_threshold:
        action = env.action_space.sample()
    else:
        action = agent.act(state)

    next_state, reward, done, _ = env.step(action)
    next_state = normalize_glyphs(next_state)

    # if env.key_in_inventory("wand") == 'f':
    #     key_count += 1
    # if key_count == 1:
    #     reward = 0.5
    # elif(env.key_in_inventory("wand") == 'f' and prev_action == 5 and last_action == 6 and action == 1):
    #     lava_count += 1
    # if lava_count == 1:
    #     reward = 1
        
    # prev_action = last_action
    # last_action = action

    agent.replay_buffer.add(state, action, reward, next_state, float(done))
    state = next_state

    episode_rewards[-1] += reward
    if done:
        state = normalize_glyphs(env.reset())
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
make_video(env, best_model, 30, 30, hyper_params["env"], 1000, f"video_{t}.mp4")