import torch
from a2c.agent import ACAgent
from a2c.helper import make_video
from environment_manager import setup_environment

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Hyper paremeters for the DQN network. Set the env and the extra info to config or plain
hyper_params = {
        'learning-rate': 0.01,
        'discount-factor': 0.99,  # discount factor
        'max_episode_length' : 100,
        'max_episodes' : 100,
        'seed' : 102,
        'env' : "MiniHack-Room-5x5-v0",
        'extra-info' : "config"  #config or plain
    }

#Create an environment using the environemnt manager
env = setup_environment(hyper_params["env"], hyper_params["extra-info"])
env.seed(hyper_params["seed"])  

#Create the A2C agent
ac = ACAgent(
    observation_space = env.observation_space,
    action_space = env.action_space,
    lr = hyper_params["learning-rate"],
    gamma = hyper_params["discount-factor"],
    max_episode_length = hyper_params["max_episode_length"],
    max_episodes = hyper_params["max_episodes"],
    env = env
)

#Save the model for evaluation
ac.save_network(f"Agents/{hyper_params['env']}/a2c_{hyper_params['env']}_{hyper_params['extra-info']}.pt")

#Save a video of the agent
env.reset()
make_video(env, ac, 30, 30, f"Agents/{hyper_params['env']}/Videos", 1000, f"a2c_video_{hyper_params['env']}_{hyper_params['extra-info']}.mp4")
