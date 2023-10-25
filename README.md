# Project: MiniHack The Planet 
## Project done by Damion, Michaela, Gavin

All the code is located in the `Code` folder. Please set that as your working directory before proceeding. 

### How to run the code:
There are 2 agents implemented, and several helper files that setup the environments for any testing or evaluation. 

#### Training the models:
Both the transformer based DQN and A2C agent have files titled `train_dqn.py` and `train_a2c.py` respectively. Hyper parameters for each agent can be tuned within those same files under the `hyper-params` dictionary. Please see the requirements below before training or evaluating any models.

#### Evaluating the models:
There is a single file titled `evaluate_models.py` that will evaluate both the DQN and A2C model. It will evaluate the models on the given environment and configuration so long as the models have been trained and saved before hand (see Training the models section above). This function will save NPY files of the average returns and steps of each model as well as the variance of the models. These results can then be visualised using the `plot_results.py` code. 

### Customizing the environments and rewards:
An environment manager was created where all the environments can be found as well as helper functions for custom rewards. Each environment is characterised by "plain" or "config". Plain means it will return the base environment with no limited actions and no custom rewards. "config" refers to an an environment that has custom rewards and/or limtied actions. These can be appended to the function within environment manager file. 

#### Environments Included:  
 - MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0 
 - MiniHack-MazeWalk-9x9-v0
 - MiniHack-Quest-Hard-v0
 - MiniHack-Room-5x5-v0
 - MiniHack-Skill-Custom-v0 (Eating Apple)


### Packages & Versions Required:  
 - pygame 2.5.2
 - minihack 0.1.5
 - nle 0.9.0
 - pandas 2.0.3
 - numpy 1.24.3
 - seaborn 0.13.0
 - torch 2.1.0

## File Structure  
We note that the `Videos` folder contains the best runs of each agent as instructed in the PDF. However, the `Saved_Runs` folder contains all raw output videos, saved models, saved returns and plots from the runs.

```
minihack_submission
│   report.pdf
│   README.md
│
└───Code
│   └───a2c
│   └───dqn
│   └───Saved_Runs
│   │
│   │   environment_manager.py
│   │   evaluate_models.py
│   │   plot_results.py
│   │   train_a2c.py
│   │   train_dqn.py
│   
└───Videos
│   └───DQN
│   └───A2C
```