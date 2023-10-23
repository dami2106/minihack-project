import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

hyper_params = {
        'env' : "MiniHack-Quest-Hard-v0", #Name of folder in  runs folder
        'type' : "config",   #config or plain 
}

#Load the results in from the saved NPY files
dqn_results = np.load(f"Saved_Runs/{hyper_params['env']}/{hyper_params['type']}_DQN.npy")
a2c_results = np.load(f"Saved_Runs/{hyper_params['env']}/{hyper_params['type']}_A2C.npy")

# Get the runs etc from the results
dqn_mean_runs = dqn_results[0]
dqn_var_runs = dqn_results[1]
dqn_mean_steps = dqn_results[2]
dqn_var_steps = dqn_results[3]

a2c_mean_runs = a2c_results[0]
a2c_var_runs = a2c_results[1]
a2c_mean_steps = a2c_results[2]
a2c_var_steps = a2c_results[3]


sns.set_theme()
sns.set_style("darkgrid")

#Plot the returns for each model
plt.plot(dqn_mean_runs, label="DQN", color="blue")
plt.plot(a2c_mean_runs, label="A2C", color="red")
plt.fill_between(range(len(dqn_mean_runs)), dqn_mean_runs-dqn_var_runs, dqn_mean_runs+dqn_var_runs, alpha=0.3, color="blue")
plt.fill_between(range(len(a2c_mean_runs)), a2c_mean_runs-a2c_var_runs, a2c_mean_runs+a2c_var_runs, alpha=0.3, color="red")

plt.xlabel("Episode")
plt.ylabel("Return")
plt.ylim(-22, 15)

if hyper_params["type"] == 'plain':
    plt.title(f"Mean return of DQN and A2C on {hyper_params['env']}\nwith no custom rewards or actions ")
else:
    plt.title(f"Mean return of DQN and A2C on {hyper_params['env']}\nwith custom rewards and actions")
plt.legend()

legend_title = None
num_cols = 4
has_title = legend_title is not None
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15 + has_title * 0.05), fancybox=True, shadow=True, ncol=num_cols, title=legend_title)

#Save the results
plt.savefig(f"Saved_Runs/{hyper_params['env']}/return_{hyper_params['type']}.png")
