import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..CausalModels import *

model = CS_3
M = 100
nphotons = 10000
sim_type = 'Exact'

# out_file = f"Data/{model.name}_outputs_{nsamples}.npz"

out_file = f"src/stochprocsim/Data/{sim_type}_{model.name}_M_{M}_photons_{nphotons}.npz"
data = np.load(out_file, allow_pickle=True)
outputs = [pd.DataFrame.from_records(x) for x in data['outputs']]
print(outputs[0])

print(f"Loaded outputs from {out_file}")
print(f"Number of unitary sets: {len(outputs)}")

def plot_distribution_at_each_output():
    # Plot distributions 
    fig, axs = plt.subplots(3,2, sharey=True, tight_layout=True)

    # 000 
    p0_0 = [row[0][0] for row in outputs]
    p1_0 = [row[0][1] for row in outputs]
    axs[0][0].hist(p0_0, bins=30, color='gold')
    axs[0][0].set_title("Path 0")
    axs[0][1].hist(p1_0, bins=30)
    axs[0][1].set_title("Path 1")

    # 001
    p0_1 = [row[1][0] for row in outputs]
    p1_1 = [row[1][1] for row in outputs]
    axs[1][0].hist(p0_1, bins=30, color='gold')
    axs[1][0].set_title("Path 0")
    axs[1][1].hist(p1_1, bins=30)
    axs[1][1].set_title("Path 1")

    # 010
    p0_2 = [row[2][0] for row in outputs]
    p1_2 = [row[2][1] for row in outputs]
    axs[2][0].hist(p0_2, bins=30, color='gold')
    axs[2][0].set_title("Path 0")
    axs[2][1].hist(p1_2, bins=30)
    axs[2][1].set_title("Path 1")

    # plt.show()

# Plot strings 
def plot__output_string_distributions():
    nbins = 100
    fig, axs = plt.subplots(4,1, sharey=True, tight_layout=True) 

    p1 = [df.iloc[0]['N path |1>'] for df in outputs]
    p01 = [df.iloc[1]['N path |1>'] for df in outputs]
    p001 = [df.iloc[2]['N path |1>'] for df in outputs]
    p000 = [df.iloc[2]['N path |0>'] for df in outputs]

    axs[0].hist(p1, bins=nbins, color='gold')
    axs[0].set_title("[1]")
    axs[1].hist(p01, bins=nbins, color='gold')
    axs[1].set_title("[0,1]")
    axs[2].hist(p001, bins=nbins, color='gold')
    axs[2].set_title("[0,0,1]")
    axs[3].hist(p000, bins=nbins, color='gold')
    axs[3].set_title("[0,0,0]")

    for ax in axs:
        ax.set_ylabel("Frequency")
    axs[-1].set_xlabel("Photon counts")
    fig.suptitle("Photon Counts per string output")

def plot_string_counts():
    """
    Get avg count of each string disttribution and std as err
    Plot scatter   
    """
    # Get string distributions 
    s1 = [df.iloc[0]['N path |1>'] for df in outputs]
    s01 = [df.iloc[1]['N path |1>'] for df in outputs]
    s001 = [df.iloc[2]['N path |1>'] for df in outputs]
    s000 = [df.iloc[2]['N path |0>'] for df in outputs]

    data = [s1,s01,s001,s000]
    # print(f'data:{data}')
    normalised_vals = [
    np.array(dist)/np.max(dist) if np.max(dist) > 0 else np.zeros_like(dist)
    for dist in data
    ]
    # err = [(np.std(dist))/(np.sqrt(N)) for dist in normalised_vals]
    err = [(np.std(dist)) for dist in normalised_vals]
    means = np.array([np.mean(dist) if np.max(dist) > 0 else 0 for dist in normalised_vals])
    x = means / sum(means) if sum(means) > 0 else means
    labels = ['[1]', '[0,1]', '[0,0,1]', '[0,0,0]']

    fig, ax = plt.subplots()
    print(f'x:{x}')
    print(f'err_norm:{err}')
    ax.set_ylabel('Avg Counts')
    ax.set_xlabel('Output String')
    ax.set_ylim(0)
    # ax.set_ylim(0, 1.1 * (x + err).max())
    ax.set_title(f'{model.name} M:{M} NPhotons:{nphotons}')
    ax.bar(labels, x, yerr=err, color='gold', edgecolor='black', width=0.5, capsize=5)
    plt.plot(labels, x, color='black', marker='o')
    # plt.show()

def get_plot_data(this_outputs):
    # Get string distributions 
    s1 = [df.iloc[0]['N path |1>'] for df in this_outputs]
    s01 = [df.iloc[1]['N path |1>'] for df in this_outputs]
    s001 = [df.iloc[2]['N path |1>'] for df in this_outputs]
    s000 = [df.iloc[2]['N path |0>'] for df in this_outputs]

    data = [s1,s01,s001,s000]

    means = np.array([np.mean(dist) for dist in data])
    total = np.sum(means)
    x = means / total   
    # err = [np.std(dist) / np.sqrt(M) / total for dist in data]
    err = [np.std(dist) / total for dist in data]

    return x, err

def plot_compare_model_strings():
    """
    Get avg count of each string distribution and std as err
    Plot scatter   
    """
    # Load Quantum
    q_out_file = f"src/stochprocsim/Data/Quantum_{model.name}_M_{M}_photons_{nphotons}.npz"
    q_data = np.load(q_out_file, allow_pickle=True)
    q_outputs = [pd.DataFrame.from_records(x) for x in q_data['outputs']]

    # Load Theory
    t_out_file = f"src/stochprocsim/Data/Exact_{model.name}_M_{M}_photons_{nphotons}.npz"
    t_data = np.load(t_out_file, allow_pickle=True)
    t_outputs = [pd.DataFrame.from_records(x) for x in t_data['outputs']]

    q_x, q_err = get_plot_data(q_outputs)
    t_x, t_err = get_plot_data(t_outputs)
    
    labels = ['[1]', '[0,1]', '[0,0,1]', '[0,0,0]']
    fig, ax = plt.subplots()
    ax.set_ylabel('Avg Counts')
    ax.set_xlabel('Output String')
    ax.set_ylim(0, 1.1 * max((q_x+q_err).max(), (t_x+t_err).max()))
    ax.set_title(f'Detection Counts for N={len(model)} Unitary, NPhotons:{nphotons}, Sampled {M} times')
    print(f'qx:{q_x}')
    print(f'tx:{t_x}')

    ax.bar(labels, q_x, yerr=q_err, color='gold', width=0.5, 
        capsize=5, label='Quantum Model')
    ax.bar(labels, t_x, facecolor='none', edgecolor='black',
        linestyle='--', linewidth=1.5, width=0.5, capsize=5, 
        label='Exact Model')

    ax.legend()
    plt.show()
# if __name__ == '__main__':
#     # # plot_distribution_at_each_output()
#     # plot_string_counts()
#     plot_compare_model_strings()
#     # # plot_string_counts_gpt()
#     # plot_distribution_at_each_output()
#     # plot__output_string_distributions()
#     plt.show() 