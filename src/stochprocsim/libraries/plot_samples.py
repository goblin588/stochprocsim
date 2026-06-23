import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_outputs(filepath: str) -> list[pd.DataFrame]:
    data = np.load(filepath, allow_pickle=True)
    return [pd.DataFrame.from_records(x) for x in data['outputs']]


def plot_distribution_at_each_output(outputs: list[pd.DataFrame]):
    fig, axs = plt.subplots(3, 2, sharey=True, tight_layout=True)
    for row, ax_pair in enumerate(axs):
        p0 = [df.iloc[row]['N path |0>'] for df in outputs]
        p1 = [df.iloc[row]['N path |1>'] for df in outputs]
        ax_pair[0].hist(p0, bins=30, color='gold')
        ax_pair[0].set_title("Path 0")
        ax_pair[1].hist(p1, bins=30)
        ax_pair[1].set_title("Path 1")


def plot_output_string_distributions(outputs: list[pd.DataFrame]):
    nbins = 100
    fig, axs = plt.subplots(4, 1, sharey=True, tight_layout=True)
    series = [
        ([df.iloc[0]['N path |1>'] for df in outputs], "[1]"),
        ([df.iloc[1]['N path |1>'] for df in outputs], "[0,1]"),
        ([df.iloc[2]['N path |1>'] for df in outputs], "[0,0,1]"),
        ([df.iloc[2]['N path |0>'] for df in outputs], "[0,0,0]"),
    ]
    for ax, (data, label) in zip(axs, series):
        ax.hist(data, bins=nbins, color='gold')
        ax.set_title(label)
        ax.set_ylabel("Frequency")
    axs[-1].set_xlabel("Photon counts")
    fig.suptitle("Photon Counts per string output")


def get_plot_data(outputs: list[pd.DataFrame]):
    """Return (normalised means, normalised stds) for the four output strings."""
    series = [
        [df.iloc[0]['N path |1>'] for df in outputs],
        [df.iloc[1]['N path |1>'] for df in outputs],
        [df.iloc[2]['N path |1>'] for df in outputs],
        [df.iloc[2]['N path |0>'] for df in outputs],
    ]
    means = np.array([np.mean(d) for d in series])
    total = np.sum(means)
    x = means / total
    err = [np.std(d) / total for d in series]
    return x, err


def plot_string_counts(outputs: list[pd.DataFrame], model_name: str, M: int, nphotons: int):
    x, err = get_plot_data(outputs)
    labels = ['[1]', '[0,1]', '[0,0,1]', '[0,0,0]']
    fig, ax = plt.subplots()
    ax.set_ylabel('Avg Counts')
    ax.set_xlabel('Output String')
    ax.set_ylim(0)
    ax.set_title(f'{model_name} M:{M} NPhotons:{nphotons}')
    ax.bar(labels, x, yerr=err, color='gold', edgecolor='black', width=0.5, capsize=5)
    plt.plot(labels, x, color='black', marker='o')


def plot_compare_model_strings(q_filepath: str, t_filepath: str, model_name: str, nphotons: int):
    q_outputs = load_outputs(q_filepath)
    t_outputs = load_outputs(t_filepath)
    q_x, q_err = get_plot_data(q_outputs)
    t_x, t_err = get_plot_data(t_outputs)

    labels = ['[1]', '[0,1]', '[0,0,1]', '[0,0,0]']
    fig, ax = plt.subplots()
    ax.set_ylabel('Avg Counts')
    ax.set_xlabel('Output String')
    ax.set_ylim(0, 1.1 * max((q_x + q_err).max(), (t_x + t_err).max()))
    ax.set_title(f'Detection Counts for {model_name}, NPhotons:{nphotons}')
    ax.bar(labels, q_x, yerr=q_err, color='gold', width=0.5, capsize=5, label='Quantum Model')
    ax.bar(labels, t_x, facecolor='none', edgecolor='black',
           linestyle='--', linewidth=1.5, width=0.5, capsize=5, label='Exact Model')
    ax.legend()
    plt.show()
