#!/usr/bin/env python

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import numpy as np
import pandas
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt


# =============================================================================
# CONSTANTS
# =============================================================================

# The column names in the Pandas Dataframe associated to the DG prediction and their uncertainties.
DG_KEY = '$\Delta$G [kcal/mol]'
DDG_KEY = 'd$\Delta$G [kcal/mol]'

# Path to the directory containing the Pandas Dataframe.
DATA_DIR_PATH = 'figure7_data'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def to_numpy_free_energy_trajectory(system_data):
    """Convert the free energy trajectories of the replicate calculations from Dataframe to array format.

    Parameters
    ----------
    system_data : pandas.Dataframe
        The data with the replicate free energy trajectories for a system.

    Returns
    -------
    free_energy_trajectories : numpy.ndarray
        free_energy_trajectories[replicate_idx][t] is the free energy of the
        replicate_idx-th replicate calculation evaluated after
        n_energy_evaluations[t] energy/force evaluations.
    n_energy_evaluations : numpy.ndarray
        The number of force/energy evaluations for each time point.

    """
    # Find all the name of the replicates.
    replicate_ids = sorted(system_data['System ID'].unique())

    # Initialize the free energy array.
    n_replicates = len(replicate_ids)
    n_time_points = int(len(system_data) / n_replicates)
    free_energy_trajectories = np.empty(shape=(n_replicates, n_time_points))

    # Extract data for each replicate.
    for replicate_idx, replicate_id in enumerate(replicate_ids):
        replicate_data = system_data[system_data['System ID'] == replicate_id]
        free_energy_trajectories[replicate_idx] = replicate_data[DG_KEY].values

    # The number of energy evaluations is identical for all replicates.
    n_energy_evaluations = np.array(sorted(system_data['N energy evaluations'].unique()))
    assert len(n_energy_evaluations) == n_time_points

    return free_energy_trajectories, n_energy_evaluations


def compute_mean_free_energy_trajectory(free_energy_trajectories):
    """Compute mean free energy trajectory and CI from replicate trajectories.

    Parameters
    ----------
    free_energy_trajectories : numpy.ndarray
        free_energy_trajectories[replicate_idx][t] is the free energy of the
        replicate_idx-th replicate calculation evaluated at the
        t-th number of energy/force evaluations.

    Returns
    -------
    mean_free_energy_trajectory : numpy.ndarray
        free_energy_trajectory[t] is the mean free energy evaluated at the
        t-th number of energy/force evaluations.
    ci_free_energy_trajectory : numpy.ndarray
        ci_free_energy_trajectory[t] is the t-based 95% confidence interval
        so that the final estimate at t is

            mean_free_energy_trajectory[t] +- ci_free_energy_trajectory[t]

    """
    mean_free_energy_trajectory = free_energy_trajectories.mean(axis=0)

    # The t-based confidence interval depends on the number of independent replicates.
    n_replicates = free_energy_trajectories.shape[0]
    t_degrees_of_freedom = n_replicates - 1
    t_statistics = scipy.stats.t.interval(alpha=0.95, df=t_degrees_of_freedom)[1]
    sem = scipy.stats.sem(free_energy_trajectories, axis=0)

    return mean_free_energy_trajectory, t_statistics * sem


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_mean_free_energy(t, mean_free_energy_trajectory, ci_trajectory, ax,
                          color_mean=None, color_ci=None, **plot_kwargs):
    """Plot mean trajectory with confidence intervals."""
    # Plot mean trajectory confidence intervals.
    upper_bound = mean_free_energy_trajectory + ci_trajectory
    lower_bound = mean_free_energy_trajectory - ci_trajectory
    ax.fill_between(t, upper_bound, lower_bound, alpha=0.15, color=color_ci)

    # Plot the mean free energy trajectory.
    ax.plot(t, mean_free_energy_trajectory, color=color_mean, alpha=1.0, **plot_kwargs)


def plot_yank_trajectories(data):
    """Plot the mean free energy trajectory and uncertainty of the YANK calculations."""
    sns.set_style('whitegrid')
    sns.set_context('paper')
    system_colors = {'CB8-G3': 'C0', 'OA-G6': 'C1'}

    # -------------------- #
    # Plot submission data #
    # -------------------- #

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.8, 2.4))

    # for system_name in unique_system_names:
    for system_name, system_data in data.items():
        color = system_colors[system_name]

        # Obtain the replicate free energy trajectoreis as numpy arrays.
        free_energy_trajectories, n_energy_evaluations = to_numpy_free_energy_trajectory(system_data)

        # Compute the mean and confidence interval.
        mean_free_energy_trajectory, ci_trajectory = compute_mean_free_energy_trajectory(free_energy_trajectories)

        # Convert number of energy evaluations into an equivalent number of nanoseconds of MD. Timestep is 2fs.
        t = n_energy_evaluations / 500 / 1000

        plot_mean_free_energy(t, mean_free_energy_trajectory, ci_trajectory, ax,
                              color_mean=color, color_ci=color, label=system_name)

        # Plot asymptotic free energy.
        asymptotic_DG = mean_free_energy_trajectory[-1]
        print(system_name, t[-1], 'ns')
        ax.plot(t, [asymptotic_DG for _ in t], color='black', ls='--', alpha=0.8, zorder=1)

    # Set limits.
    ax.set_xlim((0, 2500))
    ax.set_ylim((-13, -6))
    ax.set_xlabel('total simulation time [ns]')
    ax.set_ylabel(DG_KEY)

    plt.tight_layout()

    # Create legend.
    ax.legend(loc='lower right', ncol=3, borderpad=0.5, borderaxespad=0.2)

    # Save figure.
    plt.savefig('free_energy_trajectories.pdf')
    # plt.show()


def load_data():
    """Load the pandas dataframes containing the data used for plotting.

    This is a subset of the data that was generated for the SAMPLing challenge.
    See the file reference_free_energies.json in:

        https://github.com/samplchallenges/SAMPL6/tree/master/host_guest/Analysis/SAMPLing/Data

    Returns
    -------
    data : Dict[str, pandas.Dataframe]
        data[system_name] is the dataframe containing the replicate free
        energy trajectory for the system.
    """
    system_names = ['CB8-G3', 'OA-G6']

    data = {system_name: None for system_name in system_names}
    for system_name in data:
        json_file_path = os.path.join(DATA_DIR_PATH, system_name+'_pandas_dataframe.json')
        dataframe = pandas.read_json(json_file_path)

        # Sort the records by key as the JSON changes the ordering.
        data[system_name] = dataframe.sort_index()

    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Read the data.
    data = load_data()

    # Plot YANK calculations on a single graphics.
    plot_yank_trajectories(data)
