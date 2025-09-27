import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import Counter

def episode_lengths() -> None:
    """Prints episode length statistics. Plots episode lengths."""
    with open('experiences_data_od2d.pkl', 'rb') as f:
        experiences_data = pickle.load(f)
    wins, draws = 0, 0
    all_episode_lengths = []

    for episode_data in experiences_data:
        episode_experiences, outcome = episode_data
        all_episode_lengths.append(len(episode_experiences)-1)
        if outcome == 'draw':
            draws += 1
        else:
            wins += 1
    print('Speed: ', (sum(all_episode_lengths)-1000)/98.74,' steps/second')
    print('Average episode length: ', np.mean(all_episode_lengths))
    print('STDV episode length: ', np.std(all_episode_lengths))
    print('Wins: ', wins, ' Draws: ', draws)
    print('Number of episodes less than 10 time steps: ', len([x for x in all_episode_lengths if x <= 10]))
    plt.title('Distribution of OD2D Episode Lengths')
    plt.ylabel('Count')
    plt.xlabel('Episode Length')
    plt.hist(all_episode_lengths)
    plt.show()


def action_counts() -> None:
    """Prints action frequencies. Plots action frequencies."""
    with open('experiences_data_od2d.pkl', 'rb') as f:
        experiences_data = pickle.load(f)
    all_actions = []
    action_counts = np.zeros(28)
    for episode_data in experiences_data:
        episode_experiences, outcome = episode_data
        for obs, rews, dones, info, acts in episode_experiences:
            if acts is not None:
                for team in ['Alpha', 'Bravo']:
                    for act_id in acts[team]:
                        all_actions.append(act_id)
                        action_counts[act_id] += 1
    print('Action counts: ', action_counts)
    print('Action frequencies: ', action_counts / np.sum(action_counts))
    plt.title('Distribution of OD2D Action Frequencies')
    plt.ylabel('Count')
    plt.xlabel('Action Number')
    plt.hist(all_actions, bins=28)
    plt.show()


def mobile_fuel() -> None:
    """Prints statistics on mobile fuel usage per time step. """
    with open('mobile_fuel_data_od2d.pkl', 'rb') as f:
        mobile_fuel_data = pickle.load(f)
    all_usages = []
    for episode_data in mobile_fuel_data:
        for step in range(len(episode_data)-1):
            for unit_id in range(1, 11):
                if episode_data[step][unit_id] != 0:
                    all_usages.append(episode_data[step][unit_id] - episode_data[step+1][unit_id])
    print(all_usages)
    print('Mean fuel usage: ', np.mean(all_usages))
    print('STDV fuel usage: ', np.std(all_usages))


def primary_objective_frequencies() -> None:
    """Prints lower bound statistics on completing the primary objective."""
    with open('seeker_fuel_data_od2d.pkl', 'rb') as f:
        seeker_fuel_data = pickle.load(f)
    alpha_wins, beta_wins, other = 0, 0, 0
    for episode_data in seeker_fuel_data:
        alpha_fuels, beta_fuels = episode_data
        if alpha_fuels[-1] == 0 and beta_fuels[-1] == 0:
            other += 1
        elif alpha_fuels[-2] - alpha_fuels[-1] > 11:
            beta_wins += 1
        elif beta_fuels[-2] - beta_fuels[-1] > 11:
            alpha_wins += 1
        else:
            other += 1
    print('Alpha primary win: ', alpha_wins, ', Beta primary win: ', beta_wins, ', Other: ', other)


def termination_conditions() -> None:
    """Prints count of termination conditions."""
    with open('termination_data_od2d.pkl', 'rb') as f:
        termination_data = pickle.load(f)
    print(Counter(termination_data))


if __name__ == '__main__':
    episode_lengths()
    # action_counts()
    # mobile_fuel()
    # primary_objective_frequencies()
    # termination_conditions()
