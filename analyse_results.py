import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_prob_of_best_arm(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'], axis=1)
    mean_rew = df[['chosen_arms', 'rewards']].groupby('chosen_arms').mean()
    idx_max = mean_rew.idxmax().values[0]
    horizon = df.horizon.max()
    prob_best_arm = []

    # prob of peaking best arm
    for t in np.arange(1, horizon+1):
        mask_ = df['horizon'] == t
        prob_best_arm.append((df.loc[mask_, 'chosen_arms'] == idx_max).mean())

    plt.plot(prob_best_arm)
    plt.xlabel('Horizon [time-steps]')
    plt.ylabel('Avg. Prob. best arm at each step')
    plt.title('Accuracy')
    plt.show()


def plot_avg_reward(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'], axis=1)
    horizon = df.horizon.max()
    avg_reward = []
    # prob of peaking best arm
    for t in np.arange(1, horizon + 1):
        mask_ = df['horizon'] == t
        avg_reward.append(df.loc[mask_, 'rewards'].mean())

    plt.plot(avg_reward)
    plt.xlabel('Horizon [time-steps]')
    plt.ylabel('Avg. reward at each step')
    plt.title('Performance')
    plt.show()


def plot_cumulative_reward(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'], axis=1)
    mean_cum_aw = df[['horizon', 'cumulative_rewards']].groupby('horizon').mean()

    plt.plot(mean_cum_aw)
    plt.xlabel('Horizon [time-steps]')
    plt.ylabel('Average Cumulative Reward of all Simulations')
    plt.show()


if __name__ == "__main__":
    # plot_prob_of_best_arm('results/results_EG.csv')
    # plot_avg_reward('results/results_EG.csv')
    # plot_cumulative_reward('results/results_EG.csv')
    # plot_prob_of_best_arm('results/results_softmax_temp0.5.csv')
    # plot_avg_reward('results/results_softmax_temp0.5.csv')
    # plot_cumulative_reward('results/results_softmax_temp0.5.csv')
    # plot_prob_of_best_arm('results/results_annealing_softmax_temp0.5.csv')
    # plot_avg_reward('results/results_annealing_softmax_temp0.5.csv')
    # plot_cumulative_reward('results/results_annealing_softmax_temp0.5.csv')
    plot_prob_of_best_arm('results/results_ucb1.csv')
    plot_avg_reward('results/results_ucb1.csv')
    plot_cumulative_reward('results/results_ucb1.csv')



