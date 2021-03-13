from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from gym.wrappers.monitoring import video_recorder
import os


def initialize(layer, w):
    """weights and bias initialization"""
    layer.weight.data.uniform_(-w, w)
    layer.bias.data.uniform_(-w, w)


def plot(scores, model_name, eps=None, filename=None, noise=False):
    """
    Plot the current trend of the agent.
    :param scores: list of all episodes rewards
    :param model_name: name of the model
    :param eps: if specified, plots also the epsilon-value
    :param filename: if specified, saves the plot at the path given
    :param noise: if True, it means we did an uncertainty experiment and the plot is saved by this name
    """
    avg = [0]*100
    for i in range(len(scores)):
        if i >= 100:
            avg.append(np.mean(scores[i - 100:i]))

    fig, axis = plt.subplots()
    axis.clear()
    axis.plot(scores, 'blue', label='Score per episode', alpha=0.4)
    axis.plot(avg, 'black', label='Mean score of the last 100 episodes')
    axis.axhline(200, c='green', label='Winning score', alpha=0.7)
    axis.axhline(0, c='grey', ls='--', alpha=0.7)
    axis.set_xlabel('Episodes')
    axis.set_ylabel('Scores')
    axis.legend(loc='lower right')
    if noise:
        plt.title(f"{model_name} train with uncertainty")
    else:
        plt.title(f"{model_name} train")
    if eps is not None:
        tw_axis = axis.twinx()
        tw_axis.plot(eps, 'red', alpha=0.5)
        tw_axis.set_ylabel('Epsilon', color='red')

    if filename is not None:
        plt.savefig(filename)
    plt.close()


def uncertainty(state, mu=0, sigma=0.05):
    """
    Add uncertainty to the observations- gaussian noise
    :param state: Current State
    :param mu: mean of gaussian distribution
    :param sigma: std of gaussian distribution
    """
    noise = np.random.normal(mu, sigma, 2)
    state[0] += noise[0]
    state[1] += noise[1]
    return state


def set_seed(seed, env):
    """ Sets the random seed in the environment, pytorch, numpy and random."""
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_paths(paths):
    """
    Creates dirs for given paths (video, solved, plots)
    :param paths: a list of paths
    """
    for my_path in paths:
        if not os.path.exists(my_path):
            os.mkdir(my_path)


def test(agent, env, paths, render=True, num_episodes=100, max_t=1000, noise=False):
    """
    Test the agent using saved weights from train.
    :param agent: The agent (DQN, Full-DQN, Double-DQN, Dueling-DDQN, SAC, DDPG)
    :param env: openai gym environment
    :param paths: dictionary- {weights: weights file path, plots: plots dir}
    :param render: True shows the agent playing
    :param num_episodes: number of episodes to test the agent
    :param max_t: max time steps per episode
    :param noise: if True, it means we did an uncertainty experiment and the plot is saved by this name
    """
    try:
        agent.load(paths['weights'])
    except FileNotFoundError:
        print("File not found.")
        exit(1)

    agent.set_test()
    vid_test = video_recorder.VideoRecorder(env, path=f"./video/vid_test_{agent.model_name}.mp4")
    test_scores = []
    print("*TEST*")
    for episode in range(1, num_episodes + 1):
        s = env.reset()
        score = 0
        for t in range(1, max_t + 1):
            if render:
                env.render()
                vid_test.capture_frame()
            a = agent.choose_action(s, is_test=True)
            s_, r, d, _ = env.step(a)
            score += r
            s = s_
            t += 1
            if d:
                break

        test_scores.append(score)
        print(f"Episode {episode} - score {score:.02f}")
    test_scores = np.array(test_scores)
    success = test_scores[test_scores >= 200]
    if paths['plot_dir'] is not None:
        plt.axhline(200, c='green', label='Winning score', alpha=0.7)
        plt.plot(test_scores, c='blue', label='Score per episode', alpha=0.4)
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        plt.legend(loc='lower right')
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        if noise:
            plt.title(
                f"{agent.model_name} test with uncertainty \n success rate: {len(success) * 100 / num_episodes}% , "
                f"highest score: {np.max(test_scores):.02f}")
            plt.savefig(paths['plot_dir'] + agent.model_name + '_test_uncertainty' + curr_time + '.png')
        else:
            plt.title(f"{agent.model_name} test with success rate: {len(success)*100 / num_episodes}% and highest score: {np.max(test_scores):.02f}")
            plt.savefig(paths['plot_dir'] + agent.model_name + '_test_' + curr_time + '.png')
    print("Test finished")
    print(f"Success rate: {len(success)*100 / num_episodes}% - highest score: {np.max(test_scores):.02f}")
    env.close()


def save_video(env, model_name):
    """saves the video of testing the agent"""
    vid = video_recorder.VideoRecorder(env, path=f"./video/vid_train_{model_name}.mp4")
    return vid
