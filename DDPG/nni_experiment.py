import __init__
from datetime import datetime
import sys
from utils import plot
import gym
import numpy as np
from collections import deque
import nni
import os
import logging

from agent import DDPGAgent
from utils import set_seed

NNI = True

logger = logging.getLogger("NNI_logger")
NONE = None


def run_trial(params):

    solved_dir = './solved/'
    plots_dir = './plots/'

    if not os.path.exists(solved_dir):
        os.mkdir(solved_dir)

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    env = gym.make('LunarLanderContinuous-v2')
    seed = 0
    set_seed(seed, env)

    params.update({'model': 'ddpg'})
    params.update({'cuda_device': 0})
    params.update({'episodes': 1000})
    params.update({'hidden_size_actor': [int(params["linear_actor"]), int(params["linear_actor"])]})
    params.update({'hidden_size_critic': [int(params["linear_critic"]), int(params["linear_critic"])]})
    params.update({'state_dim': env.observation_space.shape[0]})
    params.update({'action_dim': env.action_space.shape[0]})
    params["max_steps"] = int(params["max_steps"])
    params["batch_size"] = int(params["batch_size"])
    params["buffer_size"] = int(params["buffer_size"])
    params["update_factor"] = int(params["update_factor"])
    params["learning_period"] = int(params["learning_period"])

    paths = {'solved_dir': solved_dir, 'plot_dir': plots_dir}

    agent = DDPGAgent(params)
    epsilon_list = [agent.epsilon]

    scores_deque = deque(maxlen=100)
    scores = []

    curr_time = datetime.now().strftime("%Y%m%d_%H%M")

    for episode in range(params['episodes']):
        state = env.reset()
        agent.noise.reset()
        episode_reward = 0

        for step in range(params['max_steps']):
            # action = agent.add_noise(agent.get_action(state))
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done, step)
            step += 1
            state = next_state
            episode_reward += reward
            epsilon_list.append(agent.epsilon)

            if done:
                break

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if NNI and episode % 20 == 0:
            nni.report_intermediate_result(episode)

        if len(scores_deque) == 100 and np.mean(scores_deque) >= 200:
            print('Environment solved !  ')
            print(f"\nEnvironment solved in {episode} episodes!")

            # save the model
            if not NNI:
                agent.save(paths['solved_dir'] + agent.model_name + '_' + curr_time + '_best')
            else:
                agent.save(paths['solved_dir'] + agent.model_name + '_' + nni.get_trial_id() + '_best')

            if not NNI:
                # final plot
                plot(scores,
                     filename=paths['plot_dir'] + agent.model_name + '_train_' + curr_time + '.png')

            break

        if not NNI:
            sys.stdout.write(
                f"episode: {episode}, reward: {np.round(episode_reward, decimals=2)}, average _reward: "
                f"{np.mean(scores_deque)} \n")

    if NNI:
        nni.report_final_result(episode)


def run_nni():
    try:
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params)
    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == '__main__':
    run_nni()
