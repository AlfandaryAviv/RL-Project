import gym
import argparse
from agent import *
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, help='train agent')
    parser.add_argument('--model', type=str, default='ddpg', help='Model implemented')
    parser.add_argument('--file', type=str, default=None, help='the weights file for test')
    parser.add_argument('--verbose', choices=[0, 1, 2, 3], type=int, default=2, help='Verbose mode'
                                                                           ' 0 (no plots, no logs, no video), '
                                                                           ' 1 (yes plots, no logs, no video),'
                                                                           ' 2 (yes plots, yes logs, no video), '
                                                                           ' 3 (yes plots, yes logs, yes video)')
    parser.add_argument('--render', help='show render of the environment, used only in test')
    parser.add_argument('--lr_actor', type=float, default=0.0005848867245515764, help='learning rate actor network')
    parser.add_argument('--lr_critic', type=float, default=0.0014302222580882661, help='learning rate critic network')
    parser.add_argument('--weight_decay', default=0, help='models weight decay factor')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes used in training')
    parser.add_argument('--epsilon', default=1.0232960877577921)
    parser.add_argument('--epsilon_decay', type=float, default=0.000002414332405642902)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.9829207330295889, help='discount factor')
    parser.add_argument('--learning_period', type=int, default=30, help='learning period of the model')
    parser.add_argument('--max_steps', type=int, default=900, help='maximum steps in episode')
    parser.add_argument('--update_factor', type=int, default=30, help='steps for weights update')
    parser.add_argument('--buffer_size', type=int, default=10000000, help='memory buffer size')
    parser.add_argument('--tau', default=0.0034742948471750195)
    parser.add_argument('--cuda_device_id', type=int, default=0, help='cuda device card id to use if cuda is available')
    parser.add_argument('--hidden_size_actor', default=[64, 64], help='the actor hidden sizes')
    parser.add_argument('--hidden_size_critic', default=[256, 256], help='the critic hidden sizes')
    parser.add_argument('--uncertainty_noise', default=False, help='if true adds uncertainty gaussian noise to '
                                                                  'the first two elements the observation space')
    args = parser.parse_args()

    env = gym.make('LunarLanderContinuous-v2')
    seed = 0
    set_seed(seed, env)

    params = {'model': args.model,
              'max_steps': args.max_steps,
              'learning_period': args.learning_period,
              'update_factor': args.update_factor,
              'cuda_device': args.cuda_device_id,
              'gamma': args.gamma,
              'lr_actor': args.lr_actor,
              'lr_critic': args.lr_critic,
              'weight_decay': args.weight_decay,
              'episodes': args.episodes,
              'batch_size': args.batch_size,
              'verbose': args.verbose,
              'epsilon': args.epsilon,
              'epsilon_decay': args.epsilon_decay,
              'weights_file': 'solved/DDPG_20210306_1645uncertainty_best_actor' if args.file is None and
                args.uncertainty_noise is True else 'solved/DDPG_20210306_0023_best'
                if args.file is None and args.uncertainty_noise is False else args.file,
              'render': args.render,
              'buffer_size': args.buffer_size,
              'hidden_size_actor': args.hidden_size_actor,
              'hidden_size_critic': args.hidden_size_critic,
              'tau': args.tau,
              'state_dim': env.observation_space.shape[0],
              'action_dim': env.action_space.shape[0],}

    print("Chosen parameters:")
    for key in params:
        print(f"{key}: {params[key]}")
    print("\n")

    solved_dir, plots_dir, vids_dir = './solved/', './plots/', './video/'
    paths = [solved_dir, plots_dir, vids_dir]
    create_paths(paths)

    agent = DDPGAgent(params)
    epsilon_list = [agent.epsilon]

    if args.train:
        paths = {'solved_dir': solved_dir, 'plot_dir': plots_dir}
        train(env, agent, params, epsilon_list, paths, NNI=False, noise=args.uncertainty_noise)
    else:
        paths = {'weights': params['weights_file'], 'plot_dir': plots_dir}
        test(agent, env, paths, render=False, num_episodes=100, max_t=1000, noise=args.uncertainty_noise)


if __name__ == '__main__':
    main()
