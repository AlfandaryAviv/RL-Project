import gym
import argparse

from sac_agent import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, help='train agent')
    parser.add_argument('--file', default=None, help='weight file for test mode only')
    parser.add_argument('--verbose', choices=[0, 1, 2, 3], default=2, help=' Verbose used in tarin '
                                                                           ' 0 (no plots, no logs, no video), '
                                                                           ' 1 (yes plots, no logs, no video),'
                                                                           ' 2 (yes plots, yes logs, no video), '
                                                                           ' 3 (yes plots, yes logs, yes video)')
    parser.add_argument('--render', default=False, help='render video of the environment')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--memory-size', type=int, default=1000000, help='maximum size of the replay memory buffer')
    parser.add_argument('--gamma', type=float, default=0.9852999830675684, help='discount factor')
    parser.add_argument('--lr-a', type=float, default=0.0019672261874892514, help='learning rate of entropy optimizer')
    parser.add_argument('--lr_critic', type=float, default=0.0035869489052531445, help='learning rate of critic net')
    parser.add_argument('--lr_actor', type=float, default=0.0026895448241888932, help='learning rate of actor net')
    parser.add_argument('--tau', type=float, default=0.08816540110243674, help='temperature')
    parser.add_argument('--alpha', type=float, default=0.22583093470307455, help='entropy coefficient')
    parser.add_argument('--max_time_steps', type=int, default=1000, help='number of timesteps in an episode (train)')
    parser.add_argument('--max_epochs', type=int, default=10000000, help='number of episodes')
    parser.add_argument('--delay_freq', type=int, default=5, help='frequency of delay updates')
    parser.add_argument('--cuda_device_id', type=int, default=0, help='cuda device card id to use if cuda is available')
    parser.add_argument('--uncertainty_noise', default=False, help='if true adds uncertainty gaussian noise to '
                                                                  'the first two elements the observation space')
    args = parser.parse_args()

    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    seed = 18
    set_seed(seed, env)

    device = torch.device(f"cuda:{args.cuda_device_id}" if torch.cuda.is_available() else "cpu")

    params = {
        'env': env,
        'device': device,
        'max_epochs': args.max_epochs,
        'max_time_steps': args.max_time_steps,
        'gamma': args.gamma,
        'lr_actor': args.lr_actor,
        'lr_critic': args.lr_critic,
        'lr_a': args.lr_a,
        'tau': args.tau,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'buffer_size': args.memory_size,
        "update_step": 0,
        "delay_freq": args.delay_freq,
        'verbose': args.verbose,
        'render': args.render,
        'weights_file': 'solved/SAC_20210306_1645uncertainty_best_actor' if args.file is None
        and args.uncertainty_noise is True else 'solved/SAC_solved_20210305_1321_best_actor' if args.file is None
        and args.uncertainty_noise is False else args.file}
        # 'weights_file': 'SAC_solved_20210306_0000_best'}
        # 'weights_file': 'solved/SAC_solved_20210305_1321_best'}

    print("Chosen parameters:")
    for key in params:
        print(f"{key}: {params[key]}")
    print("\n")

    solved_dir, plots_dir, vids_dir = './solved/', './plots/', './video/'
    paths = [solved_dir, plots_dir, vids_dir]
    create_paths(paths)

    agent = SACAgent(params)

    if args.train:
        paths = {'solved_dir': solved_dir, 'plot_dir': plots_dir}
        train(env, paths, agent, params['max_epochs'], params["max_time_steps"], params["batch_size"],
              verbose=params["verbose"], noise=args.uncertainty_noise)
    else:
        paths = {'weights': params['weights_file'], 'plot_dir': plots_dir}
        test(agent, env, paths, render=False, num_episodes=100, max_t=1000, noise=args.uncertainty_noise)


if __name__ == '__main__':
    main()