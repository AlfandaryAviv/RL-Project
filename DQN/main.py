import __init__
import argparse
import gym
from agents import *
from utils import *
from wrap import *


def create_env(env_name, model_name):
    env = gym.make(env_name)
    env = DiscretizeAS(env, 3)
    env = FlattenAS(env)
    # this is now a single integer
    print('Observation space: ', env.observation_space, 'Example state: ', env.observation_space.sample(),
          'Type of state: ', type(env.observation_space.sample()))
    print('Action space: ', env.action_space, 'Example action space: ', env.action_space.sample())
    seed = 18
    set_seed(seed, env)
    vid = save_video(env, model_name)
    return env, vid


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dqn', 'full_dqn', 'double_dqn', 'dueling_ddqn'], default='dueling_ddqn',
                        help='Model to use')
    parser.add_argument('--train', default=True, help='train agent')
    parser.add_argument('--file', default=None, help='weights file used in test')
    parser.add_argument('--verbose', choices=[0, 1, 2, 3], default=2, help=' Verbose used in train '
                                                                           ' 0 (no plots, no logs, no video), '
                                                                           ' 1 (yes plots, no logs, no video),'
                                                                           ' 2 (yes plots, yes logs, no video), '
                                                                           ' 3 (yes plots, yes logs, yes video)')
    parser.add_argument('--render', default=True, help='render video of the environment')
    parser.add_argument('--batch_size', default=None)
    parser.add_argument('--memory_size', default=None, help='replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=None, help='discount factor')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--episodes', type=int, default=800, help='number of episodes in train')
    parser.add_argument('--max_steps', type=int, default=1000, help='number of time steps in an episode (train)')
    parser.add_argument('--target_update', type=int, default=None, help='number of updates')
    parser.add_argument('--learn_freq', type=int, default=None, help='number of steps for agent weights update')
    parser.add_argument('--decay', type=float, default=None, help='epsilon decay')
    parser.add_argument('--cuda_device_id', type=int, default=0, help='cuda device card id to use if cuda is available')
    parser.add_argument('--max_eps', type=float, default=1.0, help='max epsilon')
    parser.add_argument('--min_eps', type=float, default=0.01, help='min epsilon')
    parser.add_argument('--uncertainty_noise', default=False, help='if true adds uncertainty gaussian noise to '
                                                                  'the first two elements the observation space')
    args = parser.parse_args()

    # best params from nni
    best_params = {"dqn": {"learn_freq": 2, "batch_size": 128, "gamma": 0.9883806533175878, "lr": 0.0012472163933024034,
                           "memory_size": 1e6, "decay": 0.9523713209404179,
                           "file": "solved/DQN_solved_20210305_1913_best",
                           "uncertainty_file":"solved/DQN_solved_20210306_1653uncertainty_best"},
                   "full_dqn": {"target_update": 300, "learn_freq": 5, "batch_size": 128, "gamma": 0.9860469753781066,
                                "lr": 0.0031532075863686176, "memory_size": 1000000, "decay": 0.9780758904639213,
                                "file": "solved/FullDQN_solved_20210305_1958_best",
                                "uncertainty_file": "solved/FullDQN_solved_20210306_1644uncertainty_best"},
                   "double_dqn": {"target_update": 400, "learn_freq": 4, "batch_size": 128,
                                  "gamma": 0.9815146119125945, "lr": 0.003922422391295347, "memory_size": 100000,
                                  "decay": 0.9633528617036936, "file": "solved/DoubleDQN_solved_20210305_2122_best"
                                  , "uncertainty_file":"solved/DoubleDQN_solved_20210306_1623uncertainty_best"},
                   "dueling_ddqn": {"target_update": 300, "learn_freq": 3, "batch_size": 64, "gamma": 0.9885947345031,
                                    "decay": 0.9727533798819855, "lr": 0.0021629171483891797, "memory_size": 100000,
                                    "file": "solved/DuelingDDQN_solved_20210305_2349_best",
                                    "uncertainty_file":"solved/DuelingDDQN_solved_20210306_1632uncertainty_best"}}

    params = {'model': args.model,
              'max_steps': int(args.max_steps),
              'target_update': best_params[args.model]['target_update'] if args.target_update is None and args.model !='dqn' else args.target_update,
              'learn_freq': best_params[args.model]['learn_freq'] if args.learn_freq is None else args.learn_freq,
              'gamma': best_params[args.model]['gamma'] if args.gamma is None else args.gamma,
              'lr': best_params[args.model]['lr'] if args.lr is None else args.lr,
              'num_episodes': int(args.episodes),
              'batch_size': int(best_params[args.model]['batch_size']) if args.batch_size is None else args.batch_size,
              'memory_size':  int(best_params[args.model]['memory_size']) if args.memory_size is None else args.memory_size,
              'verbose': int(args.verbose),
              'max_eps': args.max_eps,
              'min_eps': args.min_eps,
              'decay_rate': best_params[args.model]['decay'] if args.decay is None else args.decay,
              # 'weights_file': best_params[args.model]['file'] if args.file is None else args.file,
              'weights_file': best_params[args.model]['file'] if args.file is None and args.uncertainty_noise is False
              else best_params[args.model]['uncertainty_file'] if args.file is None and args.uncertainty_noise is True
              else args.file,
              'render': args.render}

    print("Chosen parameters:")
    for key in params:
        print(f"{key}: {params[key]}")
    print("\n")

    device = torch.device(f"cuda:{args.cuda_device_id}" if torch.cuda.is_available() else "cpu")
    print(device)

    solved_dir, plots_dir, vids_dir = './solved/', './plots/', './video/'
    paths = [solved_dir, plots_dir, vids_dir]
    create_paths(paths)

    env_name = 'LunarLanderContinuous-v2'
    env, vid = create_env(env_name, params["model"])

    agents_input = (env.observation_space.shape[0], env.action_space.n, params['lr'], params['gamma'],
                    params['memory_size'], params['batch_size'], params['max_eps'], params['min_eps'],
                    params['decay_rate'], device, vid, params['target_update'])
    # define agents
    if args.model == 'dqn':
        agents_input = list(agents_input)
        agents_input.remove(params['target_update'])
        agents_input = tuple(agents_input)
        agent = DQNAgent(*agents_input)

    elif args.model == "double_dqn":
        agent = DoubleDQNAgent(*agents_input)

    elif args.model == "full_dqn":
        agent = FullDQNAgent(*agents_input)

    else:      # dueling_dqn
        agent = DuelingDDQNAgent(*agents_input)

    if args.train:
        paths = {'solved_dir': solved_dir,
                 'plot_dir': plots_dir}

        agent.train(env, paths, num_episodes=params['num_episodes'], steps=params['max_steps'],
                    learn_freq=params['learn_freq'], verbose=params['verbose'], _nni=False, noise=args.uncertainty_noise)
    else:
        paths = {'weights': params['weights_file'],
                 'plot_dir': plots_dir}
        # test(agent, env, paths, params['render'], num_episodes=100, max_t=1000)
        test(agent, env, paths, False, num_episodes=100, max_t=1000, noise=args.uncertainty_noise)


if __name__ == "__main__":
    main()
