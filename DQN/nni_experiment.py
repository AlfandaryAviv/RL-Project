import __init__
from agents import *
from utils import *
from wrap import *
import nni
import gym
import logging


logger = logging.getLogger("NNI_logger")
NONE = None


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


def run_trial(model_name, params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    solved_dir, plots_dir, vids_dir = './solved/', './plots/', './video/'
    paths = [solved_dir, plots_dir, vids_dir]
    create_paths(paths)

    env_name = 'LunarLanderContinuous-v2'
    env, vid = create_env(env_name, model_name)

    agents_input = (env.observation_space.shape[0], env.action_space.n, params['lr'], params['gamma'],
                    int(params['memory_size']), int(params['batch_size']), 1, 0.01,
                    params['decay_rate'], device, vid, int(params['target_update']), 'exp')

    # define agents
    if model_name == 'dqn':
        agents_input = list(agents_input)
        agents_input.remove(params['target_update'])
        agents_input = tuple(agents_input)
        agent = DQNAgent(*agents_input)

    elif model_name == "double_dqn":
        agent = DoubleDQNAgent(*agents_input)

    elif model_name == "full_dqn":
        agent = FullDQNAgent(*agents_input)

    else:  # dueling_dqn
        agent = DuelingDQNAgent(*agents_input)

    paths = {'solved_dir': solved_dir, 'plot_dir': plots_dir}

    agent.train(env, paths, 500, int(params['max_steps']), int(params['learn_freq']), 0, _nni=True, noise=False)


def main_nni(model):
    try:
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(model, params)
    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == "__main__":
    my_model = 'dueling_dqn'
    main_nni(my_model)
