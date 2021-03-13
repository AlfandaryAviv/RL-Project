import gym
import logging

from sac_agent import *

logger = logging.getLogger("NNI_logger")
NONE = None


def run_trial(params):
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    seed = 18
    set_seed(seed, env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solved_dir, plots_dir, vids_dir = './solved/', './plots/', './video/'
    paths = [solved_dir, plots_dir, vids_dir]
    create_paths(paths)

    paths = {'solved_dir': solved_dir, 'plot_dir': plots_dir}

    params["buffer_size"] = int(params["buffer_size"])
    params["update_step"] = int(params["update_step"])
    params["delay_step"] = int(params["delay_step"])
    params.update({"env": env})
    params.update({"device": device})

    agent = SACAgent(params)

    train(env, paths, agent, 500, int(params["max_time_steps"]), int(params["batch_size"]), _nni=True)


def run_nni():
    try:
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params)
    except Exception as exception:
        logger.error(exception)
        raise


run_nni()