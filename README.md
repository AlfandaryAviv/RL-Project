# RL-Project

**Contact**

123shovalf@gmail.com , aviv.alfandary@gmail.com

## Overview

This is our final project in the course "Reinforcment Learning" held by Doctor Moshe Butman, Bar-Ilan University.

In this project, we have solved the ["LunarLanderContinuous-v2" open-ai gym environment](https://gym.openai.com/envs/LunarLanderContinuous-v2/), using several approaches: 

**Deep-Q-Learning algorithms assuming discrete action space**
- [DQN](https://arxiv.org/abs/1312.5602)
- [Full-DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [Double-DQN (DDQN)](https://arxiv.org/abs/1509.06461)
- [Dueling-DDQN](https://arxiv.org/abs/1511.06581)

**Algorithms assuming continuous action space**
- [SAC](https://arxiv.org/abs/1812.05905)
- [DDPG](https://arxiv.org/abs/1509.02971)

The goal is to solve the environment as fast as possible.

## Dependecies

## Cloning this repo
To install this project, run
```
git clone https://github.com/AlfandaryAviv/Reinforcement-Learning-Project
cd Reinforcement-Learning-Project
pip install -r requirements.txt
```

If this repository is cloned as pycharm projects, one needs to make the following directories as sources directories: `DQN`, `DDPG`, `SAC`.

## About This Repository

This repository contains our implementation of the above algorithms. Credits are mentioned in our pdf file.

### The Directories
- `DQN`- Contains the implementations of the 4 discrete action space algorithms: DQN, Full-DQN, Double-DQN, Dueling-DDQN.
  - `agents.py` -> All DQNs agents classes
  - `main.py` -> Main file to run
  - `model.py` -> DQNs models (neural networks)
  - `wrap.py` -> Discritizing the continuous environment
- `SAC` - Contains the implementation of the SAC RL algorithm.
  - `main_sac.py` -> Main file to run
  - `sac_agent.py` -> SAC agent class
  - `sac_model.py` -> Soft Actor and Critic models
- `DDPG` - Contains the implementation of the DDPG RL algorithm.
  - `agent.py` -> DDPG agent class
  - `main_ddpg.py` -> Main file to run
  - `model.py` -> Actor and Critic models
  - `ounoise.py` -> OU-Noise process

Each directory also contains 3 folders: `./solved`- saved weights of the best models from training mode (for testing), `./video`- saved videos of the agents, `./plots`- saved figures (train and test).

In addition, there are two common files for all the directories: `utils.py`- contains common utils functions, `replay_all.py`- replay memory implementation that is common for all.

### How to run
Each directory has its main file that contains the main function which creates the agents and launches them. Train or test can be performed.

Running one of the following commands (depends on which algorithm you would like to run) display a help message with all the optional parameters, each with an explanation, options and default.
```
cd DQN
python main.py -h
```
```
cd DDPG
python main_ddpg.py -h
```
```
cd SAC
python main_sac.py -h
```
Note that for all the algorithms, you have to specify whether you are in train or test mode (default is train), and if you are in test mode you should specify the weight file of the best model you want to use (from `solved` folder). Specifically for the DQNs, it is important to mention the name of the model you want to run (because there are 4 options). In addition, for all algorithms, the default parameters that are specified in the main file are the best ones (fine-tuned ones).

#### Examples for train (after cd to the right directory)
```
python main.py --train True --model dqn
python main.py --train True --model full_dqn
python main.py --train True --model double_dqn
python main.py --train True --model dueling_dqn
```
```
python main_sac.py --train True
```
```
python main_ddpg.py --train True 
```
If the agent succeeded in solving the environment, the weights of the model in the last episode (that they are the best) will be saved as `.pth` file in the `./solved` folder, with the name of the model and time stamp. Also, figures will be saved in `./plots` folder.

#### Test
```
python main.py --train False --model dqn --file 'solved/name_of_file'
python main.py --train False --model full_dqn --file 'solved/name_of_file'
python main.py --train False --model double_dqn --file 'solved/name_of_file'
python main.py --train False --model dueling_dqn --file 'solved/name_of_file'
```
```
python main_sac.py --train False --file 'solved/name_of_file'
```
```
python main_ddpg.py --train False --file 'solved/name_of_file'
```
Note that you can also produce video of the first 10 test episodes if you add **--render True** to the commands above.

#### Uncertainty
To run the agent with uncertainty, i.e. with adding Gaussian noise to its position, just add **--uncertainty_noise True** to the commands above.

### NNI
To find the best hyperparameters, we used a tool named [NNI](https://github.com/Microsoft/nni), for hyperparameters fine-tuning. Therefore, in all directories, files to run NNI can be found:
- `nni_experiment.py` - Similar to the main file, only this is the main file for running NNI.
- `config_{name_of_algorithm}.yml` - Configuration file needed to run NNI.
- `search_space.json` - Parameters file to determine the NNI search space.
To run NNI, you should first make sure the package `nni` is installed in your environment (appear in the requirements file) and run the following command from the appropriate directory:
```
nnictl create --config config_{name_of_algorithm}.yml --port 8085
```
port can be for example 8085, 8086 and more.
