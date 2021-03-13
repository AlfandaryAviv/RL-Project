from model import Actor, Critic
from torch.optim import Adam
import torch.nn as nn
import nni
import time
from replay_all import *
from utils import *
from ounoise import * #OUNoise


class DDPGAgent:
    """ DDPG agent """
    def __init__(self, params):

        self.model_name = 'DDPG'

        self.params = params  # params dict
        self.state_dim = self.params['state_dim']
        self.action_dim = self.params['action_dim']

        # for epsilon decay
        self.epsilon = self.params['epsilon']
        self.eps_max = 1
        self.eps_min = 0.0001
        self.epsilon_decay = params['epsilon_decay']
        self.step = 0

        # discount and temperature parameters
        self.gamma = self.params['gamma']
        self.tau = self.params['tau']

        # if using gpu
        self.cuda_id = self.params['cuda_device']
        self.device = torch.device(f'cuda:{self.cuda_id}' if torch.cuda.is_available() else 'cpu')

        # Actor network - actor, target and optimizer
        self.actor = Actor(self.state_dim, self.action_dim, self.params['hidden_size_actor'][0],
                           self.params['hidden_size_actor'][1]).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim,  self.params['hidden_size_actor'][0],
                                  self.params['hidden_size_actor'][1]).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.params["lr_actor"])

        # Critic network - critic, target and optimizer
        self.critic = Critic(self.state_dim, self.action_dim, self.params['hidden_size_critic'][0],
                             self.params['hidden_size_critic'][1]).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.params['hidden_size_critic'][0],
                                    self.params['hidden_size_critic'][1]).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.params["lr_critic"],
                                     weight_decay=self.params["weight_decay"])

        self.critic_criterion = nn.MSELoss()  # loss function

        # set the target networks parameters
        self.copy_params()

        self.memory = ReplayMemory(self.params['buffer_size'], self.device)  # replay buffer
        self.batch_size = params["batch_size"]
        self.noise = OUNoise(self.action_dim)  # OU noise

    def copy_params(self):
        """copy params to target param"""
        nets = [self.actor, self.critic]
        target_nets = [self.actor_target, self.critic_target]
        for i in range(2):
            for target_param, param in zip(target_nets[i].parameters(), nets[i].parameters()):
                target_param.data.copy_(param.data)

    def get_action(self, state):
        """
        Given a state pass it through the actor network with no gradients, move it to train mode, and add
        the OU noise multiplied by epsilon to the action, then return the clipped action.
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        self.actor.eval()  # test mode
        with torch.no_grad():
            # pass the state through the actor network to get the state
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()  # train mode
        # add OU noise multiplied by epsilon and perform clipping
        action += self.epsilon * self.noise.evolve_state()
        action = np.clip(action, -1, 1)

        return action

    def choose_action(self, curr_state, is_test=False):
        """
        Choose an action using noise.
        :param curr_state: current state of the environment
        :param is_test: if True, always choose greedy action (in testing we always choose the greedy action)
        :return: the action chosen
        """
        if not is_test:
            return self.get_action(curr_state)
        else:
            # we're using the network for inference only, we don't want to track the gradients in this case
            with torch.no_grad():
                curr_state = torch.tensor(curr_state).to(self.device)
                return self.actor(curr_state).detach().cpu().numpy()

    def decay_func(self):
        """
        Perform epsilon decay- choose the max value of the minimum epsilon and the current
        one minus the epsilon decay value
        """
        x = max(self.eps_min, self.epsilon - self.params["epsilon_decay"])
        return x

    def update(self, state, action, reward, next_state, done, step):
        """ Perform an a training update """
        # append current experience to the memory
        self.memory.append(state, action, reward, next_state, done)
        self.step = step
        # perform the update every few steps according to the learning period
        if len(self.memory.buffer) > self.batch_size and step % self.params["learning_period"] == 0:
            for _ in range(self.params["update_factor"]):
                # get memory - learn from experience
                states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.model_name)

                # Critic
                next_actions = self.actor_target(next_states)
                # current and next q-vales
                q_vals = self.critic(states, actions)
                next_q = self.critic_target(next_states, next_actions)
                # bellman equation
                q_prime = rewards + (self.gamma * next_q * (1 - dones))
                # MSE loss
                critic_loss = self.critic_criterion(q_vals, q_prime)

                # Critic loss- do backpropagation and critic optimizer updates
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.critic_optimizer.step()

                # Actor - predict next actions
                actions_pred = self.actor(states)
                # compute loss, do backpropagation and actor optimizer updates
                actor_loss = -self.critic(states, actions_pred).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # weights of model will be copied to model target
                nets = [self.critic, self.actor]
                target_nets = [self.critic_target, self.actor_target]
                for i in range(2):
                    for target_param, param in zip(target_nets[i].parameters(), nets[i].parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                # epsilon decay
                self.epsilon = self.decay_func()
                # reset noise
                self.noise.reset()

    def save(self, filename):
        """
        Save the network weights.
        :param filename: path
        """
        torch.save(self.actor.state_dict(), filename + "_actor.pth")

    def load(self, filename):
        """
        Load the network weights.
        :param filename: path of the weight file
        """
        self.actor.load_state_dict(torch.load(filename + ".pth", map_location=self.device))

    def set_test(self):
        self.actor.eval()


def train(env, agent, params, epsilon_list, paths, NNI=False, noise=False):
    scores_deque = deque(maxlen=100)
    scores = []
    vid = video_recorder.VideoRecorder(env, path="./video/vid_train_ddpg.mp4")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    time_start = time.time()

    for episode in range(params['episodes']):
        state = env.reset()
        agent.noise.reset()

        if noise:  # add uncertainty if required
            state = uncertainty(state)

        episode_reward = 0

        for step in range(params['max_steps']):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done, step)
            step += 1
            state = next_state
            episode_reward += reward
            epsilon_list.append(agent.epsilon)

            if done:
                break
            elif params['verbose'] == 3:
                env.render()
                vid.capture_frame()

        scores_deque.append(episode_reward)
        scores.append(episode_reward)
        avg_score = np.mean(scores_deque)

        if NNI and episode % 20 == 0:
            nni.report_intermediate_result(episode)

        if params['verbose']>1 and not NNI:
            s = (int(time.time() - time_start))

            print(f'Episode {episode} Score: {episode_reward:.2f} Average Score: {avg_score:.2f}, '
                  f'Time: { s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}')

        if len(scores_deque) == 100 and avg_score >= 200:
            if params['verbose']>1:
                print(f'Environment solved ! \nEnvironment solved in {episode} episodes!')

            # save the model
            if params['verbose'] >= 1:
                a = nni.get_trial_id() if NNI else curr_time
                if noise:
                    agent.save(paths['solved_dir'] + agent.model_name + '_' + a + 'uncertainty_best')
                else:
                    agent.save(paths['solved_dir'] + agent.model_name + '_' + a + '_best')

                if not NNI:
                    # final plot
                    if noise:
                        plot(scores, agent.model_name,
                             filename=paths['plot_dir'] + agent.model_name + '_train_uncertainty' + curr_time + '.png', noise=noise)
                    else:
                        plot(scores, agent.model_name, filename=paths['plot_dir'] + agent.model_name + '_train_' +
                                                                curr_time + '.png', noise=noise)

            break
    if NNI:
        nni.report_final_result(episode)

    env.close()
