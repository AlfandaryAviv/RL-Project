import torch.optim as optim
import time
import nni
import torch.nn.functional as F

from sac_model import *
from replay_all import *
from utils import *


class SACAgent:
    """SAC agent"""
    def __init__(self, params):

        # model name, environment and device
        self.model_name = "SAC"
        self.env = params["env"]
        self.device = params['device']

        # replay buffer - memory
        self.memory = ReplayMemory(params["buffer_size"], self.device)

        # size of observation and action space
        self.action_range = [self.env.action_space.low, self.env.action_space.high]
        obs_space_size = self.env.observation_space.shape[0]
        action_space_size = self.env.action_space.shape[0]

        # initialize networks
        self.critic1 = SoftCritic(obs_space_size, action_space_size).to(self.device)
        self.critic2 = SoftCritic(obs_space_size, action_space_size).to(self.device)
        self.target_critic1 = SoftCritic(obs_space_size, action_space_size).to(self.device)
        self.target_critic2 = SoftCritic(obs_space_size, action_space_size).to(self.device)
        self.actor = Actor(self.action_range, obs_space_size, action_space_size, self.device).to(self.device)

        # copy params to target param
        self.copy_params()

        # hyper-parameters
        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.update_step = params["update_step"]
        self.delay_freq = params["delay_freq"]
        self.batch_size = params["batch_size"]

        # entropy temperature
        self.alpha = params["alpha"]
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=params["lr_a"])

        # initialize optimizers
        self.c1_optimizer = optim.Adam(self.critic1.parameters(), lr=params["lr_critic"])
        self.c2_optimizer = optim.Adam(self.critic2.parameters(), lr=params["lr_critic"])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params["lr_actor"])

    def copy_params(self):
        """copy params to target param"""
        critic_nets = [self.critic1, self.critic2]
        target_nets = [self.target_critic1, self.target_critic2]
        for i in range(2):
            for target_param, param in zip(target_nets[i].parameters(), critic_nets[i].parameters()):
                target_param.data.copy_(param)

    def choose_action(self, curr_state, is_test=False):
        """
        Choose an action using noise.
        :param curr_state: current state of the environment
        :param is_test: if True, always choose greedy action (in testing we always choose the greedy action)
        :return: the action chosen
        """
        if not is_test:
            return self.actor.get_action(curr_state)
        else:
            # we're using the network for inference only, we don't want to track the gradients in this case
            with torch.no_grad():
                state = torch.tensor(curr_state).to(self.device)
                action, _ = self.actor.sample(state, only_action=True)
                return action.detach().cpu().numpy()

    @staticmethod
    def update_optimizers(optimizer, loss):
        """update optimizers in update function"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def delayed_update(self, states, new_actions, log_pi):
        """update only if update_step % delay_step == 0"""
        # compute q values given current states and new actions and find the minimum
        q1_value, q2_value = self.critic1.forward(states, new_actions), self.critic2.forward(states, new_actions)
        min_q_value = torch.min(q1_value, q2_value)
        # compute the actor loss
        actor_loss = (self.alpha * log_pi - min_q_value).mean()

        # update optimizers
        self.update_optimizers(self.actor_optimizer, actor_loss)

        # copy params to target param
        critic_nets = [self.critic1, self.critic2]
        target_nets = [self.target_critic1, self.target_critic2]

        for i in range(2):
            for target_param, param in zip(target_nets[i].parameters(), critic_nets[i].parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def update_temperature(self, log_pi):
        """Update the temperature (entropy formula)"""
        # compute entropy loss and update optimizer
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.update_optimizers(self.alpha_optimizer, alpha_loss)
        # update alpha and step
        self.alpha = self.log_alpha.exp()
        self.update_step += 1

    def update(self):
        """Perform the whole SAC algorithm"""
        # get memory - learn from experience
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.model_name)

        # predict actions and log of policy
        next_actions, next_log_pi = self.actor.sample(next_states)
        # predict q1 and q2 values
        pred_q1, pred_q2 = self.target_critic1(next_states, next_actions), self.target_critic2(next_states,
                                                                                               next_actions)
        # predict q value of the target network
        pred_q_target = torch.min(pred_q1, pred_q2) - self.alpha * next_log_pi
        # predict overall expected q value
        exp_q = rewards + (1 - dones) * self.gamma * pred_q_target

        # q loss - critic loss
        q1_value, q2_value = self.critic1.forward(states, actions), self.critic2.forward(states, actions)
        q1_loss, q2_loss = F.mse_loss(q1_value, exp_q.detach()), F.mse_loss(q2_value, exp_q.detach())

        # update q networks
        self.update_optimizers(self.c1_optimizer, q1_loss)
        self.update_optimizers(self.c2_optimizer, q2_loss)

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.actor.sample(states)
        if self.update_step % self.delay_freq == 0:
            self.delayed_update(states, new_actions, log_pi)

        # update temperature
        self.update_temperature(log_pi)

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


def train(env, paths, agent, max_episodes, max_steps, batch_size, verbose=1, render=False, _nni=False, noise=False):
    avg_scores = deque(maxlen=100)
    scores, episode_rewards = [], []
    time_start = time.time()
    update_step = 0

    vid = video_recorder.VideoRecorder(env, path="./video/vid_train_sac.mp4")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")

    for episode in range(max_episodes):
        state = env.reset()

        if noise:  # add uncertainty if required
            state = uncertainty(state)

        episode_reward = 0

        for step in range(max_steps):
            if verbose == 3:
                env.render()
                vid.capture_frame()

            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.append(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.memory.buffer) > batch_size:
                agent.update()
                update_step += 1

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                break

            state = next_state
        scores.append(episode_reward)
        avg_scores.append(episode_reward)
        avg_score = np.mean(avg_scores)

        if verbose > 1 and not _nni:
            s = int(time.time() - time_start)
            print(f'Episode {episode} Score: {episode_reward:.2f} Average Score: {avg_score:.2f},'
                  f' Time: {s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}')

        if episode % 20 == 0 and _nni:  # report intermediate results for nni
            nni.report_intermediate_result(episode)

        if len(avg_scores) == 100 and avg_score >= 200:  # if this condition is True then we solved the environment
            if verbose > 1:
                print('Environment solved !   ')
                print(f"\nEnvironment solved in {episode:d} episodes!\tAverage Score: {avg_score:.2f}")

            a = str(nni.get_trial_id()) if _nni else curr_time
            if verbose > 0:
                if noise:
                    agent.save(paths['solved_dir'] + agent.model_name + '_' + a + 'uncertainty_best')
                else:
                    agent.save(paths['solved_dir'] + agent.model_name + '_' + a + '_best')

                if not _nni:  # final plot
                    # final plot
                    if noise:
                        plot(scores, agent.model_name,
                             filename=paths['plot_dir'] + agent.model_name + '_train_uncertainty' + curr_time + '.png',
                             noise=noise)
                    else:
                        plot(scores, agent.model_name, filename=paths['plot_dir'] + agent.model_name + '_train_' +
                                                                curr_time + '.png', noise=noise)
            break

    if _nni:
        # report final results of NNI
        nni.report_final_result(episode)

    env.close()
