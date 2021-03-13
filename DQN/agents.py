import nni
import torch.optim as optim
import torch.nn.functional as F
import time
from utils import *
from replay_all import *
from models import DQN, DuelingDQN


class Agent:
    """Parent class agent of all the agents"""
    def __init__(self, max_memory_size, batch_size, eps_max, eps_min, decay_rate, device):

        self.model_name = "Base"
        self.device = device

        # policy
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.decay_rate = decay_rate
        self.curr_step = 0
        self.curr_eps = eps_max
        self.output_dim = 0
        self.policy_net = None

        # replay memory
        self.memory = ReplayMemory(max_memory_size, device)
        self.batch_size = batch_size

        # for mp4 recording
        self.vid = None

    def decay_eps(self):
        """
        Perform epsilon decay- choose the minimum value of the minimum epsilon and the current
        one multiplied by the epsilon decay value
        """
        return max(self.eps_min, self.curr_eps * self.decay_rate)

    def choose_action(self, curr_state, is_test=False):
        """
        Choose an action using epsilon greedy.
        :param curr_state: current state of the environment
        :param is_test: if True, always choose greedy action (in testing we always choose the greedy action)
        :return: action
        """
        self.curr_step += 1
        # epsilon greedy
        if not is_test and np.random.random() < self.curr_eps:
            return np.random.randint(0, self.output_dim)
        else:
            # choose the greedy action
            with torch.no_grad():
                curr_state = torch.tensor(curr_state, dtype=torch.float, device=self.device)
                return self.policy_net(curr_state).argmax().item()

    def learn(self):
        pass

    def set_test(self):
        """ Sets the network in evaluation mode """
        self.policy_net.eval()

    def save(self, filename):
        """
        Save the network weights.
        :param filename: path
        """
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        """
        Load the network weights.
        :param filename: path of the weights file
        """
        self.policy_net.load_state_dict(torch.load(filename + ".pth", map_location=self.device))

    def train(self, env, paths, num_episodes=2000, steps=1000, learn_freq=4, verbose=2, _nni=True, noise=False):
        """
        Training the agents, saving weights when env is solved and plotting (save and plot according to verbose)
        :param env: openai gym environment
        :param paths: dictionary that contains: {solved_dir: weight files dir, plots_dir}
        :param num_episodes: number of episodes to perform training
        :param steps: max time steps per episode
        :param learn_freq: how often should the networks weight be updated
        :param verbose: verbose mode as described in parser
        :param _nni: True if running nni, else False
        :param noise: True for uncertainty states, else False
        """
        avg_scores = deque(maxlen=100)
        scores, losses, epsilons = [], [], []
        updates = 0

        time_start = time.time()

        self.policy_net.train()
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")

        for ep in range(1, num_episodes + 1):
            state = env.reset()

            if noise:  # add uncertainty if required
                state = uncertainty(state)

            score = 0
            for t in range(1, steps + 1):
                if verbose == 3:  # video if required
                    env.render()
                    self.vid.capture_frame()

                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                score += reward
                state = next_state

                if len(self.memory.buffer) >= self.batch_size and self.curr_step % learn_freq == 0:
                    losses.append(self.learn())
                    updates += 1

                if done:
                    self.curr_eps = self.decay_eps()
                    break

            avg_scores.append(score)
            scores.append(score)
            epsilons.append(self.curr_eps)
            avg_reward = np.mean(avg_scores)

            if verbose > 1:
                if not _nni:
                    s = int(time.time() - time_start)
                    print(f"Episode {ep} Score: {score:.2f}. Average Score: {avg_reward:.2f}  "
                          f"Time: {s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}")

            if ep % 20 == 0 and _nni:  # report intermediate results for nni
                nni.report_intermediate_result(ep)

            if avg_reward > 200 and len(avg_scores) == 100:  # if this condition is True then we solved the environment
                if verbose > 1:
                    print(f"\nEnvironment solved! score {avg_reward} at episode {ep}.")

                a = str(nni.get_trial_id()) if _nni else curr_time
                if noise:
                    self.save(paths['solved_dir'] + self.model_name + '_solved_' + a + 'uncertainty_best'+".pth")
                else:
                    self.save(paths['solved_dir'] + self.model_name + '_solved_' + a + '_best'+'.pth')

                if not _nni:  # final plot
                    # final plot
                    if noise:
                        plot(scores, self.model_name,
                             filename=paths['plot_dir'] + self.model_name + '_train_uncertainty' + curr_time + '.png',
                             noise=noise)
                    else:
                        plot(scores, self.model_name, filename=paths['plot_dir'] + self.model_name + '_train_' +
                                                                curr_time + '.png', noise=noise)

                break

        if verbose > 0:
            print("Training finished.")

        if _nni:
            nni.report_final_result(ep)
            print("Saved final results of nni")

        env.close()


class DQNAgent(Agent):
    """Agent of the DQN network."""
    def __init__(self, input_dim, output_dim, lr, gamma, max_memory_size, batch_size, eps_start, eps_end, eps_decay,
                 device, vid):

        super().__init__(max_memory_size, batch_size, eps_start, eps_end, eps_decay, device)
        self.model_name = "DQN"
        self.gamma = gamma  # discount factor
        self.vid = vid  # for video

        # define the policy network
        self.hid_size1 = 64
        self.hid_size2 = 64
        self.output_dim = output_dim
        self.policy_net = DQN(input_dim, output_dim, self.hid_size1, self.hid_size2).to(device)

        # define optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def learn(self):
        """Update the network weights and return the loss"""
        # sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.model_name)
        # calculate current and next q values
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.policy_net(next_states).max(1, keepdim=True)[0].detach()
        # calculate target by formula
        target = (rewards + self.gamma * next_q_values * (1 - dones)).to(self.device)
        # compute loss and update optimizer
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class FullDQNAgent(DQNAgent):
    """An expansion of the DQN Agent, with a target net for the Q-targets."""
    def __init__(self, input_dim, output_dim, lr, gamma, max_memory_size, batch_size, eps_start, eps_end, eps_decay,
                 device, vid, target_update=100):

        super().__init__(input_dim, output_dim, lr, gamma, max_memory_size, batch_size, eps_start, eps_end, eps_decay,
                         device, vid)

        self.model_name = "FullDQN"
        self.target_update = target_update  # frequency of target network update
        self.updated = 0

        # define target network
        self.output_dim = output_dim
        self.target_net = DQN(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def learn(self):
        """
        Q-targets are computed using the target network and this way the network weights are updated.
        Loss is returned
        """
        # sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.model_name)
        # calculate current and next q values
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1, keepdim=True)[0].detach()
        # calculate target by formula
        target = (rewards + self.gamma * next_q_values * (1 - dones)).to(self.device)
        # compute loss and update optimizer
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updated += 1
        # Every self.target_update updates, clone the policy_net
        if self.updated % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()


class DoubleDQNAgent(FullDQNAgent):
    """ An expansion of the FullQNAgent is DoubleDQN Agent."""
    def __init__(self, input_dim, output_dim, lr, gamma, max_memory_size, batch_size, eps_start, eps_end, eps_decay,
                 device, vid, target_update=100):

        super().__init__(input_dim, output_dim, lr, gamma, max_memory_size, batch_size, eps_start, eps_end, eps_decay,
                         device, vid, target_update)

        self.model_name = "DoubleDQN"

    def learn(self):
        """
        Q-targets are computed using the target network and this way the network weights are updated.
        Loss is returned
        """
        # sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.model_name)
        # calculate current q value, next argmax q value (greedy) and next q values
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values_argmax = self.policy_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states).gather(1, next_q_values_argmax.unsqueeze(1)).detach()
        # calculate target by formula
        target = (rewards + self.gamma * next_q_values * (1 - dones)).to(self.device)
        # compute loss and update optimizer
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updated += 1
        # Every self.target_update updates, clone the policy_net
        if self.updated % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()


class DuelingDDQNAgent(Agent):
    """
    DuelingDQN is an expansion of the base agent, and its architecture is the dueling one, as explained in the
    paper, where both policy and target networks exist.
    """
    def __init__(self, input_dim, output_dim, lr, gamma, max_memory_size, batch_size, eps_start, eps_end, eps_decay,
                 device, vid, target_update):

        super().__init__(max_memory_size, batch_size, eps_start, eps_end, eps_decay, device)

        self.model_name = "DuelingDDQN"
        self.gamma = gamma  # discount factor
        self.vid = vid  # for video
        self.updated = 0
        self.target_update = target_update  # frequency of target network update

        # define hidden sizes
        self.output_dim = output_dim
        self.hid_size_linear = 64
        self.hid_size_adv = 32
        self.hid_size_val = 32

        # define policy net
        self.policy_net = DuelingDQN(input_dim, output_dim, self.hid_size_linear, self.hid_size_adv, self.hid_size_val)\
            .to(device)
        # define target net
        self.target_net = DuelingDQN(input_dim, output_dim, self.hid_size_linear, self.hid_size_adv, self.hid_size_val)\
            .to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # define optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def learn(self):
        """
        Q-targets are computed using the target network and this way the network weights are updated.
        Loss is returned
        """
        # sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.model_name)
        # calculate current q value, next argmax q value (greedy) and next q values
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values_argmax = self.policy_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states).gather(1, next_q_values_argmax.unsqueeze(1)).detach()
        # calculate target by formula
        target = (rewards + self.gamma * next_q_values * (1 - dones)).to(self.device)
        # compute loss and update optimizer
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updated += 1
        # Every self.target_update updates, clone the policy_net
        if self.updated % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()