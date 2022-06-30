import argparse
import sys
import random
import os
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["OMP_NUM_THREADS"] = "10"
import numpy as np
import scipy
from params import Params
import pickle
import time
import statistics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from operator import add, sub

import pickle

NUM_ENV = 4

class PPOStorage:
    def __init__(self, num_inputs, num_outputs, max_size=64000):
        self.states = torch.zeros(max_size, num_inputs).to(device)
        self.next_states = torch.zeros(max_size, num_inputs).to(device)
        self.actions = torch.zeros(max_size, num_outputs).to(device)
        self.dones = torch.zeros(max_size, 1, dtype=torch.int8).to(device)
        self.log_probs = torch.zeros(max_size).to(device)
        self.rewards = torch.zeros(max_size).to(device)
        self.q_values = torch.zeros(max_size, 1).to(device)
        self.mean_actions = torch.zeros(max_size, num_outputs).to(device)
        self.counter = 0
        self.sample_counter = 0
        self.max_samples = max_size
    def sample(self, batch_size):
        idx = torch.randint(self.counter, (batch_size,), device=device)
        return self.states[idx, :], self.actions[idx, :], self.next_states[idx, :], self.rewards[idx], self.q_values[idx, :], self.log_probs[idx]
    def clear(self):
        self.counter = 0
    def push(self, states, actions, next_states, rewards, q_values, log_probs, size):
        self.states[self.counter:self.counter + size, :] = states.detach().clone()
        self.actions[self.counter:self.counter + size, :] = actions.detach().clone()
        self.next_states[self.counter:self.counter + size, :] = next_states.detach().clone()
        self.rewards[self.counter:self.counter + size] = rewards.detach().clone()
        self.q_values[self.counter:self.counter + size, :] = q_values.detach().clone()
        self.log_probs[self.counter:self.counter + size] = log_probs.detach().clone()
        self.counter += size

    def discriminator_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.next_states[self.sample_counter - batch_size:self.sample_counter, :]

    def critic_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.q_values[self.sample_counter - batch_size:self.sample_counter,:]

    def actor_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.actions[self.sample_counter - batch_size:self.sample_counter, :], self.q_values[self.sample_counter - batch_size:self.sample_counter, :], self.log_probs[self.sample_counter - batch_size:self.sample_counter]

    def permute(self):
        permuted_index = torch.randperm(self.max_samples)
        self.states[:, :] = self.states[permuted_index, :]
        self.actions[:, :] = self.actions[permuted_index, :]
        self.q_values[:, :] = self.q_values[permuted_index, :]
        self.log_probs[:] = self.log_probs[permuted_index]


class RL(object):
    def __init__(self, env, hidden_layer=[64, 64]):
        self.env = env
        # self.env.env.disableViewer = False
        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        self.hidden_layer = hidden_layer

        self.params = Params()
        self.Net = ActorCriticNet
        self.model = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer)
        self.discriminator = Discriminator(10 * 2, [128, 128]) #<---Dsicriminator initialization FIX
        self.model.share_memory()
        self.test_mean = []
        self.test_std = []

        self.noisy_test_mean = []
        self.noisy_test_std = []
        self.fig = plt.figure()
        # self.fig2 = plt.figure()
        self.lr = self.params.lr
        plt.show(block=False)

        self.test_list = []
        self.noisy_test_list = []

        self.best_score_queue = mp.Queue()
        self.best_score = mp.Value("f", 0)
        self.max_reward = mp.Value("f", 5)

        self.best_validation = 1.0
        self.current_best_validation = 1.0


        self.gpu_model = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer)
        self.gpu_model.to(device)
        self.model_old = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer).to(device)
        self.discriminator.to(device)

        self.base_controller = None
        self.base_policy = None

        self.total_rewards = []

    def run_test(self, num_test=1):
        state = self.env.reset()
        ave_test_reward = 0

        total_rewards = []
        if self.num_envs > 1:
            test_index = 1
        else:
            test_index = 0

        for i in range(num_test):
            total_reward = 0
            while True:
                state = self.shared_obs_stats.normalize(state)
                mu = self.gpu_model.sample_best_actions(state)
                state, reward, done, _ = self.env.step(mu)
                total_reward += reward[test_index].item()

                if done[test_index]:
                    state = self.env.reset()
                    # print(self.env.position)
                    # print(self.env.time)
                    ave_test_reward += total_reward / num_test
                    total_rewards.append(total_reward)
                    break
        # print("avg test reward is", ave_test_reward)
        reward_mean = statistics.mean(total_rewards)
        reward_std = statistics.stdev(total_rewards)
        self.test_mean.append(reward_mean)
        self.test_std.append(reward_std)
        self.test_list.append((reward_mean, reward_std))
        # print(self.model.state_dict())

    def run_test_with_noise(self, num_test=10):

        reward_mean = statistics.mean(self.total_rewards)
        reward_std = statistics.stdev(self.total_rewards)
        # print(reward_mean, reward_std, self.total_rewards)
        self.noisy_test_mean.append(reward_mean)
        self.noisy_test_std.append(reward_std)
        self.noisy_test_list.append((reward_mean, reward_std))

        print("reward mean,", reward_mean)
        print("reward std,", reward_std)

    def save_reward_stats(self, stats_name):
        with open(stats_name, 'wb') as f:
            np.save(f, np.array(self.noisy_test_mean))
            np.save(f, np.array(self.noisy_test_std))

    def plot_statistics(self):

        plt.clf()
        ax = self.fig.add_subplot(121)
        # ax2 = self.fig.add_subplot(122)
        low = []
        high = []
        index = []
        noisy_low = []
        noisy_high = []
        for i in range(len(self.noisy_test_mean)):
            # low.append(self.test_mean[i] - self.test_std[i])
            # high.append(self.test_mean[i] + self.test_std[i])
            noisy_low.append(self.noisy_test_mean[i] - self.noisy_test_std[i])
            noisy_high.append(self.noisy_test_mean[i] + self.noisy_test_std[i])
            index.append(i)
        plt.xlabel('iterations')
        plt.ylabel('average rewards')
        # ax.plot(self.test_mean, 'b')
        ax.plot(self.noisy_test_mean, 'g')
        # ax.fill_between(index, low, high, color='cyan')
        ax.fill_between(index, noisy_low, noisy_high, color='r')
        # ax.plot(map(sub, test_mean, test_std))
        self.fig.canvas.draw()
        # plt.savefig("test.png")

    def collect_samples_vec(self, num_samples, start_state=None, noise=-2.5, env_index=0, random_seed=1):

        # if start_state == None:
        #     start_state = self.env.reset()
        start_state = self.env.reset()
        samples = 0
        done = False
        states = []
        next_states = []
        actions = []
        # mean_actions = []
        rewards = []
        values = []
        q_values = []
        real_rewards = []
        log_probs = []
        dones = []
        noise = self.base_noise * self.explore_noise.value
        self.gpu_model.set_noise(noise)

        state = start_state
        total_reward1 = 0
        total_reward2 = 0
        calculate_done1 = False
        calculate_done2 = False
        self.total_rewards = []
        start = time.time()
        state = torch.from_numpy(state).to(device).type(torch.cuda.FloatTensor) #DIFF
        while samples < num_samples:
            with torch.no_grad():
                action, mean_action = self.gpu_model.sample_actions(state)
                log_prob = self.gpu_model.calculate_prob(state, action, mean_action)

            states.append(state.clone())
            actions.append(action.clone())
            log_probs.append(log_prob.clone())
            next_state, reward, done, _ = self.env.step(action)

            # rewards.append(reward.clone())
            next_state = torch.from_numpy(next_state).to(device).type(torch.cuda.FloatTensor)
            reward = torch.from_numpy(reward).to(device).type(torch.cuda.FloatTensor)
            done = torch.from_numpy(done).to(device).type(torch.cuda.IntTensor)

            dones.append(done.clone())

            next_states.append(next_state.clone())

            reward = self.discriminator.compute_disc_reward(state[:, 0:10], next_state[:, 0:10]) * 0.0 + 1. * reward

            rewards.append(reward.clone())

            state = next_state.clone()

            samples += 1

            self.env.reset_time_limit()
        print("sim time", time.time() - start)
        start = time.time()
        counter = num_samples - 1
        R = self.gpu_model.get_value(state)
        while counter >= 0:
            R = R * (1 - dones[counter].unsqueeze(-1))
            R = 0.99 * R + rewards[counter].unsqueeze(-1)
            q_values.insert(0, R)
            counter -= 1
            # print(len(q_values))
        for i in range(num_samples):
            self.storage.push(states[i], actions[i], next_states[i], rewards[i], q_values[i], log_probs[i], self.num_envs)
        self.total_rewards = self.env.get_total_reward()
        print("processing time", time.time() - start)

    def load_motion_data(self):
        self.motion_data = np.loadtxt("2d_walking.txt")
        self.motion_data[:, 3:5] *= -1
        self.motion_data[:, 6:8] *= -1
        self.motion_data[:, 1] += 1.5

    def sample_motion_data(self, phase):
        motion_index = (phase * 10 / 50).astype(int)
        residual = (phase - motion_index * 5).astype(int)
        motion1 = self.motion_data[motion_index]
        motion2 = self.motion_data[motion_index + 1]
        residual = np.expand_dims(residual, axis=1)
        motion = motion1 * (5 - residual) / 5.0 + motion2 * residual / 5.0
        cos_phase = np.expand_dims(np.cos(phase * 3.1415 * 2 / 50), axis=1)
        sin_phase = np.expand_dims(np.sin(phase * 3.1415 * 2 / 50), axis=1)
        return torch.from_numpy(np.concatenate((motion[:, 1:9], sin_phase, cos_phase), axis=1).astype(np.float32)).to(device)

    def update_discriminator(self, batch_size, num_epoch):
        self.discriminator.train()
        optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        for k in range(num_epoch):
            batch_states, batch_next_states = self.storage.discriminator_sample(batch_size)
            policy_d = self.discriminator.compute_disc(batch_states[:, 0:10], batch_next_states[:, 0:10])
            policy_loss = (policy_d + torch.ones(policy_d.size(), device=device)) ** 2
            policy_loss = policy_loss.mean()

            phase = np.random.choice(45, batch_size)
            batch_expert_states = self.sample_motion_data(phase)
            batch_expert_next_states = self.sample_motion_data(phase + 1)
            expert_d = self.discriminator.compute_disc(batch_expert_states, batch_expert_next_states)
            expert_loss = (expert_d - torch.ones(expert_d.size(), device=device)) ** 2
            expert_loss = expert_loss.mean()

            grad_penalty = self.discriminator.grad_penalty(batch_expert_states, batch_expert_next_states)

            print(policy_loss, expert_loss, grad_penalty)

            total_loss = policy_loss + expert_loss + 5 * grad_penalty
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # removed fresh update
    def update_critic(self, batch_size, num_epoch):
        self.gpu_model.train()
        optimizer = optim.Adam(self.gpu_model.parameters(), lr=10 * self.lr)

        storage = self.storage
        gpu_model = self.gpu_model

        for k in range(num_epoch):
            batch_states, batch_q_values = storage.critic_sample(batch_size)
            batch_q_values = batch_q_values  # / self.max_reward.value
            v_pred = gpu_model.get_value(batch_states)

            loss_value = (v_pred - batch_q_values) ** 2
            loss_value = 0.5 * loss_value.mean()

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()


    def update_actor(self, batch_size, num_epoch):
        self.gpu_model.train()
        optimizer = optim.Adam(self.gpu_model.parameters(), lr=self.lr)

        storage = self.storage
        gpu_model = self.gpu_model
        model_old = self.model_old
        params_clip = self.params.clip

        for k in range(num_epoch):
            batch_states, batch_actions, batch_q_values, batch_log_probs = storage.actor_sample(batch_size)

            batch_q_values = batch_q_values  # / self.max_reward.value

            with torch.no_grad():
                v_pred_old = gpu_model.get_value(batch_states)

            batch_advantages = (batch_q_values - v_pred_old)

            probs, mean_actions = gpu_model.calculate_prob_gpu(batch_states, batch_actions)
            probs_old = batch_log_probs  # model_old.calculate_prob_gpu(batch_states, batch_actions)
            ratio = (probs - (probs_old)).exp()
            ratio = ratio.unsqueeze(1)
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(1 - params_clip, 1 + params_clip) * batch_advantages
            loss_clip = -(torch.min(surr1, surr2)).mean()

            total_loss = loss_clip + 0.001 * (mean_actions ** 2).mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        # print(self.shared_obs_stats.mean.data)
        if self.lr > 1e-4:
            self.lr *= 0.99
        else:
            self.lr = 1e-4

    def save_model(self, filename):
        torch.save(self.gpu_model.state_dict(), filename)

    def save_shared_obs_stas(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def save_statistics(self, filename):
        statistics = [self.time_passed, self.num_samples, self.test_mean, self.test_std, self.noisy_test_mean, self.noisy_test_std]
        with open(filename, 'wb') as output:
            pickle.dump(statistics, output, pickle.HIGHEST_PROTOCOL)

    def collect_samples_multithread(self):
        # queue = Queue.Queue()
        import time
        self.num_envs = NUM_ENV
        self.start = time.time()
        self.lr = 1e-3
        self.weight = 10
        num_threads = 1
        self.num_samples = 0
        self.time_passed = 0
        score_counter = 0
        total_thread = 0
        max_samples = 6000
        self.storage = PPOStorage(self.num_inputs, self.num_outputs, max_size=max_samples)
        seeds = [i * 100 for i in range(num_threads)]

        self.explore_noise = mp.Value("f", -2.5)
        self.base_noise = np.ones(self.num_outputs)
        noise = self.base_noise * self.explore_noise.value
        self.model.set_noise(noise)
        self.gpu_model.set_noise(noise)
        self.env.reset()
        self.load_motion_data()
        for iterations in range(100): #200000
            iteration_start = time.time()
            print(self.model_name)
            while self.storage.counter < max_samples:
                self.collect_samples_vec(300, noise=noise)
            start = time.time()

            self.update_critic(max_samples // 4, 40)
            self.update_actor(max_samples // 4, 40)
            self.update_discriminator(max_samples // 4, 40) #commented out
            self.storage.clear()

            if (iterations + 1) % 100 == 0:
                self.run_test_with_noise(num_test=2)
                self.plot_statistics()
                plt.savefig(self.model_name + "test.png")

            print("update policy time", time.time() - start)
            print("iteration time", iterations, time.time() - iteration_start)

            if (iterations + 1) % 100 == 0:
                self.save_model(self.model_name + "iter%d.pt" % (iterations))
                plt.savefig(self.model_name + "test.png")

        self.save_reward_stats("reward_stats.npy")
        self.save_model(self.model_name + "final.pt")

    def add_env(self, env):
        self.env_list.append(env)


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-log_dir', type=str, default='results/')
parser.add_argument('-exp_name', type=str, default='default_exp')
parser.add_argument('-time_steps', type=int, default=5e7)
parser.add_argument('-imp', action='store_true')  # use random impulses
parser.add_argument('-c_imp', action='store_true')  # use constant impulses

parser.add_argument('-f', type=float, default=300)  # define force
parser.add_argument('-ft', action='store_true')
parser.add_argument('-vis_ref', action='store_true')

parser.add_argument('-t', action='store_true')
parser.add_argument('-v', action='store_true')

if __name__ == '__main__':
    import json

    from my_pd_walker import PD_Walker2dEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env

    import torch
    import torch.optim as optim
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import torch.utils.data
    from model import ActorCriticNet, Discriminator

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 2  # 8
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    # create environment from the configuration file
    args = parser.parse_args()
    num_cpu = NUM_ENV
    env = PD_Walker2dEnv(args=args)
    train_env = make_vec_env(lambda: env, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    print("env_created")
    ppo = RL(train_env, [128, 128])

    ppo.base_dim = ppo.num_inputs

    ppo.model_name = "stats/test_amp/"
    ppo.max_reward.value = 1  # 50

    training_start = time.time()
    ppo.collect_samples_multithread()
    print("training time", time.time() - training_start)

