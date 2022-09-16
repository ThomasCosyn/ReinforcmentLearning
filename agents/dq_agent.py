import torch
import torch.nn as nn
from gym.spaces import Box
import torch.optim as optim
from collections import deque
from citylearn.citylearn import CityLearnEnv
import random
import numpy as np

class DQN(nn.Module):

    def __init__(self, env, learning_rate):

        super(DQN,self).__init__()
        input_features = env.observation_space[0].shape[0]
        action_space = env.action_space[0].shape[0]

        self.dense1 = nn.Linear(in_features = input_features, out_features = 128)
        self.dense2 = nn.Linear(in_features = 128, out_features = 64)
        self.dense3 = nn.Linear(in_features = 64, out_features = 32)
        self.dense4 = nn.Linear(in_features = 32, out_features = action_space)

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

    def forward(self, x):

        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = torch.tanh(self.dense4(x))

        return x

class ExperienceReplay:

    def __init__(self, env, buffer_size, min_replay_size = 1000):

        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen = buffer_size)
        self.reward_buffer = deque([-200.0], maxlen = 100)

        print('Please wait, the experience replay buffer will be filled with random transitions')

        obs = self.env.reset()
        for _ in range(self.min_replay_size):

            action = [env.action_space[0].sample() for i in range(len(env.action_space))]
            new_obs, rew, done, _ = env.step(action)
            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                obs = env.reset()

        print('Initialization with random transitions is done!')

    def add_data(self, data):
        self.replay_buffer.append(data)

    def sample(self, batch_size):

        # Echantillonage d'un batch de transitions
        transitions = random.sample(self.replay_buffer, batch_size)
        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])

        # Conversion en tensors
        observations_t = torch.as_tensor(observations, dtype = torch.float32)
        actions_t = torch.as_tensor(actions, dtype = torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype = torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype = torch.float32).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype = torch.float32)

        return observations_t, actions_t, rewards_t, dones_t, new_observations_t


class dqAgent:

    def __init__(self) -> None:
        pass  