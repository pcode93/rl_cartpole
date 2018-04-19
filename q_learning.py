import random
import gym
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class Memory:
    def __init__(self, state, action, reward, next_state, terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal

GAMMA = 0.99
EPSILON = 1.0
TARGET_UPDATE_INTERVAL = 100
BATCH_SIZE = 32

q_func = Net(4, 2, 128)
target_func = Net(4, 2, 128)
target_func.load_state_dict(q_func.state_dict())

optimizer = optim.RMSprop(q_func.parameters())
loss = nn.SmoothL1Loss()
memories = deque(maxlen=1000)

env = gym.make('CartPole-v0')

def var(x):
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    return Variable(torch.from_numpy(x)).type(torch.FloatTensor)

def select_action(x):
    if random.random() <= EPSILON:
        return env.action_space.sample()
    else:
        return q_func(var(x)).data.max(0)[1].view(1, 1)[0, 0]

step_count = 0
for i_episode in range(5000):
    observation = env.reset()

    for t in range(1000):
        if i_episode % 10 == 0:
            env.render()

        step_count += 1

        previous_obs = observation
        action = select_action(observation)
        observation, reward, done, info = env.step(action)

        memories.append(Memory(previous_obs, action, reward, observation, done))

        EPSILON = max(0.1, 1 - step_count / 3000)

        if step_count >= 1000:
            optimizer.zero_grad()

            batch = np.random.choice(memories, min(BATCH_SIZE, len(memories)), replace=False)

            for memory in batch:
                target = var(memory.reward + (GAMMA * target_func(var(memory.next_state)).data.max(0)[0] if not memory.terminal else 0))
                output = q_func(var(memory.state))[memory.action]

                loss(output, target).backward()

            optimizer.step()

            if step_count % TARGET_UPDATE_INTERVAL == 0:
                target_func.load_state_dict(q_func.state_dict())

        if done:
            if i_episode % 10 == 0:
                print(i_episode, t + 1)
            break