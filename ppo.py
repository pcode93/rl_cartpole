import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.distributions import Categorical
from collections import deque

import numpy as np
import gym

class Net(nn.Module):
    def __init__(self, n_hidden):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(4, n_hidden)
        self.actor = nn.Linear(n_hidden, 2)
        self.critic = nn.Linear(n_hidden, 1)

    def forward(self, x):
        features = F.relu(self.fc1(x))
        return nn.Softmax()(self.actor(features)), self.critic(features)

env = gym.make('CartPole-v0')

n_epochs = 5000
n_iters = 4
gamma = 0.99
lam = 0.95
eps = 0.2
T = 512
num_batches = 16
batch_size = T // num_batches

model = Net(64)
optimizer = optim.Adam(model.parameters(), lr=2.5e-4)

def advantages(rewards, values, dones):
    summed_reward = 0
    returns = [0] * len(rewards)

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (1 - dones[i]) * values[i + 1] - values[i]
        summed_reward = summed_reward * (1 - dones[i]) * gamma * lam + delta
        returns[i] = summed_reward

    return returns

def var(x):
    return Variable(torch.from_numpy(x).float())

total_rewards = deque(maxlen=100)
observation = env.reset()
current_reward = 0
num_eps = 0

for i_episode in range(n_epochs):
    old_probs = []
    actions = []
    states = []
    rewards = []
    values = []
    dones = []

    for t in range(T):
        if num_eps % 50 == 0:
            env.render()

        probs, value = model(var(observation))
        states.append(observation)

        m = Categorical(probs.data)
        action = m.sample()
    
        actions.append(action[0])
        old_probs.append(m.log_prob(action)[0])

        observation, reward, done, info = env.step(action[0])
        rewards.append(reward)
        values.append(value.data[0])
        dones.append(done)
        current_reward += reward

        if done:
            num_eps += 1
            total_rewards.append(current_reward)

            current_reward = 0
            observation = env.reset()
    
    _, value = model(var(observation))

    advs = np.vstack(advantages(rewards, values + [value.data[0]], dones))
    returns = np.vstack(values) + np.copy(advs)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    states = np.vstack(states)
    old_probs = np.vstack(old_probs)

    batch_range = list(range(batch_size))

    for n in range(n_iters):
        idx = np.random.permutation(T)
        
        for num_batch in range(num_batches):
            b_idx = idx[num_batch * batch_size : (num_batch + 1) * batch_size]

            b_advs = np.copy(advs[b_idx])
            b_returns = np.copy(returns[b_idx])
            b_old_probs = np.copy(old_probs[b_idx])
            b_states = np.copy(states[b_idx])
            b_actions = np.copy(np.array(actions)[b_idx])

            optimizer.zero_grad()

            probs, values = model(var(b_states))
            r = torch.exp(torch.log(probs[batch_range, b_actions]) - var(b_old_probs).squeeze(1))
            A = var(b_advs).squeeze(1)

            L_policy = -torch.min(r * A, r.clamp(1 - eps, 1 + eps) * A).mean()
            L_value = (values - var(b_returns)).pow(2).mean()
            L_entropy = (-((probs * torch.log(probs)).sum(-1))).mean()

            L = L_value + L_policy - 0.01 * L_entropy
            
            L.backward()
            optimizer.step()

    if i_episode % 10 == 0:
        print("Episode: {0}, Last 100 mean: {1}".format(
            i_episode, np.mean(total_rewards)))
