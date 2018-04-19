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
n_iters = 5
gamma = 0.99
eps = 0.2

model = Net(64)
optimizer = optim.Adam(model.parameters()) 

def discount(rewards):
    summed_reward = 0
    discount = 1
    discounted = []

    for i, reward in reversed(list(enumerate(rewards))):
        summed_reward += reward
        discount *= gamma
        discounted.append(discount * summed_reward)

    return list(reversed(discounted))

def var(x):
    return Variable(torch.from_numpy(x).float())

total_rewards = deque(maxlen=100)
for i_episode in range(n_epochs):
    old_probs = []
    actions = []
    states = []
    rewards = []
    values = []
    
    observation = env.reset()
    optimizer.zero_grad()

    for t in range(200):
        if i_episode % 10 == 0:
            env.render()
        
        probs, value = model(var(observation))
        states.append(observation)

        m = Categorical(probs)
        action = m.sample()
    
        actions.append(action.data[0])
        old_probs.append(m.probs[action].data[0])

        observation, reward, done, info = env.step(action.data[0])
        rewards.append(reward)
        values.append(value.data[0])
          
        if done:
            total_reward = sum(rewards)
            total_rewards.append(total_reward)

            if i_episode % 10 == 0:
                print("Episode: {0}, Score: {1}, Last 100 mean: {2}".format(
                    i_episode, total_reward, np.mean(total_rewards)))
        
            rewards = np.vstack(discount(rewards))
            states = np.vstack(states)
            old_probs = np.vstack(old_probs)
            values = np.vstack(values)
   
            advantages = rewards - values
            batch = list(range(len(actions)))

            for n in range(n_iters):
                optimizer.zero_grad()

                probs, values = model(var(states))
                r = probs[batch, actions] / var(old_probs).squeeze(1)
                A = var(advantages).squeeze(1)

                L_policy = torch.min(r * A, r.clamp(1 - eps, 1 + eps) * A)
                L_value = nn.MSELoss()(values, var(rewards))
                L_entropy = -((probs * torch.log(probs)).sum(1))
  
                L = -(L_policy - L_value + 0.01 * L_entropy)
                
                L.sum().backward()
                optimizer.step()

            break
