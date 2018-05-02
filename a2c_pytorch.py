import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.distributions import Categorical
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
n_iters = 10000
gamma = 0.99

model = Net(64)

optimizer = optim.RMSprop(model.parameters()) 
critic_loss = nn.SmoothL1Loss()

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

for i_episode in range(n_epochs):
    rewards = []
    logps = []
    values = []
    
    observation = env.reset()
    optimizer.zero_grad()

    for t in range(n_iters):
        if i_episode % 10 == 0:
            env.render()
        
        probs, value = model(var(observation))
        
        m = Categorical(probs)
        action = m.sample()
    
        logps.append(-m.log_prob(action))
        values.append(value)

        observation, reward, done, info = env.step(action.data[0])
        rewards.append(reward)
          
        if done:
            if i_episode % 10 == 0:
                print(i_episode, t + 1)
        
            rewards = discount(rewards)

            for logp, reward, value in zip(logps, rewards, values):
                (logp * (reward - value.data[0])).backward(retain_graph=True)
            
            critic_loss(torch.stack(values), var(np.array(rewards))).backward()

            optimizer.step()
            break
