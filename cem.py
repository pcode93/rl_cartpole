import gym
import numpy as np
from collections import deque

env = gym.make('CartPole-v0')

def eval_theta(theta, render=False):
    observation = env.reset()
    summed_reward = 0

    for t in range(1000):
        if render:
            env.render()

        action = int(1.0 / (1.0 + np.exp(-np.dot(theta, observation))) > 0.5)
        observation, reward, done, info = env.step(action)
        summed_reward += reward

        if done:
            return summed_reward

n_iter = 250
mu = 25
perc_best = 0.25
dim = env.observation_space.shape[0]
theta_mean = np.zeros(dim)
theta_std = np.ones(dim)
total_rewards = deque(maxlen=25)
average_rewards = []

for i in range(n_iter):
    thetas = np.random.normal(theta_mean, theta_std, (mu, dim))
    rewards = [eval_theta(theta) for theta in thetas]

    best, _ = zip(*list(sorted(zip(thetas, rewards), key=lambda x: x[1], reverse=True))[:int(perc_best * mu)])
    theta_mean = np.mean(best, 0)
    theta_std = np.std(best, 0)

    total_rewards.append(eval_theta(best[0], render=True))
    average_rewards.append(np.mean(total_rewards))

    if i % 10 == 0:
        print(np.mean(total_rewards))
