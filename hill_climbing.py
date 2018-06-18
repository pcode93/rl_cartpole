import gym
import numpy as np
from collections import deque

env = gym.make('CartPole-v0')
total_rewards = deque(maxlen=25)
average_rewards = []

W_best = np.random.rand(*env.observation_space.shape)
best_reward = 0

for iteration in range(250):
    observation = env.reset()
    summed_reward = 0
    W = W_best + 0.01 * (200 - best_reward) * np.random.random()

    for t in range(1000):
        env.render()
        action = int(1.0 / (1.0 + np.exp(-np.dot(W, observation))) > 0.5)

        observation, reward, done, info = env.step(action)
        summed_reward += reward

        if done:
            total_rewards.append(summed_reward)
            average_rewards.append(np.mean(total_rewards))

            if iteration % 10 == 0:
                print(np.mean(total_rewards))

            if summed_reward > best_reward:
                best_reward = summed_reward
                W_best = W

            break