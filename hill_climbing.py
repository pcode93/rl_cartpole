import gym
import numpy as np

env = gym.make('CartPole-v0')

W_best = np.random.rand(*env.observation_space.shape)
best_reward = 0

for iteration in range(200):
    observation = env.reset()
    summed_reward = 0
    W = W_best + (2 * np.random.random() - 1) if best_reward < 200 else W_best

    for t in range(1000):
        env.render()
        action = int(1.0 / (1.0 + np.exp(-np.dot(W, observation))) > 0.5)

        observation, reward, done, info = env.step(action)
        summed_reward += reward

        if done:
            if summed_reward > best_reward:
                best_reward = summed_reward
                W_best = W

            print(iteration + 1, summed_reward + 1)
            break