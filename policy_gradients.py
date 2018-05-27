import numpy as np
import gym
import math
import random

lr = 0.01
def actions(x, w):
	return 1 / (1 + math.exp(-np.dot(x,w)))

def bernoulli(p):
	return 0 if random.uniform(0, 1) < 1 - p else 1

def pad(x):
    return np.pad(x, (0, 200 - len(x)), 'constant', constant_values=(0,0))

def discount(rewards):
	summed_reward = 0
	discounted = []

	for i, reward in reversed(list(enumerate(rewards))):
		summed_reward += reward
		discounted.append(summed_reward)

	return list(reversed(discounted))

def update(rewards, baselines, grads, obs, w):
	updates = [grad * ob * (r - b) for grad, ob, r, b in zip(grads, obs, rewards, baselines)]
	return w + lr * np.sum(updates, axis=0)

weights = np.random.normal(size=4)
env = gym.make('CartPole-v0')

summed_rewards = []
avg_rewards = np.zeros(200)

for i_episode in range(5000):
	obs = []
	grads = []
	rewards = []
	observation = env.reset()

	for t in range(1000):
		env.render()
		obs.append(observation)

		probability = actions(observation, weights)
		action = bernoulli(probability)
		grads.append(action - probability)

		observation, reward, done, info = env.step(action)
		rewards.append(reward)

		if done:
		    print(i_episode, t + 1)

		    rewards = discount(rewards)
		    summed_rewards.append(pad(rewards))
		    avg_rewards = np.mean(summed_rewards, 0)

		    weights = update(rewards, avg_rewards[:len(rewards)], grads, obs, weights)
		    break
