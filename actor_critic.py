import numpy as np
import gym
import math
import random

lr = 0.01
def actor(x, w):
    return 1 / (1 + math.exp(-np.dot(x, w)))

def critic(x, w):
    return np.dot(x, w)

def bernoulli(p):
    return 0 if random.uniform(0, 1) < 1 - p else 1

def discount(rewards):
    summed_reward = 0
    discounted = []

    for i, reward in reversed(list(enumerate(rewards))):
        summed_reward += reward
        discounted.append(summed_reward)

    return list(reversed(discounted))

def update(rewards, values, grads, obs, W_actor, W_critic):
    critic_updates = [(v - r) * ob for v, r, ob in zip(values, rewards, obs)]
    actor_updates = [grad * ob * (r - v) for grad, ob, r, v in zip(grads, obs, rewards, values)]
    return W_actor + lr * np.sum(actor_updates, axis=0), W_critic - lr * np.sum(critic_updates, axis=0)

W_actor = np.random.normal(size=4)
W_critic = np.random.normal(size=4)
env = gym.make('CartPole-v0')

for i_episode in range(5000):
    obs = []
    grads = []
    rewards = []
    values = []
    observation = env.reset()

    for t in range(1000):
        env.render()
        obs.append(observation)
        values.append(critic(observation, W_critic))

        probability = actor(observation, W_actor)
        action = bernoulli(probability)
        grads.append(action - probability)

        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        if done:
            print(i_episode, t + 1)

            W_actor, W_critic = update(discount(rewards), values, grads, obs, W_actor, W_critic)
            break
