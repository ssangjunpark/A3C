import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import numpy as np

from util import create_networks, image_transformer

gym.register_envs(ale_py)

class Worker:
    def __init__(self, environmen_id, model_size_initializers):
        self.env = gym.make(environmen_id, render_mode='rgb_array')

        # create own copy of models
        self.policy_model, self.value_model = create_networks(self.env.action_space.n, model_size_initializers[0], model_size_initializers[1], model_size_initializers[2], model_size_initializers[3])

    def copy_param_from_parent():
        pass

    def update_parnet_param():
        pass
    
    def play_episode(self, update_period_steps):
        current_step = 0
        done = False

        obs, info = self.env.reset()
        rewards_at_terminal = 0

        obs_transformed = image_transformer(obs, [84,84])
        # we stack 4 inital frame as our state // as shape (H x W x 4)
        obs = np.stack([obs_transformed] * 4, axis=2)

        while not done:
            pi_eval = self.policy_model.predict(obs)
            action = np.random.choice(self.env.action_space.n, 1, p=pi_eval)
            
            observation, reward, done, truncated, info = self.env.step(action)

            current_step += 1
            if (done) or (current_step > update_period_steps):
                # we want to clauclate the gradient and update the main network
                print('update main')
                current_step = 0

            rewards_at_terminal += reward
            obs_transformed = image_transformer(observation, [84, 84])
            obs = np.append(obs[:, :, 1:], np.expand_dims(obs_transformed, axis=2), axis=2)