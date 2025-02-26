import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import numpy as np
from gymnasium.wrappers import RecordVideo

from util import create_networks, image_transformer

gym.register_envs(ale_py)

class Worker:
    def __init__(self, environmen_id, model_size_initializers, global_counter):
        self.env = gym.make(environmen_id, render_mode='rgb_array')
        self.env = RecordVideo(self.env, video_folder='test', episode_trigger= lambda x : True)
        self.global_counter = global_counter

        # create own copy of models
        self.policy_model, self.value_model = create_networks(self.env.action_space.n, model_size_initializers[0], model_size_initializers[1], model_size_initializers[2], model_size_initializers[3])

    def copy_param_from_parent(self, parent_policy_model, parent_value_model):
        self.policy_model.set_weights(parent_policy_model.get_weights())
        self.value_model.set_weights(parent_value_model.get_weights())
        # noice

    def update_parnet_param(self):
        # i dont like this 
        pass
    
    def play_episode(self, update_period_steps):
        current_step = 0
        done = False

        n_step_state = []
        n_step_action = []
        n_step_reward = []

        obs, info = self.env.reset()
        rewards_at_terminal = 0

        obs_transformed = image_transformer(obs, [84,84])
        # we stack 4 inital frame as our state // as shape (H x W x 4)
        obs = np.stack([obs_transformed] * 4, axis=2)

        while not done:
            pi_obs = np.expand_dims(obs, axis=0).astype(np.float32)
            pi_eval = self.policy_model.predict(pi_obs)
            action = np.random.choice(self.env.action_space.n, p=pi_eval[0])
            n_step_state.append(obs)
            n_step_action.append(action)

            observation, reward, done, truncated, info = self.env.step(action)
            n_step_reward.append(reward)

            current_step += 1
            global_step = next(self.global_counter)
            if (done) or (current_step > update_period_steps):
                # we want to clauclate the gradient and update the main network
                print('update main')

                self.update_parnet_param()

                n_step_state = []
                n_step_action = []
                n_step_reward = []

                current_step = 0

            rewards_at_terminal += reward
            obs_transformed = image_transformer(observation, [84, 84])
            obs = np.append(obs[:, :, 1:], np.expand_dims(obs_transformed, axis=2), axis=2)
            print(global_step)
        return global_step


    def run(self, total_number_of_steps, update_period_steps):
        global_step = 0
        while (total_number_of_steps > global_step):
            global_step = self.play_episode(update_period_steps)
