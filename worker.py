import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import numpy as np
from gymnasium.wrappers import RecordVideo

from util import create_networks, image_transformer

gym.register_envs(ale_py)

class Worker:
    def __init__(self, id_, environmen_id, model_size_initializers, global_counter, episode_counter, return_list, param_lock):
        self.id_ = id_
        self.env = gym.make(environmen_id, render_mode='rgb_array')
        # self.env = RecordVideo(self.env, video_folder='test', episode_trigger= lambda x : True)
        self.global_counter = global_counter
        self.return_list = return_list
        self.param_lock = param_lock
        self.episode_counter = episode_counter

        # create own copy of models
        # self.policy_model, self.value_model = create_networks(self.env.action_space.n, model_size_initializers[0], model_size_initializers[1], model_size_initializers[2], model_size_initializers[3])

        lol = gym.make("ALE/Breakout-v5", render_mode='rgb_array') # bad design fix it one day :(
        obs, _ = lol.reset()
        obs = image_transformer(obs, [84, 84])
        obs = np.stack([obs] * 4, axis=2)
        self.policy_model, self.value_model = create_networks(lol.action_space.n, model_size_initializers[0], model_size_initializers[1], model_size_initializers[2], model_size_initializers[3])
        self.policy_model.predict(np.expand_dims(obs, axis=0).astype(np.float32))
        self.value_model.predict(np.expand_dims(obs, axis=0).astype(np.float32))
        lol.close()

    def copy_param_from_parent(self, parent_policy_model, parent_value_model):
        with self.param_lock:
            self.policy_model.model.set_weights(parent_policy_model.model.get_weights())
            self.value_model.model.set_weights(parent_value_model.model.get_weights())
        # noice

    def save_parent_parameter(self, parent_policy_model, parent_value_model, policy_file_name, value_file_name):
        with self.param_lock:
            parent_policy_model.model.save_weights(policy_file_name)
            parent_value_model.model.save_weights(value_file_name)
            
    def update_parnet_param(self, states, actions, rewards, dones, parent_policy_model, parent_value_model):
        advantages = []
        returns = []
        gamma = 0.99
        if dones[-1] == True:
            g = 0
        else:
            g = self.value_model.predict(np.expand_dims(states[-1], axis=0).astype(np.float32))
        
        for i in range(len(states) - 1, -1, -1):
            g = rewards[i] + gamma * g
            returns.append(g)
            advantage = g - self.value_model.predict(np.expand_dims(states[i], axis=0).astype(np.float32))
            advantages.append(advantage)

        actions = np.array(actions).astype(np.int32)
        states = np.array(states).astype(np.float32)
        advantages = advantages[::-1]
        returns = returns[::-1]
        advantages = np.array(advantages).astype(np.float32)
        returns = np.array(returns).astype(np.float32)
        advantages = np.squeeze(advantages)
        returns = np.squeeze(returns)

        # print(actions.shape)
        # print(states.shape)
        # print(advantages.shape)
        # print(returns.shape)
        # exit()


        policy_gradients = self.policy_model.calculate_gradients(actions, states, advantages, 0.01)
        value_gradients = self.value_model.calculate_gradients(states, returns)

        # print('nice work!')

        # for i, (grad, var) in enumerate(zip(policy_gradients, parent_policy_model.model.trainable_variables)):
        #     print(f"Gradient {i} shape: {grad.shape}, Variable {i} shape: {var.shape}")

        # exit()
        with self.param_lock:
            parent_policy_model.optimizer.apply_gradients(zip(policy_gradients, parent_policy_model.model.trainable_variables))
            parent_value_model.optimizer.apply_gradients(zip(value_gradients, parent_value_model.model.trainable_variables))
            
        # print(f"Worker {self.id_} parent param update successfully!")

    
    def play_episode(self, update_period_steps, parent_policy_model, parent_value_model):
        current_step = 0
        done = False

        n_step_state = []
        n_step_action = []
        n_step_reward = []
        n_step_done = []

        obs, info = self.env.reset()
        rewards_at_terminal = 0

        obs_transformed = image_transformer(obs, [84,84])
        # we stack 4 inital frame as our state // as shape (H x W x 4)
        obs = np.stack([obs_transformed] * 4, axis=2)
        
        self.copy_param_from_parent(parent_policy_model, parent_value_model)

        while not done:
            pi_obs = np.expand_dims(obs, axis=0).astype(np.float32)
            pi_eval = self.policy_model.predict(pi_obs)
            # print(pi_eval[0])
            action = np.random.choice(self.env.action_space.n, p=pi_eval[0].numpy())
            # print("Sum of pi_eval[0]:", np.sum(pi_eval[0]))
            #TODO: MAKE SURE TO REMOVE UNIFORM SAMPLEING
            # action = self.env.action_space.sample()
            n_step_state.append(obs)
            n_step_action.append(action)

            observation, reward, done, truncated, info = self.env.step(action)
            n_step_reward.append(reward)
            n_step_done.append(done)

            current_step += 1
            global_step = next(self.global_counter)
            if done or current_step > update_period_steps:
                # we want to clauclate the gradient and update the main network
                # print('update main')

                self.update_parnet_param(n_step_state, n_step_action, n_step_reward, n_step_done, parent_policy_model, parent_value_model)

                n_step_state = []
                n_step_action = []
                n_step_reward = []

                current_step = 0

                self.copy_param_from_parent(parent_policy_model, parent_value_model)

            rewards_at_terminal += reward
            obs_transformed = image_transformer(observation, [84, 84])
            obs = np.concatenate([obs[:, :, 1:], np.expand_dims(obs_transformed, axis=2)], axis=2)
        
        print(f"Worker ID:{self.id_}   Reward: {rewards_at_terminal}")
        self.return_list.append(rewards_at_terminal)

        global_episode_count = next(self.episode_counter)

        return global_step, global_episode_count

    def run(self, total_number_of_steps, update_period_steps, parent_policy_model, parent_value_model):
        global_step = 0
        while (total_number_of_steps > global_step):
            global_step, global_episode_count = self.play_episode(update_period_steps, parent_policy_model, parent_value_model)
            print("Global timestep: ", global_step, "     Global episode: ", global_episode_count)
            if global_episode_count % 100 == 0:
                self.save_parent_parameter(parent_policy_model, parent_value_model, f'weights/policy_weights{global_episode_count}.weights.h5', f'weights/value_weights{global_episode_count}.weights.h5')
                print("Weights Successfully saved!")