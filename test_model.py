import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from util import create_networks, image_transformer
import numpy as np
import ale_py

gym.register_envs(ale_py)
MODEL_SIZE_INITIALIZAER = [[(16, 8, 4), (32, 4, 2)], [256], [], []]


def play_one(env, policy_model):
    done = False
    rewards = 0

    obs, _ = env.reset()
    obs_transformed = image_transformer(obs, [84,84])
    obs = np.stack([obs_transformed] * 4, axis=2)

    while not done:
        pi_obs = np.expand_dims(obs, axis=0).astype(np.float32)
        pi_eval = policy_model.predict(pi_obs)
        action = np.random.choice(env.action_space.n, p=pi_eval[0].numpy())

        observation, reward, done, _, _ = env.step(action)
        rewards += reward
        obs_transformed = image_transformer(observation, [84, 84])
        obs_transformed = np.expand_dims(obs_transformed, axis=2)
        obs = np.concatenate([obs[:, :, 1:], obs_transformed], axis=2)


    print("Total rewards: ", rewards)

def test_model():
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    env = RecordVideo(env, video_folder='recordings', episode_trigger= lambda x : True)

    policy_model, _ = create_networks(env.action_space.n, MODEL_SIZE_INITIALIZAER[0], MODEL_SIZE_INITIALIZAER[1], MODEL_SIZE_INITIALIZAER[2], MODEL_SIZE_INITIALIZAER[3])
    
    policy_model.model.load_weights('weights/policy_weights4300.weights.h5')

    play_one(env, policy_model)
    
    env.close()
    
if __name__ == "__main__":
    test_model()