import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
from datetime import datetime

gym.register_envs(ale_py)

def play_one_episode(env, episode_number):    
    # our initial state s(0)
    obs, _ = env.reset()

    # we are going to keep track of reward by the end of the episode
    rewards = 0
    
    #wait until our episode terminates
    done = False

    # repeat until our episode is over (and collect our trajectory for update)
    while not done:
        t_start = datetime.now()
        #we are going to sample random action a(t)
        action = env.action_space.sample()

        # take our action to get s(t+1), r(t+1), done flag
        obs, reward, done, _, _ = env.step(action)

        #this is how the observation looks like!
        # print(obs)

        # we increment the reward
        rewards += reward

    print("Episode: ", episode_number, " Rewards: ", rewards, " Total time taken: ", (datetime.now() - t_start))


if __name__ == "__main__":
    #instantiate our environment and record video. Each episode will be recorded and saved under the file test_video_recording
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    env = RecordVideo(env, 'test_video_recording', lambda x : True)

    # if you have pygame installed, comment out line 38+39 and uncomment line 42. This will give you a live view of environment
    # we are going to play 10 episodes
    for episode_number in range(10):
        play_one_episode(env, episode_number + 1)
        

    env.close()
