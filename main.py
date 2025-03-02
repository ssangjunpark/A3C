import tensorflow as tf
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import threading
import multiprocessing
import itertools
from datetime import datetime

from worker import Worker

from util import create_networks, image_transformer, smooth


gym.register_envs(ale_py)


TOTAL_NUMBER_OF_STEPS = 1000000000
UPDATE_PERIOD_STEPS = 5

MODEL_SIZE_INITIALIZAER = [[(16, 8, 4), (32, 4, 2)], [256], [], []]


def instantiate_workers(thread_safe_global_counter, thread_safe_global_episode_counter, returns_per_episode, num_workers=-1):
    workers = []
    if (num_workers == -1) or (num_workers > multiprocessing.cpu_count()):
        num_workers = multiprocessing.cpu_count()

    param_lock = threading.Lock()
    
    for id in range(num_workers):
        worker = Worker(id, "ALE/Breakout-v5", MODEL_SIZE_INITIALIZAER, thread_safe_global_counter, thread_safe_global_episode_counter, returns_per_episode, param_lock)
        workers.append(worker)
        print("Worker ", {id}, " created!")

    return workers

def main():
    t_start = datetime.now()
    lol = gym.make("ALE/Breakout-v5", render_mode='rgb_array') # bad design fix it one day :(
    obs, _ = lol.reset()
    obs = image_transformer(obs, [84, 84])
    obs = np.stack([obs] * 4, axis=2)
    parent_policy_model, parent_value_model = create_networks(lol.action_space.n, MODEL_SIZE_INITIALIZAER[0], MODEL_SIZE_INITIALIZAER[1], MODEL_SIZE_INITIALIZAER[2], MODEL_SIZE_INITIALIZAER[3])
    parent_policy_model.predict(np.expand_dims(obs, axis=0).astype(np.float32))
    parent_value_model.predict(np.expand_dims(obs, axis=0).astype(np.float32))
    lol.close()
    thread_safe_global_counter = itertools.count()
    thread_safe_global_episode_counter = itertools.count()

    returns_per_episode = []

    workers = instantiate_workers(thread_safe_global_counter, thread_safe_global_episode_counter, returns_per_episode, num_workers = -1)

    # print(workers)
    # exit()

    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=worker.run, args=(TOTAL_NUMBER_OF_STEPS, UPDATE_PERIOD_STEPS, parent_policy_model, parent_value_model))
        worker_threads.append(t)
        t.start()

    for t in worker_threads:
        t.join()

    returns_per_episode = np.array(returns_per_episode)
    np.savetxt("returns_per_episode.txt", returns_per_episode)

    plt.plot(returns_per_episode, label='reward per episode')
    plt.plot(smooth(returns_per_episode), label='average over 100 episodes')
    plt.legend()
    plt.savefig('results.png')
    
    print(f"Training Complete for {TOTAL_NUMBER_OF_STEPS} time steps")
    print(datetime.now() - t_start)


    # print(returns_per_episode)

    # worker = Worker(1, "ALE/Breakout-v5", MODEL_SIZE_INITIALIZAER, thread_safe_global_counter, returns_per_episode)
    # worker.run(TOTAL_NUMBER_OF_STEPS, UPDATE_PERIOD_STEPS, parent_policy_model, parent_value_model)


if __name__ == "__main__":
    with tf.device('/GPU:0'):
        main()