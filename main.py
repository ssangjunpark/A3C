import tensorflow as tf
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import threading
import multiprocessing
import itertools

from worker import Worker

from util import create_networks, image_transformer


gym.register_envs(ale_py)


TOTAL_NUMBER_OF_STEPS = 100000000
UPDATE_PERIOD_STEPS = 5

MODEL_SIZE_INITIALIZAER = [[(16, 8, 4), (32, 4, 2)], [256], [], []]


def instantiate_workers(thread_safe_global_counter, returns_per_episode, num_workers=-1):
    workers = []
    if (num_workers == -1) or (num_workers > multiprocessing.cpu_count()):
        num_workers = multiprocessing.cpu_count()
    
    for id in range(num_workers):
        worker = Worker(id, "ALE/Breakout-v5", MODEL_SIZE_INITIALIZAER, thread_safe_global_counter, returns_per_episode)
        workers.append(worker)

    return workers

def main():
    lol = gym.make("ALE/Breakout-v5", render_mode='rgb_array') # bad design fix it one day :(
    obs, _ = lol.reset()
    obs = image_transformer(obs, [84, 84])
    obs = np.stack([obs] * 4, axis=2)
    parent_policy_model, parent_value_model = create_networks(lol.action_space.n, MODEL_SIZE_INITIALIZAER[0], MODEL_SIZE_INITIALIZAER[1], MODEL_SIZE_INITIALIZAER[2], MODEL_SIZE_INITIALIZAER[3])
    parent_policy_model.predict(np.expand_dims(obs, axis=0).astype(np.float32))
    parent_value_model.predict(np.expand_dims(obs, axis=0).astype(np.float32))
    lol.close()
    thread_safe_global_counter = itertools.count()

    returns_per_episode = []

    workers = instantiate_workers(thread_safe_global_counter, returns_per_episode)

    # print(workers)
    # exit()

    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=worker.run, args=(TOTAL_NUMBER_OF_STEPS, UPDATE_PERIOD_STEPS, parent_policy_model, parent_value_model))
        worker_threads.append(t)
        t.start()

    for t in worker_threads:
        t.join()


    # lock = threading.Lock()
    
    # #start worker threads later lol i dont want to do this right now
    # worker_threads = []
    # for worker in workers:
    #     worker_function = lambda: worker.run(TOTAL_NUMBER_OF_STEPS, UPDATE_PERIOD_STEPS, parent_policy_model, parent_value_model)
    #     t = threading.Thread(target=worker_function)
    #     t.start()
    #     worker_threads.append(t)


    # for t in worker_threads:
    #     t.join()

    # print(returns_per_episode)

    # worker = Worker(1, "ALE/Breakout-v5", MODEL_SIZE_INITIALIZAER, thread_safe_global_counter, returns_per_episode)
    # worker.run(TOTAL_NUMBER_OF_STEPS, UPDATE_PERIOD_STEPS, parent_policy_model, parent_value_model)



if __name__ == "__main__":
    with tf.device('/gpu:0'):
        main()