import tensorflow as tf
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import threading
import multiprocessing
import itertools

from worker import Worker

from util import create_networks


gym.register_envs(ale_py)


TOTAL_NUMBER_OF_STEPS = 30
UPDATE_PERIOD_STEPS = 5

MODEL_SIZE_INITIALIZAER = [[(32, 8, 4), (64, 4, 2), (64, 3, 1), (128, 3, 1)], [512], [], []]


def instantiate_workers(num_workers=-1):
    workers = list()
    if (num_workers == -1) or (num_workers > multiprocessing.cpu_count()):
        num_workers = multiprocessing.cpu_count()
    
    for id in range(num_workers):
        pass
    
    return workers

def main():
    lol = gym.make("ALE/Breakout-v5", render_mode='rgb_array') # bad design fix it one day :(
    
    parent_policy_model, parent_value_model = create_networks(lol.action_space.n, MODEL_SIZE_INITIALIZAER[0], MODEL_SIZE_INITIALIZAER[1], MODEL_SIZE_INITIALIZAER[2], MODEL_SIZE_INITIALIZAER[3])
    thread_safe_global_counter = itertools.count()

    returns = list()

    workers = instantiate_workers()

    #start worker threads later lol i dont want to do this right now

    worker = Worker("ALE/Breakout-v5", MODEL_SIZE_INITIALIZAER, thread_safe_global_counter)
    worker.run(TOTAL_NUMBER_OF_STEPS, UPDATE_PERIOD_STEPS)



if __name__ == "__main__":
    main()