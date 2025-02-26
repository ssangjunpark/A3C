import tensorflow as tf
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import threading
import multiprocessing
import itertools

from policy_model import policy_model
from value_model import value_model


gym.register_envs(ale_py)


TOTAL_NUMBER_OF_STEPS = 1e5
UPDATE_PERIOD_STEPS = 5


def instantiate_workers(num_workers=-1):
    workers = list()
    if (num_workers == -1) or (num_workers > multiprocessing.cpu_count()):
        num_workers = multiprocessing.cpu_count()
    
    for id in (num_workers):
        pass
    
    return workers

def main():
    thread_safe_global_counter = itertools.count()

    returns = list()

    workers = instantiate_workers()

    #start worker threads


if __name__ == "__main__":
    main()