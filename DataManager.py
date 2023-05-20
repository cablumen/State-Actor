import random
from collections import deque
import numpy as np

import Settings


class DataManager:
    def __init__(self):
        # [[state, action_index, predicted_next_state, next_state, reward, done], ...]
        self.replay_data = deque(maxlen=10000)

    #       action data
    def get_actor_data(self):
        return self.__get_random_actor_data()

    def is_actor_training_ready(self):
        return len(self.replay_data) >= Settings.BATCH_SIZE * 2

    #       critic data
    def get_critic_data(self):
        return self.__get_random_critic_data()

    def is_critic_training_ready(self):
        return len(self.replay_data) >= Settings.BATCH_SIZE

    #       other
    def put_replay_data(self, step_record):
        return self.replay_data.append(step_record)

    def reset(self):
        self.replay_data.clear()

    #       private functions
    def __get_random_actor_data(self):
        train_x = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE+1), dtype=np.float32)
        train_y = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)
        test_x = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE+1), dtype=np.float32)
        test_y = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)

        # randomly generate action indicies to sample
        random_indicies = random.sample(range(0, len(self.replay_data)), Settings.BATCH_SIZE * 2)
        sampled_training_indicies = random_indicies[0:Settings.BATCH_SIZE]
        sampled_validation_indicies = random_indicies[Settings.BATCH_SIZE:]

        # populate test sample
        batch_index = 0
        for random_index in sampled_training_indicies:
            record = self.replay_data[random_index]
            train_x[batch_index] = np.insert(record[0], 0, record[1], axis=1)
            train_y[batch_index] = record[3] - record[0]
            batch_index += 1

        # populate validation sample
        batch_index = 0
        for random_index in sampled_validation_indicies:
            record = self.replay_data[random_index]
            test_x[batch_index] = np.insert(record[0], 0, record[1], axis=1)
            test_y[batch_index] = record[3] - record[0]
            batch_index += 1

        return train_x, train_y, test_x, test_y

    def __get_random_critic_data(self):
        states = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)
        next_states = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)
        rewards = np.empty(Settings.BATCH_SIZE, dtype=float)
        dones = np.empty(Settings.BATCH_SIZE, dtype=np.int32)

        random_indicies = random.sample(range(0, len(self.replay_data)), Settings.BATCH_SIZE)

        # populate test sample
        batch_index = 0
        for random_index in random_indicies:
            action_data = self.replay_data[random_index]

            states[batch_index] = action_data[0]
            next_states[batch_index] = action_data[3]
            rewards[batch_index] = action_data[4]
            dones[batch_index] = action_data[5]
            batch_index += 1

        return states, next_states, rewards, dones
