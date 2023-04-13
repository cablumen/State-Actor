import numpy as np
import random
import os

from ActorModel import ActorModel
from DataManager import DataManager
from Logger import Logger
from CriticModel import CriticModel
import Settings


class ActionDictionary:
    def __init__(self, run_folder, sub_model_architecture, critic_architecture):
        architecture_folder = os.path.join(run_folder, sub_model_architecture.name + " " + critic_architecture.name)
        if not os.path.isdir(architecture_folder):
            os.mkdir(architecture_folder)
            
        self.logger = Logger(architecture_folder)
        self.__data_manager = DataManager()

        self.__critic_model = CriticModel(self.logger, self.__data_manager, critic_architecture)
        self.__actor_model = ActorModel(self.logger, self.__data_manager, sub_model_architecture)

        self.__training_step = 0

    def get_critic_name(self):
        return self.__critic_model.name

    def get_actor_name(self):
        return self.__actor_model.name

    def put_record_data(self, step_record):
        self.logger.log_step_record(step_record)
        self.__data_manager.put_replay_data(step_record)

    def predict_random_action(self, state):
        random_action_index = random.randint(0, Settings.ACTION_SIZE - 1)
        state_w_action = np.insert(state, 0, random_action_index, axis=1)
        delta_next_state = self.__actor_model.predict(state_w_action)
        predicted_next_state = state + delta_next_state
        return random_action_index, predicted_next_state

    def predict_optimal_action(self, state):
        predicted_next_states = []
        predicted_rewards = []
        for action_index in range(Settings.ACTION_SIZE):
            state_w_action = np.insert(state, 0, action_index, axis=1)
            delta_next_state = self.__actor_model.predict(state_w_action)
            predicted_next_state = state + delta_next_state
            predicted_next_states.append(predicted_next_state)
            predicted_rewards.append(self.__critic_model.predict(predicted_next_state))

        optimal_action_index = np.argmax(predicted_rewards)
        optimal_next_state = predicted_next_states[optimal_action_index]
        return optimal_action_index, optimal_next_state

    def train_models(self):
        if self.__training_step % 5 == 0:
            self.__critic_model.train()
            self.__actor_model.train() 
            self.__training_step = 0

        self.__training_step += 1

    def evaluate_models(self):
        self.__actor_model.evaluate()

    def reset(self):
        self.__data_manager.reset()
        self.__critic_model.reset()
        self.__actor_model.reset()
            