import os
import time
import gym
import numpy as np

from ActionDictionary import ActionDictionary
from Architectures import Architectures
import Settings


class Controller:
    def __init__(self):
        # self.env = gym.make("CartPole-v1", render_mode='human')
        self.env = gym.make("CartPole-v1")

        run_folder = self.get_run_folder()
        experiments = [(Architectures.FC_5_32, Architectures.REWARD_3_64)] # format: (sub_model_architecture, reward_architecture)

        self.run_experiments(run_folder, experiments)

    def get_run_folder(self):
        # get current directory path
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # create sub-folder for logs
        log_folder = os.path.join(dir_path, "logs")
        if not os.path.isdir(log_folder):
            os.mkdir(log_folder)

        # create sub-folder for specific run
        run_folder = os.path.join(log_folder, str(time.strftime("%Y-%m-%d_%H-%M-%S")))
        if not os.path.isdir(run_folder):
            os.mkdir(run_folder)

        return run_folder

    def run_experiments(self, log_folder, experiments):
        for sub_model_architecture, reward_architecture in experiments:
            action_dictionary = ActionDictionary(log_folder, sub_model_architecture, reward_architecture)
            for action_dictionary.logger.session in range(Settings.SESSION_COUNT):
                self.epsilon_exploration(action_dictionary)
                action_dictionary.reset()

    # execute best predicted action with increasing probability
    def epsilon_exploration(self, action_dictionary):
        print("\nController(epsilon_exploration): submodel:" + action_dictionary.get_actor_name() + " reward:" + action_dictionary.get_critic_name())
        epsilon = Settings.EPSILON_START

        rewards_list = []
        for action_dictionary.logger.episode in range(Settings.EPISODE_COUNT):
            state = self.env.reset()[0]
            state = np.reshape(state, [1, Settings.OBSERVATION_SIZE])
            reward_for_episode = 0
            for action_dictionary.logger.step in range(Settings.MAX_TRAINING_STEPS):
                if np.random.rand() < epsilon:
                    action_index, predicted_next_state = action_dictionary.predict_random_action(state)
                else:
                    action_index, predicted_next_state = action_dictionary.predict_optimal_action(state)

                next_state, reward,  done, info, _ = self.env.step(action_index)
                action_dictionary.put_record_data([state, action_index, predicted_next_state, next_state, reward, done])

                next_state = np.reshape(next_state, [1, Settings.OBSERVATION_SIZE])
                reward_for_episode += reward
                state = next_state

                action_dictionary.train_models()

                if done:
                    break

            # reduce probability of random action
            if epsilon > Settings.EPSILON_MIN:
                epsilon *= Settings.EPSILON_DECAY

            action_dictionary.evaluate_models()
            rewards_list.append(reward_for_episode)
            last_rewards_mean = np.mean(rewards_list[-30:])
            print(f"\tEpisode: {action_dictionary.logger.episode:3.0f} || Reward: {reward_for_episode:3.0f} || Average Reward: {last_rewards_mean:3.4f}")

        action_dictionary.logger.log_session(rewards_list)

if __name__ == '__main__':
    Controller()
