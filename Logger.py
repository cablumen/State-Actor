import csv
import os
import matplotlib.pyplot as plt

import Settings

class Logger:
    def __init__(self, architecture_folder, logging_enabled=True):
        self.architecture_folder = architecture_folder
        self.logging_enabled = logging_enabled

        # create sub-folders for architecture data
        self.session_path = None

        self.reward_episode_writer = None
        self.reward_session_writer = None

        self.actor_train_writer = None
        self.actor_evaluate_writer = None

        self.__set_plt_params()

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self.session_path = os.path.join(self.architecture_folder, "Session " + str(value))
        if not os.path.isdir(self.session_path):
            os.mkdir(self.session_path)

        if self.logging_enabled and Settings.LOG_TRAINING:
            actor_training_path = os.path.join(self.session_path, 'training.csv')
            training_file = open(actor_training_path, 'w', newline='')
            self.actor_train_writer = csv.writer(training_file)
            self.actor_train_writer.writerow(["Epoch", "Training MSE", "Validation MSE"])

        if self.logging_enabled and Settings.LOG_EVALUATION:
            actor_evaluation_path = os.path.join(self.session_path, 'evaluation.csv')
            evaluation_file = open(actor_evaluation_path, 'w', newline='')
            self.actor_evaluate_writer = csv.writer(evaluation_file)
            self.actor_evaluate_writer.writerow(["Step", "Evaluation MSE"])

        if self.logging_enabled and Settings.LOG_SESSION_REWARD:
            session_reward_evaluation_path = os.path.join(self.session_path, 'reward.csv')
            session_reward_file = open(session_reward_evaluation_path, 'w', newline='')
            self.reward_session_writer = csv.writer(session_reward_file)
            self.reward_session_writer.writerow(["Episode", "Cumulative Reward"])

        self._session = value

    @property
    def episode(self):
        return self._episode

    @episode.setter
    def episode(self, value):
        if self.logging_enabled and Settings.LOG_EPISODE_REWARD:
            reward_path = os.path.join(self.session_path, "episode_" + str(value))
            reward_file = open(reward_path, 'w', newline='')
            self.reward_episode_writer = csv.writer(reward_file)
            self.reward_episode_writer.writerow(["Step", "Reward"])

        self._episode = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def log_step_record(self, step_record):
        if self.logging_enabled and Settings.LOG_EPISODE_REWARD:
            self.reward_episode_writer.writerow([self.step, step_record[4]])

    def print(self, string, log_level = 0):
        if self.logging_enabled and log_level.value >= Settings.LOG_LEVEL:
            print(string)

    def log_training(self, training_history):
        if self.logging_enabled and Settings.LOG_TRAINING:
            for epoch in range(len(training_history.epoch)):
                epoch_mse = training_history.history["mse"][epoch]
                epoch_val_mse = training_history.history["val_mse"][epoch]
                self.actor_train_writer.writerow([epoch, epoch_mse, epoch_val_mse])

    def log_evaluation(self, mse):
        if self.logging_enabled and Settings.LOG_EVALUATION:
            self.actor_evaluate_writer.writerow([self.episode, mse])

    def log_session(self, reward_history):
        if self.logging_enabled and Settings.LOG_SESSION_REWARD:
            for episode_index in range(1, len(reward_history)):
                episode_reward = reward_history[episode_index]
                self.reward_session_writer.writerow([episode_index, episode_reward])

    def visualize_training(self, model_name, training_history):
        # AtomicModel logs
        plt.figure()
        plt.title(model_name + " training history")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        epoch = training_history.epoch
        accuracy = training_history.history["accuracy"]
        val_accuracy = training_history.history["val_accuracy"]

        plt.plot(epoch, accuracy, "-r", label="Accuracy")
        plt.plot(epoch, val_accuracy, "-b", label="Validation Accuracy")

        plt.legend(loc="center right")
        # fig_path = os.path.join(self.log_folder, model_name + " training.png")
        # plt.savefig(fig_path)

    def __set_plt_params(self):
        background_color = "#1E1E1E"
        foreground_color = "#DBDBDB"
        plt.rcParams["figure.facecolor"] = background_color
        plt.rcParams["figure.edgecolor"] = background_color

        plt.rcParams["axes.facecolor"] = background_color
        plt.rcParams["axes.edgecolor"] = foreground_color
        plt.rcParams["axes.titlecolor"] = foreground_color
        plt.rcParams["axes.labelcolor"] = foreground_color

        plt.rcParams["xtick.color"] = foreground_color
        plt.rcParams["xtick.labelcolor"] = foreground_color
        plt.rcParams["ytick.color"] = foreground_color
        plt.rcParams["ytick.labelcolor"] = foreground_color

        plt.rcParams["legend.labelcolor"] = foreground_color
