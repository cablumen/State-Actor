import csv
import matplotlib.pyplot as plt
import os
import time

import Settings

class Logger:
    def __init__(self, logging_enabled=True):
        self.__logging_enabled = logging_enabled

        # get current directory path
        dir_path = os.path.dirname(os.path.abspath(__file__))
        
        # create sub-folder for all experiment graphs
        experiment_path = os.path.join(dir_path, "experiment records")
        if not os.path.isdir(experiment_path):
            os.mkdir(experiment_path)

        # create sub-folder for specific experiment run
        self.__log_folder = os.path.join(experiment_path, str(time.strftime("%Y-%m-%d_%H-%M-%S")))
        if not os.path.isdir(self.__log_folder):
            os.mkdir(self.__log_folder)

        evaluation_path = os.path.join(self.__log_folder, "model accuracy.txt")
        self.__accuracy_file = open(evaluation_path, "w")

        self.__set_plt_params()

    def print(self, string, log_level = 0):
        if self.__logging_enabled and log_level.value >= Settings.LOG_LEVEL:
            print(string)

    def log_training(self, model_name, training_history):
        # create csv writer
        training_path = os.path.join(self.__log_folder, model_name + '_log.csv')
        training_file = open(training_path, 'w', newline='')
        training_csv = csv.writer(training_file)

        # AtomicModel logs
        if type(training_history) is not dict:
            # write column headers
            training_csv.writerow(["Epoch", "TA", "VA"])

            # write training history
            for epoch in range(len(training_history.epoch)):
                epoch_accuracy = training_history.history["accuracy"][epoch]
                epoch_val_accuracy = training_history.history["val_accuracy"][epoch]
                training_csv.writerow([epoch, epoch_accuracy, epoch_val_accuracy])

        # CompositeModel logs
        else:
            # write column headers
            column_headers = ["Epoch"]
            for digit_index in range(10):
                column_headers.extend(["Model " + str(digit_index) + " TA", "Model " + str(digit_index) + " VA"])
            training_csv.writerow(column_headers)

            # write training history
            for epoch in range(len(list(training_history.values())[0].epoch)):
                record_row = [None] * 21
                record_row[0] = epoch
                for digit, digit_history in training_history.items():
                    record_row[1 + (digit * 2)] = digit_history.history["accuracy"][epoch]
                    record_row[1 + (digit * 2) + 1] = digit_history.history["val_accuracy"][epoch]

                training_csv.writerow(record_row)

    def log_evaluation(self, model_name, val_accuracy):
        self.__accuracy_file.write(model_name + ": " + str(val_accuracy) +"\n")

    def visualize_training(self, model_name, training_history):
        # AtomicModel logs
        if type(training_history) is not dict:
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
            fig_path = os.path.join(self.__log_folder, model_name + " training.png")
            plt.savefig(fig_path)

        # CompositeModel logs
        else:
            epoch = list(training_history.values())[0].epoch

            plt.figure()
            plt.title(model_name + " accuracy history")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            for digit, digit_history in training_history.items():
                digit_accuracy = digit_history.history["accuracy"]
                plt.plot(epoch, digit_accuracy, label="sub-model " + str(digit))

            plt.legend(loc="center right")
            fig_path = os.path.join(self.__log_folder, model_name + " accuracy.png")
            plt.savefig(fig_path)

            plt.figure()
            plt.title(model_name + " validation accuracy history")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            for digit, digit_history in training_history.items():
                digit_val_accuracy = digit_history.history["val_accuracy"]
                plt.plot(epoch, digit_val_accuracy, label="sub-model " + str(digit))

            plt.legend(loc="center right")
            fig_path = os.path.join(self.__log_folder, model_name + " val accuracy.png")
            plt.savefig(fig_path)

    def log_misses(self, model_name, misses):
        # create csv writer
        miss_path = os.path.join(self.__log_folder, model_name + '_misses.csv')
        miss_file = open(miss_path, 'w', newline='')
        miss_csv = csv.writer(miss_file)

        miss_csv.writerow(["Miss Difference"])
        for miss in misses:
            miss_csv.writerow([miss])

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