import numpy as np
import tensorflow as tf
import random

import Settings

class DataManager:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.__x_train, y_train), (self.__x_test, y_test) = mnist.load_data()
    
        # convert to "one-hot" vectors using the to_categorical function
        self.__y_train = tf.keras.utils.to_categorical(y_train, Settings.LABELS, dtype='uint8')
        self.__y_test = tf.keras.utils.to_categorical(y_test, Settings.LABELS, dtype='uint8')

        # create datasets for digit sub-models
        # {digit -> ([y_train], [y_test])
        self.__digit_data = {}
        for digit_index in range(Settings.LABELS):
            digit_y_train = np.array([[i[digit_index]] for i in self.__y_train])
            digit_y_test = np.array([[i[digit_index]] for i in self.__y_test])
            self.__digit_data[digit_index] = (digit_y_train, digit_y_test)

    #       get training data
    def get_training_data(self):
        return self.__x_train, self.__y_train

    def get_training_datasize(self):
        return len(self.__y_train)

    def get_batch_training_data(self):
        return self.__get_batched_data(self.__x_train, self.__y_train)

    def get_random_training_data(self):
        return self.__get_random_data(self.__x_train, self.__y_train)
        
    #       get test data
    def get_test_data(self):
        return self.__x_test, self.__y_test

    def get_test_datasize(self):
        return len(self.__y_test)

    def get_batch_test_data(self):
        return self.__get_batched_data(self.__x_test, self.__y_test)

    def get_random_test_data(self):
        return self.__get_random_data(self.__x_test, self.__y_test)
    
    #       get digit training data
    def get_digit_training_data(self, digit_index):
        return self.__x_train, self.__digit_data[digit_index][0]

    def get_batch_digit_training_data(self, digit_index):
        return self.__get_batched_data(self.__x_train, self.__digit_data[digit_index][0])
        
    def get_random_digit_training_data(self, digit_index):
        return self.__get_random_data(self.__x_train, self.__digit_data[digit_index][0])


    #       get digit test data
    def get_digit_test_data(self, digit_index):
        return self.__x_test, self.__digit_data[digit_index][1]

    def get_batch_digit_test_data(self, digit_index):
        return self.__get_batched_data(self.__x_test, self.__digit_data[digit_index][1])

    def get_random_digit_test_data(self, digit_index):
        return self.__get_random_data(self.__x_test, self.__digit_data[digit_index][1])
    

    #       private functions
    def __get_random_data(self, x_data, y_data):
        y_datasize = len(y_data)
        assert Settings.BATCH_SIZE <= y_datasize, "DataManager(get_random_data): expected batch size smaller than {y_datasize}, got {Settings.BATCH_SIZE}"

        random_x = np.empty((Settings.BATCH_SIZE, x_data[0].shape[0], x_data[0].shape[1]), dtype=np.uint8)
        random_y = np.empty((Settings.BATCH_SIZE, Settings.LABELS), dtype=np.uint8)

        # randomly generate indicies to sample
        random_indicies = random.sample(range(0, len(y_data)), Settings.BATCH_SIZE)

        # populate test sample
        batch_index = 0
        for data_index in random_indicies:
            random_x[batch_index] = x_data[data_index]
            random_y[batch_index] = y_data[data_index]
            batch_index += 1

        return random_x, random_y

    def __get_batched_data(self, x_data, y_data):
        y_datasize = len(y_data)
        assert Settings.BATCH_SIZE <= y_datasize, "DataManager(get_batched_data): expected batch size smaller than {y_datasize}, got {Settings.BATCH_SIZE}"

        batched_x, batched_y = [], []
        batch_count = int(y_datasize / Settings.BATCH_SIZE)
        if y_datasize % Settings.BATCH_SIZE != 0:
            batch_count += 1
        
        for batch_index in range(batch_count):
            range_start = batch_index * Settings.BATCH_SIZE

            if (batch_index + 1) * Settings.BATCH_SIZE <= y_datasize:
                range_end = (batch_index + 1) * Settings.BATCH_SIZE
            else:
                range_end = y_datasize

            batched_x.append(x_data[range_start:range_end])
            batched_y.append(y_data[range_start:range_end])

        return batched_x, batched_y

if __name__ == '__main__':
    DataManager()