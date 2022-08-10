import numpy as np

import Settings
from Settings import LogLevel

class AtomicModel:
    def __init__(self, logger, data_manager, architecture):
        self.__logger = logger
        self.__data_manager = data_manager
        self.__name = architecture.name

        model = architecture.value
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.__model = model

    def get_name(self):
        return self.__name

    def train(self):
        self.__logger.print("AtomicModel(train): " + self.__name, LogLevel.INFO)
        train_x, train_y = self.__data_manager.get_training_data()
        test_x, test_y = self.__data_manager.get_test_data()

        train_verbosity = True if Settings.LogLevel == 0 else False
        training_history = self.__model.fit(train_x, train_y, batch_size=Settings.BATCH_SIZE, epochs=Settings.EPOCHS, validation_data=(test_x, test_y), verbose=train_verbosity)

        self.__logger.log_training(self.__name, training_history)
        self.__logger.visualize_training(self.__name, training_history)
        
    def predict(self, data):
        self.__logger.print("AtomicModel(predict): " + self.__name, LogLevel.INFO)
        return self.__model.predict(data, batch_size=Settings.BATCH_SIZE)

    def evaluate(self):
        self.__logger.print("AtomicModel(evaluate): " + self.__name, LogLevel.INFO)
        test_x_batches, test_y_batches = self.__data_manager.get_batch_test_data()

        miss_count = 0
        for data_batch in range(len(test_y_batches)):
            test_x_batch = test_x_batches[data_batch]

            actual_y = np.argmax(test_y_batches[data_batch], axis=1)
            predict_y = np.argmax(self.predict(test_x_batch), axis=1)

            miss_count += np.count_nonzero(predict_y != actual_y)

        val_accuracy = 1 - (miss_count / self.__data_manager.get_test_datasize())
        self.__logger.log_evaluation(self.__name, val_accuracy)

