import numpy as np
import tensorflow as tf

import Settings
from Settings import LogLevel

class CompositeModel:
    def __init__(self, logger, data_manager, architecture, digits=range(10)):
        self.__logger = logger
        self.__data_manager = data_manager
        self.__name = architecture.name
        sub_model = architecture.value
        
        #       Initialize sub-models
        # {digit -> TF model}
        self.__sub_models = {}
        for digit in digits:
            digit_model = tf.keras.models.clone_model(sub_model)
            
            # binary crossentropy assumes y_pred in [0, 1]
            digit_model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
            self.__sub_models[digit] = digit_model

    def train(self):
        self.__logger.print("CompositeModel(train): " + self.__name, LogLevel.INFO)
        training_history = {}
        train_verbosity = True if Settings.LogLevel == 0 else False
        for digit, digit_model in self.__sub_models.items():
            self.__logger.print("CompositeModel:\ttraining sub-model " + str(digit), LogLevel.INFO)
            digit_train_x, digit_train_y = self.__data_manager.get_digit_training_data(digit)
            digit_test_x, digit_test_y = self.__data_manager.get_digit_test_data(digit)
            
            history = digit_model.fit(digit_train_x, digit_train_y, batch_size=Settings.BATCH_SIZE, epochs=Settings.EPOCHS, validation_data=(digit_test_x, digit_test_y), verbose=train_verbosity)
            training_history[digit] = history

        self.__logger.log_training(self.__name, training_history)
        self.__logger.visualize_training(self.__name, training_history)
        
    def predict(self, data):
        self.__logger.print("CompositeModel(predict): " + self.__name, LogLevel.INFO)
        prediction_size = len(data)
        predictions = np.zeros((prediction_size, 10), dtype=np.float32)
        for digit, digit_model in self.__sub_models.items():
            prediction = digit_model.predict(data, batch_size=prediction_size)
            
            batch_indices = [i for i in range(prediction_size)]
            digit_indices = [digit] * prediction_size
            digit_predictions = [i[0] for i in prediction]
            predictions[[batch_indices], [digit_indices]] = digit_predictions
        return predictions

    def evaluate(self):
        self.__logger.print("CompositeModel(evaluate): " + self.__name, LogLevel.INFO)
        test_x_batches, test_y_batches = self.__data_manager.get_batch_test_data()

        miss_count = 0
        miss_differences = []
        for data_batch in range(len(test_y_batches)):
            test_x_batch = test_x_batches[data_batch]

            actual_y = np.argmax(test_y_batches[data_batch], axis=1)
            predictions = self.predict(test_x_batch)
            predict_y = np.argmax(predictions, axis=1)

            for index, (actual, prediction) in enumerate(zip(actual_y, predict_y)):
                if actual != prediction:
                    miss_differences.append(predictions[index][prediction] - predictions[index][actual])
                    miss_count += 1

        self.__logger.log_misses(self.__name, miss_differences)

        val_accuracy = 1 - (miss_count / self.__data_manager.get_test_datasize())
        self.__logger.log_evaluation(self.__name, val_accuracy)
