from collections import deque
import csv
import os
import shutil
import statistics


start_index = episode_index - 9
if start_index < 0:
    start_index = 0
reward_smoothing_10 = np.mean(reward_history[start_index:episode_index])

start_index = episode_index - 29
if start_index < 0:
    start_index = 0
reward_smoothing_30 = np.mean(reward_history[start_index:episode_index])

dir_path = os.path.dirname(os.path.abspath(__file__))

record_path = os.path.join(dir_path, 'training records')

# create fresh sub-folder for all training records
aggregation_dir = os.path.join(dir_path, 'record aggregation')
if os.path.isdir(aggregation_dir):
    shutil.rmtree(aggregation_dir)
    os.mkdir(aggregation_dir)
else:
    os.mkdir(aggregation_dir)

# there are 1 csv per action per row 
self.__csv_readers = {}
self.__csv_writers = {}
for sub_directory in os.listdir(record_path):
    directory_path = os.path.join(record_path, sub_directory)
    for run_file_name in os.listdir(directory_path):
        file_name, file_extension = os.path.splitext(run_file_name)
        if file_extension == ".csv":
            action_name = file_name.replace("_log", "")

            if action_name not in self.__csv_readers:
                self.__csv_readers[action_name] = []

            
            if action_name not in self.__csv_writers:
                aggregation_path = os.path.join(aggregation_dir, action_name + '.csv')
                aggregation_file = open(aggregation_path, 'w', newline='')
                aggregation_writer = csv.writer(aggregation_file)
                self.__csv_writers[action_name] = aggregation_writer

            
            run_path = os.path.join(directory_path, run_file_name)
            run_file = open(run_path, 'r', newline='')
            run_reader = csv.reader(run_file)
            self.__csv_readers[action_name].append(run_reader)

for action_name, csv_reader_list in self.__csv_readers.items():
    records_per_action = len(csv_reader_list)
    record_index = 1

    action_data = []

    # add headers for first reader
    header_list = ["Batch"]
    for header_index in range(records_per_action):
        header_list.append("Model_" + str(header_index) + " training loss")
        header_list.append("Model_" + str(header_index) + " average loss")

    action_data.append(header_list)

    for csv_reader in csv_reader_list:
        next(csv_reader)
        row_averages = deque(maxlen=20)
        for row in csv_reader:
            batch = int(row[0])
            training_loss = float(row[1])

            # if a batch number exceeds the current data size, append an empty row
            if batch >= len(action_data):
                action_data.append([None] * (2 * records_per_action  + 1))
                action_data[batch][0] = batch

            row_averages.append(training_loss)
            cur_average = statistics.mean(row_averages)

            action_data[batch][record_index] = training_loss
            action_data[batch][record_index + 1] = cur_average

        record_index += 2

    self.__csv_writers[action_name].writerows(action_data)
