import csv
import os
import numpy as np


# get path to record folder
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
file_names = os.listdir(cur_dir_path)
print("Select directory from list")
for name in file_names:
    if os.path.isdir(name):
        print("\t" + name)

# record_dir_name = input()
record_dir_name = "Reward Architecture Testing"
record_dir_path = os.path.join(cur_dir_path, record_dir_name)

# get path to each run folder
run_paths = []
run_names = os.listdir(record_dir_path)
for name in run_names:
    run_path = os.path.join(record_dir_path, name)
    if os.path.isdir(run_path):
        run_paths.append(run_path)

for run_path in run_paths:
    model_name = os.path.basename(run_path)

    # get path to each session folder
    session_paths = []
    session_names = os.listdir(run_path)
    for name in session_names:
        session_path = os.path.join(run_path, name)
        if os.path.isdir(session_path):
            session_paths.append(session_path)

    print(str(model_name) + ": detected " + str(len(session_paths)) + " session runs")

    run_evaluations = []
    for session_path in session_paths:
        evaluation_reader = csv.reader(open(os.path.join(session_path, "reward.csv")))
        next(evaluation_reader)

        session_evaluations = []
        session_evaluations_mean = []
        for row in evaluation_reader:
            evaluation_mse = float(row[1])
            session_evaluations.append(evaluation_mse)
            session_evaluations_mean.append(np.mean(session_evaluations[-30:]))

        run_evaluations.append(session_evaluations_mean[-1])

    run_mean = np.mean(run_evaluations)
    print("\trun_mean=" + str(run_mean))
    print("")
