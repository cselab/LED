# module load daint-gpu
# FOR PYTORCH
# module load cray-python/3.8.2.1
# source ${HOME}/venv-python3.8-pytorch1.9/bin/activate

# cd  /users/pvlachas/STF/Code/Methods/

import numpy as np


def find_all(a_str, sub):
    start = 0
    indexes = []
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return indexes
        indexes.append(start)
        start += len(sub)  # use start += 1 to find overlapping matches
    return indexes


filename = "FHN_2_greasy_arnn_train_logfile_JID25279312.txt"
with open(filename, 'r') as myfile:
    data = myfile.read()

substring = "completed successfully on node"
indexes = find_all(data, substring)

completed_tasks = []
for index in indexes:
    temp = data[index - 29:index]
    # print(temp)
    numbers = [int(s) for s in temp.split() if s.isdigit()]
    assert (numbers[0] == numbers[1])
    # print(numbers)
    completed_tasks.append(numbers[0])

NUM_TASKS = 192
all_tasks = np.arange(1, NUM_TASKS + 1, 1).astype(int)
all_tasks = set(all_tasks)
completed_tasks = set(completed_tasks)

print("Number of total tasks: {:}".format(len(all_tasks)))
print("Number of completed tasks: {:}".format(len(completed_tasks)))

remainining_tasks = all_tasks.difference(completed_tasks)

print("Number of Remaining tasks:")
print(len(remainining_tasks))

print("Remaining tasks:")
print(remainining_tasks)

# {104, 180, 92, 165}
