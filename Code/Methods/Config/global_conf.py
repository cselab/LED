#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import sys
import socket
import os

hostname = socket.gethostname()
global_params = lambda: 0
global_params.cluster = \
(hostname[:len('local')]=='local') * 'local' + \
(hostname[:len('eu')]=='eu') * 'euler' + \
(hostname[:len('daint')]=='daint') * 'daint' + \
(hostname[:len('barry')]=='barry') * 'barry' + \
(hostname[:len('barrycontainer')]=='barrycontainer') * 'barry' + \
(hostname[:len('nid')]=='nid') * 'daint'

HIPPO = False
if global_params.cluster == 'euler':
    print("[Config] RUNNING IN EULER CLUSTER.")
    SCRATCH = os.environ['SCRATCH']
    project_path = SCRATCH + "/LED/Code"
    global_params.scratch_path = SCRATCH

elif global_params.cluster == 'barry':
    print("[Config] RUNNING IN BARRY CLUSTER.")
    # SCRATCH = os.environ['SCRATCH']
    SCRATCH = "/scratch/pvlachas"
    project_path = SCRATCH + "/LED/Code"
    global_params.scratch_path = SCRATCH

elif global_params.cluster == 'daint':
    print("[Config] RUNNING IN DAINT CLUSTER.")
    SCRATCH = os.environ['SCRATCH']
    project_path = SCRATCH + "/LED/Code"
    global_params.scratch_path = SCRATCH

    # PROJECTFOLDER = "/project/s929/pvlachas"
    # project_path = PROJECTFOLDER + "/LED/Code"

elif global_params.cluster == 'local':
    # Running in the local repository, pick whether you are loading data-sets from the hippo database, or using a local data folder.
    if HIPPO:
        print("[Config] DATA LOADING FROM HIPPO.")
        HOME = os.environ['HOME']
        project_path = HOME + "/hippo/LED/Code"
        global_params.scratch_path = HOME

    else:
        HOME = os.environ['HOME']
        global_params.scratch_path = HOME

        print("[Config] RUNNING IN LOCAL REPOSITORY.")
        config_path = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(os.path.dirname(config_path))

else:
    # IF NOTHING FROM THE ABOVE, RESORT TO LOCAL
    print("[Config] RUNNING IN LOCAL REPOSITORY (hostname {:} not resolved).".
          format(hostname))
    # raise ValueError("Avoid running in local repository.")

    global_params.scratch_path = os.environ['HOME']

    if HIPPO:
        print("[Config] DATA LOADING FROM HIPPO.")
        HOME = os.environ['HOME']
        project_path = HOME + "/hippo/LED/Code"
    else:
        print("[Config] RUNNING IN LOCAL REPOSITORY.")
        config_path = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(os.path.dirname(config_path))

global_params.home_path = os.environ['HOME']

print("[Config] HOME PATH = {:}".format(global_params.home_path))
print("[Config] SCRATCH PATH = {:}".format(global_params.scratch_path))

# hostname = socket.gethostname()
# global_params = lambda: 0

# config_path = os.path.dirname(os.path.abspath(__file__))
# project_path = os.path.dirname(os.path.dirname(config_path))

print("[Config] PROJECT PATH = {:}".format(project_path))

global_params.global_utils_path = "./Models/Utils"

global_params.saving_path = project_path + "/Results/{:s}"
global_params.project_path = project_path

global_params.data_path_train = project_path + "/Data/{:s}/Data/train"
global_params.data_path_test = project_path + "/Data/{:s}/Data/test"
global_params.data_path_val = project_path + "/Data/{:s}/Data/val"
global_params.data_path_gen = project_path + "/Data/{:s}/Data"

# PATH TO LOAD THE PYTHON MODELS
global_params.py_models_path = "./Models/{:}"

# PATHS FOR SAVING RESULTS OF THE RUN
global_params.model_dir = "/Trained_Models/"
global_params.fig_dir = "/Figures/"
global_params.results_dir = "/Evaluation_Data/"
global_params.logfile_dir = "/Logfiles/"
