#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import glob, os
import numpy as np
import argparse
from Utils.utils import *
from tabulate import tabulate
import pathlib
from pathlib import Path

# ADDING PARENT DIRECTORY TO PATH
import os, sys, inspect

from pathlib import Path

import socket
hostname=socket.gethostname()
print(hostname)

sys.path.append('../../Methods')

from Config.global_conf import global_params

Experiment_Name = "Experiment_Daint_Large"
# Experiment_Name=None

for system_name in [
    "FHN",
    # "KSGP64L22",
    # "KSGP64L22Large",
    # "cylRe100",
    # "cylRe1000",
    # "cylRe100HR",
    # "cylRe1000HR",
    # "cylRe100HRDt005",
    # "cylRe1000HRDt005",
]:

    DIR_NAME = "./{:}/Postprocessing_Logs".format(system_name)
    os.makedirs(DIR_NAME, exist_ok=True)

    saving_path = getSavingPath(Experiment_Name, system_name, global_params)
    logfile_path = saving_path + global_params.logfile_dir

    print(system_name)
    print(logfile_path)

    # FIELDS = [
    #     "model_name", "memory", "total_training_time", "n_model_parameters",
    #     "n_trainable_parameters"
    # ]

    FIELDS = [
        "model_name",
    ]

    PYTHON_TYPES = [str, float, float, int, int]
    filename = "train"
    model_list_train, _ = parseModelFields(logfile_path, FIELDS, PYTHON_TYPES,
                                        filename)
    # print(model_list_train)
    print("Number of train files processed {:}.".format(len(model_list_train)))

    test_result_file = \
    [
    "train",
    ]


    test_result_fields = \
    [
    "model_name",
    "total_training_time",
    ]
    test_result_abbrev_fields = \
    [
    "NAME",
    "TRAIN-TIME",
    ]
    test_result_types = \
    [
    str,
    float,
    ]

    COLUMNS_TO_SORT = [1]
    DESCENTINGS = [False]

    for col in range(len(COLUMNS_TO_SORT)):
        COLUMN_TO_SORT = COLUMNS_TO_SORT[col]
        descenting = DESCENTINGS[col]

        fieldlist = test_result_fields
        fieldlist_abbrev = test_result_abbrev_fields
        typelist = test_result_types
        filenamelist = test_result_file
        for filename in filenamelist:

            model_list, model_dict = parseModelFields(logfile_path, fieldlist,
                                                      typelist, filename)

            print("Number of {:} files processed {:}.".format(
                filename, len(model_list)))
            # assert(len(model_list) > 0)

            COLUMN_NAME = fieldlist[COLUMN_TO_SORT]

            model_list_sorted = sortModelList(model_list,
                                              COLUMN_TO_SORT,
                                              descenting=descenting)

            filename = DIR_NAME + '/sorted_{:}_{:}'.format(
                COLUMN_NAME, filename)

            # max_length = np.max([len(st) for st in fieldlist_abbrev])

            table_header = ["RANK"]
            for i in range(1, len(fieldlist_abbrev)):
                table_header.append(fieldlist_abbrev[i])
            table_header.append(fieldlist_abbrev[0])

            # print(table_header)
            iter_ = 0
            table = []
            while (len(model_list_sorted) > iter_):
                model = model_list_sorted[iter_]
                print(model)
                table_line = []
                table_line.append("{:d}".format(iter_ + 1))
                # print(model)
                for element_iter in range(1, len(model)):
                    element = model[element_iter]
                    print(element)
                    element = float(element)
                    element = element / 60. # / 60.
                    table_line.append("{:.2f}".format(element))

                table_line.append(str(model[0]))
                # print(table_line)
                iter_ += 1
                table.append(table_line)

            with open(filename, 'w') as file_object:
                file_object.write(
                    tabulate(table, headers=table_header, tablefmt='orgtbl'))
