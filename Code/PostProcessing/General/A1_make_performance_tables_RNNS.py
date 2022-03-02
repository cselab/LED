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

sys.path.append('../../Methods')
from Config.global_conf import global_params

Experiment_Name = "Experiment_Daint_Large"
# Experiment_Name=None

for system_name in [
    "cylRe100_vortPres_veryLowRes",
]:

    DIR_NAME = "./{:}/Postprocessing_Logs".format(system_name)
    os.makedirs(DIR_NAME, exist_ok=True)

    saving_path = getSavingPath(Experiment_Name, system_name, global_params)
    logfile_path = saving_path + global_params.logfile_dir

    print(system_name)
    print(logfile_path)


    test_result_file = \
    [
        "results_iterative_latent_forecasting_test", # IN THE CASE OF CNN-RNN (Autoencoding)
        "results_iterative_latent_forecasting_val",
        # "results_iterative_state_forecasting_test", # IN THE CASE OF CONV-RNN
        # "results_iterative_state_forecasting_val",
    ]

    test_result_fields = \
    [
    "model_name",
    "RMSE_avg",
    # "drag_coef_error_rel",
    ]
    test_result_abbrev_fields = \
    [
    "NAME",
    "RMSE",
    # "CD-REL",
    ]
    test_result_types = \
    [
    str,
    float,
    # float,
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

            model_list, model_dict = parseModelFields(logfile_path, fieldlist, typelist, filename)

            print("Number of {:} files processed {:}.".format(
                filename, len(model_list)))
            assert(len(model_list) > 0)

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
                table_line = []
                table_line.append("{:d}".format(iter_ + 1))
                # print(model)
                for element_iter in range(1, len(model)):
                    element = model[element_iter]
                    # element = typelist[element_iter](element)
                    if isinstance(element, float):
                        table_line.append("{:.4f}".format(element))
                    elif isinstance(element, str):
                        table_line.append(str(element))
                    else:
                        # print(table_line)
                        # print(element)
                        raise ValueError("Unknown type.")
                table_line.append(str(model[0]))
                # print(table_line)
                iter_ += 1
                table.append(table_line)

            with open(filename, 'w') as file_object:
                file_object.write(
                    tabulate(table, headers=table_header, tablefmt='orgtbl'))
