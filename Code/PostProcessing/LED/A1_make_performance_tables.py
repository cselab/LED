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
    # "FHN",
    # "KSGP64L22",
    # "KSGP64L22Large",
    # "cylRe100",
    # "cylRe1000",
    "cylRe100HR",
    "cylRe1000HR",
    "cylRe100HRDt005",
    "cylRe1000HRDt005",
]:

    DIR_NAME = "./{:}/Postprocessing_Logs".format(system_name)
    os.makedirs(DIR_NAME, exist_ok=True)

    saving_path = getSavingPath(Experiment_Name, system_name, global_params)
    logfile_path = saving_path + global_params.logfile_dir

    print(system_name)
    print(logfile_path)


    test_result_file = \
    [
        # "results_multiscale_forecasting_micro_10_macro_1000_test",
        "results_iterative_latent_forecasting_test",
        "results_iterative_latent_forecasting_val",
        # "results_iterative_state_forecasting_test",
        # "results_iterative_state_forecasting_val",
    ]

    if system_name == "FHN":

        test_result_fields = \
        [
        "model_name",
        # "MSE_avg",
        "RMSE_avg",
        # "L1_hist_error_mean",
        ]
        test_result_abbrev_fields = \
        [
        "NAME",
        # "MSE",
        "RMSE",
        # "L1-HIST",
        ]
        test_result_types = \
        [
        str,
        # float,
        float,
        # float,
        ]


        COLUMNS_TO_SORT = [1]
        DESCENTINGS = [False]

    elif system_name in ["KSGP64L22", "KSGP64L22Large"]:

        test_result_fields = \
        [
        "model_name",
        # "MSE_avg",
        "RMSE_avg",
        "L1_hist_error_mean",
        "ux_uxx_l1_hist_error",
        ]
        test_result_abbrev_fields = \
        [
        "NAME",
        # "MSE",
        "RMSE",
        "L1-HIST",
        "UX-L1",
        ]
        test_result_types = \
        [
        str,
        # float,
        float,
        float,
        float,
        ]

        COLUMNS_TO_SORT = [1, 2, 3]
        DESCENTINGS = [False, False, False]

    elif system_name in [
    # "cylRe100",
    # "cylRe1000",
    "cylRe100HR",
    "cylRe1000HR",
    "cylRe100HRDt005",
    "cylRe1000HRDt005",
    ]:

        test_result_fields = \
        [
        "model_name",
        # "MSE_avg",
        "RMSE_avg",
        # "CORR_avg",
        # "L1_hist_error_mean",
        "drag_coef_error_rel",
        ]
        test_result_abbrev_fields = \
        [
        "NAME",
        # "MSE",
        "RMSE",
        # "CORR",
        # "L1-HIST",
        "CD-REL",
        ]
        test_result_types = \
        [
        str,
        # float,
        float,
        # float,
        # float,
        float,
        ]

        COLUMNS_TO_SORT = [1, 2]
        DESCENTINGS = [False, False]
        
        # COLUMNS_TO_SORT = [1, 2, 3]
        # DESCENTINGS = [False, False, False]

    else:
        raise ValueError("Unknown system_name {:}".format(system_name))


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
