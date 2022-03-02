
#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import glob, os
import pickle
import argparse
from Utils import utils
import os, sys, inspect
import re

import socket
hostname=socket.gethostname()
print(hostname)
sys.path.append('../../Methods')
from Config.global_conf import global_params

Experiment_Name="Experiment_Daint_Large"
# Experiment_Name="Experiment_Barry"


# for system_name in ["FHN"]:
for system_name in [
    # "FHN",
    "KSGP64L22Large",
    # "cylRe100HR",
    # "cylRe1000HR",
    ]:

    if system_name == "FHN":
        model_names = \
        [
        # "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        # "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_.",
        # # "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_5.0-DMN_10-LD_(.*)",
        # "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_1.0-DMN_10-LD_(.*)",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_.$",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_2-8-16-32-4-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_.",
        ]

        labels = \
        [
        # "PCA",
        # # "DiffMaps w=5.0",
        # # "DiffMaps",
        # "DiffMaps w=1.0",
        "AE",
        "CNN-AE",
        ]

        # model_names = \
        # [
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x(.*)-SL_(.*)-LFO_1-LFL_1",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x(.*)-SL_(.*)-LFO_1-LFL_1",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_mlp-R_3x100-SL_(.*)-LFO_1-LFL_1",
        # "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-SOLVER_pinv-SIZE_(.*)-DEG_(.*)-R_(.*)-S_(.*)-REG_(.*)-NS_(.*)",
        # # "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_5-PORD_3-WS_7",
        # "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-TYPE_continuous-PO_(.*)-THRES_(.*)-LIB_poly-INT_(.*)",
        # ]


        # labels = \
        # [
        # "AE-LSTM-end2end",
        # "AE-LSTM",
        # "AE-MLP",
        # "AE-RC",
        # "AE-SINDy",
        # ]

    elif system_name == "KSGP64L22Large":
        # model_names = \
        # [
        # "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        # # "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_5.0-DMN_10-LD_(.*)",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_.$",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_.$",
        # # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        # ]
        # labels = \
        # [
        # "PCA",
        # # "DiffMaps w=1.0",
        # "AE",
        # "CNN-AE",
        # # "CNN-AE T",
        # ]


        model_names = \
        [
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-C_lstm-R_(.*)-SL_(.*)-LFO_1-LFL_1",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_(.*)-SL_(.*)-LFO_0-LFL_1",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_mlp-R_(.*)-SL_(.*)-LFO_1-LFL_1",
        "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-SOLVER_pinv-SIZE_(.*)-DEG_(.*)-R_(.*)-S_(.*)-REG_(.*)-NS_10",
        # "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-TYPE_continuous-PO_3-THRES_0.001-LIB_poly-INT_5-PORD_3-WS_7",
        "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-TYPE_continuous-PO_(.*)-THRES_(.*)-LIB_poly-INT_1",
        ]

        labels = \
        [
        "CNN-LSTM-end2end",
        "CNN-LSTM",
        "CNN-MLP",
        "CNN-RC",
        "CNN-SINDy",
        ]

    elif system_name == "cylRe100HR":

        # model_names = \
        # [
        # # "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_\w",
        # "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_8",
        # ]
        # labels = \
        # [
        # "CNN",
        # ]


        model_names = \
        [
        # "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
        "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x(.*)-SL_(.*)-LFO_0-LFL_1",
        "RC-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-SOLVER_pinv-SIZE_(.*)-DEG_(.*)-R_(.*)-S_(.*)-REG_(.*)-NS_(.*)",
        # "SINDy-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-TYPE_continuous-PO_1-THRES_1e-05-LIB_poly-INT_5-PORD_3-WS_7",
        "SINDy-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-TYPE_continuous-PO_(.*)-THRES_(.*)-LIB_poly-INT_(.*)",
        ]
        
        labels = \
        [
        "CNN-LSTM",
        "CNN-RC",
        "CNN-SINDy",
        ]
    else:
        raise ValueError("Unknown system {:}.".format(system_name))


    saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)
    logfile_path = saving_path + global_params.logfile_dir

    print(system_name)
    print(logfile_path)

    FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    filename = "train"
    FIELDS=[
    "model_name",
    "total_training_time",
    ]

    PYTHON_TYPES=[
    str,
    float,
    ]


    data_dict = {}
    for modelnum in range(len(model_names)):
        modelname = model_names[modelnum]

        model_list, model_dict = utils.parseModelFields(logfile_path, FIELDS, PYTHON_TYPES, filename)

        model_regex            = re.compile(modelname)

        data = []
        for model in model_list:
            modelname_ = model[0]
            if model_regex.search(modelname_):
                value_ = float(model[1])
                value_ = value_ / 60. # Minutes
                data.append(value_)

        assert modelname not in data_dict
        data_dict.update({
                         modelname:{"data":data,
                         "label":labels[modelnum],
                         }})

    for modelname in data_dict.keys():
        print("#" * 20)
        label = data_dict[modelname]["label"]
        data = data_dict[modelname]["data"]
        data = np.array(data)
        print(label)
        print("Training time:")
        print("{:.2f}/{:.2f}/{:.2f} minutes".format(np.min(data), np.mean(data), np.max(data)))
        data = data/60.
        print("{:.2}/{:.2}/{:.2} hours".format(np.min(data), np.mean(data), np.max(data)))











