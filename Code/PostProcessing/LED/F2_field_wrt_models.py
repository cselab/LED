
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

""" PLOTTING PARAMETERS """
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm

print("-V- Matplotlib Version = {:}".format(matplotlib.__version__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('system_name', type=str)
parser.add_argument('SEARCH_FOR_DATA', type=int)
parser.add_argument('MAKE_FIGURE', type=int)
parser.add_argument('Experiment_Name', type=str)
args = parser.parse_args()
system_name = str(args.system_name)
MAKE_FIGURE = int(args.MAKE_FIGURE)
SEARCH_FOR_DATA = int(args.SEARCH_FOR_DATA)
Experiment_Name = str(args.Experiment_Name)

print("system_name = {:}".format(system_name))
print("MAKE_FIGURE = {:}".format(MAKE_FIGURE))
print("SEARCH_FOR_DATA = {:}".format(SEARCH_FOR_DATA))
print("Experiment_Name = {:}".format(Experiment_Name))

# # Experiment_Name="Experiment_Daint_Large"
# # Experiment_Name="Experiment_Barry"
# Experiment_Name=None

# # SEARCH_FOR_DATA=True
# # MAKE_FIGURE=False

# # SEARCH_FOR_DATA=False
# # MAKE_FIGURE=True

# SEARCH_FOR_DATA=True
# MAKE_FIGURE=True

""" Selection of color pallete designed for colorblind people """
color_labels = [
(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
(0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
(0.8352941176470589, 0.3686274509803922, 0.0),
(0.8, 0.47058823529411764, 0.7372549019607844),
(0.792156862745098, 0.5686274509803921, 0.3803921568627451),
(0.984313725490196, 0.6862745098039216, 0.8941176470588236),
(0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
(0.9254901960784314, 0.8823529411764706, 0.2),
(0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
]

linestyles = ['-','--','-.',':','-','--','-.',':']
linemarkers = ["x","o","s","d",">","<",">"]
linemarkerswidth = [3,2,2,2,2,2,2]



# with_legend = True
with_legend = False
# legend_str_list = ["", "_legend"]
if with_legend:
    FONTSIZE=12
    legend_str_list = ["_legend"]
    markersize = 6
else:
    FONTSIZE=26
    legend_str_list = [""]
    markersize = 10


font = {'size':FONTSIZE, 'family':'Times New Roman'}
matplotlib.rc('xtick', labelsize=FONTSIZE) 
matplotlib.rc('ytick', labelsize=FONTSIZE) 
matplotlib.rc('font', **font)

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Plotting parameters
rc('text', usetex=True)
plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'

# FIGTYPE="png"
FIGTYPE="pdf"


def field2LabelDictAverage():
    dict_ = {
        "NAD": "MNAD",
        "RMSE": "RMSE",
        # "NRMSE": "NRMSE",
        "NRMSE": "MNAD",
        "MSE": "MSE",
        "mnad_act": "MNAD$(u, \\tilde{u})$",
        "mnad_in": "MNAD$(v, \\tilde{v})$",
        "state_dist_L1_hist_error": "L1-NHD",
        "state_dist_wasserstein_distance": "WD",
        "CORR":"Correlation",
        "drag_coef_error_rel_all":"$|C_d-\\tilde{C_d}|/|\\tilde{C_d}|$",
        # "mse_avg": "MSE",
        # "rmse_avg": "RMSE",
        # "abserror_avg": "ABS",
        # "state_dist_L1_hist_error_all": "L1-NHD",
        # "state_dist_wasserstein_distance_all": "WD",
        # "state_dist_L1_hist_error_avg": "L1-NHD",
        # "state_dist_wasserstein_distance_avg": "WD",
        # "rmnse_avg_over_ics": "RMNSE",
        # "mnad_avg_over_ics_act": "NAD$(u, \\tilde{u})$",
        # "mnad_avg_over_ics_in": "NAD$(v, \\tilde{v})$",
        # "mnad_avg_over_ics": "NAD",
    }
    return dict_

field2Label = field2LabelDictAverage()


# set_ = "train"
# set_ = "test"


# for system_name in ["FHN"]:
# for system_name in ["KSGP64L22"]:

if system_name == "FHN":

    # model_names = \
    # [
    # "CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_114-CHANNELS_2-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_selu-DKP_1.0-LD_4-SOLVER_pinv-SIZE_1000-DEG_10-R_1.0-S_2.0-REG_0.001-NS_1",
    # "CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_114-CHANNELS_2-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_selu-DKP_1.0-LD_4-TYPE_continuous-PO_2-THRES_0.01-LIB_poly-INT_20-PORD_3-WS_11-D_32",
    # "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_114-CHANNELS_2-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_selu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_0-LFL_1",
    # ]
    # labels = \
    # [
    # "CNN-RC",
    # "CNN-SINDy",
    # "CNN-LSTM",
    # ]

    # filename_base = "results_multiscale_forecasting_micro_10_macro_1000_test"


    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1",
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x64-SL_60-LFO_1-LFL_1",
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_mlp-R_3x100-SL_25-LFO_1-LFL_1",
    "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_1.0-REG_1e-05-NS_10",
    # "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_5-PORD_3-WS_7",
    "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_1",
    ]


    labels = \
    [
    "AE-LSTM-end2end",
    "AE-LSTM",
    "AE-MLP",
    "AE-RC",
    "AE-SINDy",
    ]

    filenames_base = [
    "results_iterative_latent_forecasting_val",
    "results_iterative_latent_forecasting_test",
    ]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["MSE", "mnad_act", "mnad_in"]
    # logPlot = np.ones_like(fields2plot)
    logPlot = np.zeros_like(fields2plot)

    widths = 0.1

elif system_name == "KSGP64L22":

    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-C_lstm-R_1x128-SL_25-LFO_1-LFL_1",
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_13-C_lstm-R_1x256-SL_25-LFO_0-LFL_1",
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_7-PRETRAIN-AE_1-RS_7-C_mlp-R_3x100-SL_25-LFO_0-LFL_1",
    "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_2.0-REG_0.0-NS_10",
    # "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_1",
    "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_1",
    ]

    labels = \
    [
    "CNN-LSTM-end2end",
    "CNN-LSTM",
    "CNN-MLP",
    "CNN-RC",
    "CNN-SINDy",
    ]

    filenames_base = [
    "results_iterative_latent_forecasting_val",
    "results_iterative_latent_forecasting_test",
    ]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["RMSE", "NAD", "state_dist_L1_hist_error"]
    # state_dist_wasserstein_distance
    logPlot = np.ones_like(fields2plot)

    widths = 0.1


elif system_name == "KSGP64L22Large":

    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-C_lstm-R_1x512-SL_50-LFO_1-LFL_1",
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x512-SL_50-LFO_0-LFL_1",
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_mlp-R_3x100-SL_25-LFO_1-LFL_1",
    "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_2.0-REG_0.0-NS_10",
    # "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-TYPE_continuous-PO_3-THRES_0.001-LIB_poly-INT_5-PORD_3-WS_7",
    "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-TYPE_continuous-PO_3-THRES_0.0001-LIB_poly-INT_1",
    ]

    labels = \
    [
    "CNN-LSTM-end2end",
    "CNN-LSTM",
    "CNN-MLP",
    "CNN-RC",
    "CNN-SINDy",
    ]

    filenames_base = [
    "results_iterative_latent_forecasting_val",
    "results_iterative_latent_forecasting_test",
    ]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["RMSE", "NAD", "state_dist_L1_hist_error"]
    # state_dist_wasserstein_distance
    logPlot = np.ones_like(fields2plot)

    widths = 0.1

elif system_name in ["cylRe100"]:

    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_1-LFL_1",
    # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_mlp-R_3x100-SL_20-LFO_1-LFL_1",
    "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_2-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_2.0-REG_1e-05-NS_10",
    "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_2-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_1",
    ]
    labels = \
    [
    "CNN-LSTM",
    # "CNN-MLP",
    "CNN-RC",
    "CNN-SINDy",
    ]

    filenames_base = [
    "results_iterative_latent_forecasting_val",
    "results_iterative_latent_forecasting_test",
    ]

    fields2plot = ["RMSE", "CORR"]
    logPlot = [True, False]
    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]
    widths = 0.1

    for mdn in [2, 0]:
        del color_labels[mdn]
        del linestyles[mdn]
        del linemarkers[mdn]
        del linemarkerswidth[mdn]


# elif system_name in [""]:

#     model_names = \
#     [
#     "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_3-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_1-LFL_1",
#     # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_3-PRETRAIN-AE_1-RS_7-C_mlp-R_3x100-SL_20-LFO_1-LFL_1",
#     "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_3-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_1.0-REG_0.0001-NS_10",
#     "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_3-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_1",
#     ]
#     labels = \
#     [
#     "CNN-LSTM",
#     # "CNN-MLP",
#     "CNN-RC",
#     "CNN-SINDy",
#     ]

#     filenames_base = [
#     "results_iterative_latent_forecasting_val",
#     "results_iterative_latent_forecasting_test",
#     ]

#     fields2plot = ["RMSE", "CORR"]
#     logPlot = [True, False]
#     dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]
#     widths = 0.1


#     for mdn in [2, 0]:
#         del color_labels[mdn]
#         del linestyles[mdn]
#         del linemarkers[mdn]
#         del linemarkerswidth[mdn]

elif system_name in ["cylRe1000HRDt005"]:

    # model_names = \
    # [
    # "GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",
    # "RC-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-SOLVER_pinv-SIZE_200-DEG_10-R_0.99-S_1.0-REG_0.001-NS_10",
    # "SINDy-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-TYPE_continuous-PO_1-THRES_0.001-LIB_poly-INT_5",
    # ]

    model_names = \
    [
    "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
    "RC-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-SOLVER_pinv-SIZE_200-DEG_10-R_0.99-S_2.0-REG_0.0001-NS_10",
    "SINDy-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-TYPE_continuous-PO_1-THRES_0.001-LIB_poly-INT_1",
    ]

    labels = \
    [
    "CNN-LSTM",
    "CNN-RC",
    "CNN-SINDy",
    ]

    filenames_base = [
    "results_iterative_latent_forecasting_val",
    "results_iterative_latent_forecasting_test",
    ]

    fields2plot = ["NRMSE", "drag_coef_error_rel_all"]
    logPlot = [True, False]
    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]
    widths = 0.1


    for mdn in [2, 0]:
        del color_labels[mdn]
        del linestyles[mdn]
        del linemarkers[mdn]
        del linemarkerswidth[mdn]

elif system_name in ["cylRe1000HR"]:

    # model_names = \
    # [
    # # "GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",
    # "GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-OPT_adabelief-C_lstm-R_1x64-SL_25-RS_17-LR_0.01",
    # "RC-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-SOLVER_pinv-SIZE_200-DEG_10-R_0.99-S_0.5-REG_0.001-NS_10",
    # "SINDy-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-TYPE_continuous-PO_1-THRES_0.001-LIB_poly-INT_5",
    # ]

    model_names = \
    [
    # "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_17-C_lstm-R_1x32-SL_10-LFO_0-LFL_1",
    # "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_10-LFO_0-LFL_1",
    "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
    "RC-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-SOLVER_pinv-SIZE_200-DEG_10-R_0.99-S_0.5-REG_0.001-NS_10",
    # "SINDy-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-TYPE_continuous-PO_1-THRES_0.001-LIB_poly-INT_5",
    "SINDy-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-TYPE_continuous-PO_1-THRES_0.001-LIB_poly-INT_1",
    ]

    labels = \
    [
    "CNN-LSTM",
    "CNN-RC",
    "CNN-SINDy",
    ]

    filenames_base = [
    "results_iterative_latent_forecasting_val",
    "results_iterative_latent_forecasting_test",
    ]

    # fields2plot = ["RMSE", "drag_coef_error_rel_all", "NRMSE"]
    # logPlot = [True, False, True]

    fields2plot = ["NRMSE", "drag_coef_error_rel_all"]
    logPlot = [True, False]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]
    widths = 0.1


    for mdn in [2, 0]:
        del color_labels[mdn]
        del linestyles[mdn]
        del linemarkers[mdn]
        del linemarkerswidth[mdn]

elif system_name in ["cylRe100HRDt005"]:

    model_names = \
    [
    "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x64-SL_10-LFO_0-LFL_1",
    "RC-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-SOLVER_pinv-SIZE_200-DEG_10-R_0.99-S_2.0-REG_1e-05-NS_10",
    "SINDy-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-TYPE_continuous-PO_1-THRES_1e-05-LIB_poly-INT_5-PORD_3-WS_7",
    ]


    # model_names = \
    # [
    # "GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-OPT_adabelief-C_lstm-R_1x16-SL_10-RS_17-LR_0.01",
    # "RC-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-SOLVER_pinv-SIZE_200-DEG_10-R_0.99-S_0.5-REG_0.001-NS_10",
    # "SINDy-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-TYPE_continuous-PO_2-THRES_1e-05-LIB_poly-INT_5-PORD_3-WS_7",
    # ]
    labels = \
    [
    "CNN-LSTM",
    "CNN-RC",
    "CNN-SINDy",
    ]

    filenames_base = [
    "results_iterative_latent_forecasting_val",
    "results_iterative_latent_forecasting_test",
    ]

    fields2plot = ["NRMSE", "drag_coef_error_rel_all"]
    logPlot = [True, False]
    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]
    widths = 0.1


    for mdn in [2, 0]:
        del color_labels[mdn]
        del linestyles[mdn]
        del linemarkers[mdn]
        del linemarkerswidth[mdn]

elif system_name in ["cylRe100HR"]:

    model_names = \
    [
    # "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
    "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_10-LFO_0-LFL_1",
    "RC-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-SOLVER_pinv-SIZE_200-DEG_10-R_0.99-S_2.0-REG_0.0001-NS_10",
    # "SINDy-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-TYPE_continuous-PO_1-THRES_1e-05-LIB_poly-INT_5-PORD_3-WS_7",
    "SINDy-GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-TYPE_continuous-PO_1-THRES_1e-05-LIB_poly-INT_1",
    ]
    

    # model_names = \
    # [
    # "GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",
    # "RC-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-SOLVER_pinv-SIZE_200-DEG_10-R_0.99-S_0.5-REG_0.001-NS_10",
    # "SINDy-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-TYPE_continuous-PO_1-THRES_0.001-LIB_poly-INT_5",
    # ]
    labels = \
    [
    "CNN-LSTM",
    "CNN-RC",
    "CNN-SINDy",
    ]

    filenames_base = [
    "results_iterative_latent_forecasting_val",
    "results_iterative_latent_forecasting_test",
    ]

    fields2plot = ["NRMSE", "drag_coef_error_rel_all"]
    logPlot = [True, False]
    # fields2plot = ["RMSE"]
    # logPlot = [True]
    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]
    widths = 0.1


    for mdn in [2, 0]:
        del color_labels[mdn]
        del linestyles[mdn]
        del linemarkers[mdn]
        del linemarkerswidth[mdn]
        


else:
    raise ValueError("system_name={:} not found.".format(system_name))

saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)

print(system_name)

if MAKE_FIGURE:
    FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
    os.makedirs(FIGURES_PATH, exist_ok=True)


DATA_PATH = "./{:}/{:}/Data".format(system_name, Experiment_Name)
os.makedirs(DATA_PATH, exist_ok=True)


# widths = 0.8 / len(model_names)

# field2plot = fields2plot[0]

for filename_base in filenames_base:

    for field2plotNum in range(len(fields2plot)):
        field2plot = fields2plot[field2plotNum]
        plotLog = logPlot[field2plotNum]

        data_dict = {}

        plot_name_base = "F2_{:}_{:}_wrt_model".format(filename_base, field2plot)

        if SEARCH_FOR_DATA:
            """ Create figure for specific field """
            for modelnum in range(len(model_names)):
                modelname = model_names[modelnum]

                filename = filename_base + ".pickle"
                result_path = saving_path + os.sep + "/Evaluation_Data" + os.sep + modelname + os.sep + filename

                assert os.path.isfile(result_path), "File {:} not found".format(result_path)

                data_result = utils.loadDataPickle(result_path)
                # for key in data_result: print(key)
                # print(data_result)
                assert field2plot in data_result, "field {:} not found in {:}.".format(field2plot, result_path)
                data = data_result[field2plot]

                # print(data)
                print(np.shape(data))
                # print(data[0])

                # assert len(np.shape(data)) == 2, "np.shape(data)={:}".format(np.shape(data))

                if len(np.shape(data)) == 2:
                    num_ics, prediction_horizon = np.shape(data)
                    # Mean over time
                    data = np.mean(data, axis=(1))

                elif len(np.shape(data)) == 1:
                    num_ics = np.shape(data)
                    prediction_horizon = np.shape(data_result["MSE"])[1]

                elif len(np.shape(data)) == 3:
                    num_ics, prediction_horizon, C = np.shape(data)
                    assert C==4
                    # Mean over channels
                    data = np.mean(data, axis=(2))
                    # Mean over time
                    data = np.mean(data, axis=(1))

                elif len(np.shape(data))==0:
                    data = np.array([data])
                    prediction_horizon = np.shape(data_result["MSE"])[1]

                else:
                    print(np.shape(data))
                    raise ValueError("Not implemented.")

                # Mean over initial conditions
                data_mean = np.mean(data, axis=(0), keepdims=True)
                # min and max over initial conditions
                data_min = np.min(data, axis=(0), keepdims=True)
                data_max = np.max(data, axis=(0), keepdims=True)

                x_pos = np.arange(np.shape(data_mean)[0])
                y_data = data_mean
                y_data_range = data_max - data_min

                data_dict.update({
                                 modelname:{
                                 "prediction_horizon":prediction_horizon,
                                 "data":data,
                                 "x_pos":x_pos,
                                 "y_data":y_data,
                                 "y_data_min":data_min,
                                 "y_data_max":data_max,
                                 "y_data_range":y_data_range,
                                 "label_model":labels[modelnum],
                                 "color":color_labels[modelnum],
                                 "marker":linemarkers[modelnum],
                                 "markerwidth":linemarkerswidth[modelnum],
                                 }})
            utils.saveDataPickle(data_dict, DATA_PATH + "/{:}".format(plot_name_base), add_file_format=True)

        else:

            data_dict_load = utils.loadDataPickle(DATA_PATH + "/{:}".format(plot_name_base), add_file_format=True)
            data_dict.update(data_dict_load)


            
        if MAKE_FIGURE:



            for legend_str in legend_str_list:

                fig_path = FIGURES_PATH + "/{:}{:}.pdf".format(plot_name_base, legend_str)
                fig, ax = plt.subplots()
                iter_ = 0
                x_ticks = []
                x_labels = []
                for modelname in model_names:

                    prediction_horizon = data_dict[modelname]["prediction_horizon"]

                    x_pos = list(data_dict[modelname]["x_pos"] + 0.1 * iter_)
                    x_label = data_dict[modelname]["label_model"]
                    x_ticks.append(x_pos)
                    x_labels.append(x_label)

                    violin_parts = ax.violinplot(data_dict[modelname]["data"],
                                  positions=x_pos,
                                  widths=widths,
                                  )

                    ax.errorbar(data_dict[modelname]["x_pos"] + 0.1 * iter_,
                                data_dict[modelname]["y_data"],
                                yerr=[data_dict[modelname]["y_data"]-data_dict[modelname]["y_data_min"], data_dict[modelname]["y_data_max"]-data_dict[modelname]["y_data"]],
                                # alpha=0.5,
                                color=data_dict[modelname]["color"],
                                marker=data_dict[modelname]["marker"],
                                label=data_dict[modelname]["label_model"],
                                markeredgewidth=data_dict[modelname]["markerwidth"],
                                linewidth=2,
                                markersize=markersize,
                                ls='none',
                                )


                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(data_dict[modelname]["color"])
                        pc.set_edgecolor(data_dict[modelname]["color"])
                    for partname in ('cbars','cmins','cmaxes'):
                        vp = violin_parts[partname]
                        vp.set_edgecolor(data_dict[modelname]["color"])
                        

                    iter_ += 1

                # ax.set_xticks([])
                # ax.set_xticklabels([])

                # print(x_ticks)
                # print(x_labels)
                # x_ticks = [temp[0] for temp in x_ticks]
                # ax.set_xticks(x_ticks)
                # ax.set_xticklabels(x_labels, rotation = 45)


                # ax.yaxis.grid(True)
                # plt.ylim(bottom=0)

                ax.yaxis.grid(True)
                ax.set_xticks([])
                ax.set_xticklabels([])
                plt.ylim(bottom=0)


                ylabel = field2Label[field2plot]
                plt.ylabel(r"{:}".format(ylabel))


                tf = utils.formatTime(prediction_horizon * dt)
                title_str = "$T_{f}=" + "{:}".format(tf) + "$"

                # print(title_str)
                plt.title(r"{:}".format(title_str), pad=20)
                if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

                # plt.show()
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()




            if plotLog:


                for legend_str in legend_str_list:

                    fig_path = FIGURES_PATH + "/{:}_log{:}.pdf".format(plot_name_base, legend_str)

                    fig, ax = plt.subplots()
                    iter_ = 0
                    for modelname in model_names:

                        data = np.log10(data_dict[modelname]["data"])
                        y_data = np.log10(data_dict[modelname]["y_data"])
                        y_err_min = np.log10(data_dict[modelname]["y_data"]) - np.log10(data_dict[modelname]["y_data_min"])
                        y_err_max = np.log10(data_dict[modelname]["y_data_max"]) - np.log10(data_dict[modelname]["y_data"])

                        violin_parts = ax.violinplot(data,
                                      positions=data_dict[modelname]["x_pos"] + 0.1 * iter_,
                                      widths=widths,
                                      )

                        ax.errorbar(data_dict[modelname]["x_pos"] + 0.1 * iter_,
                                    y_data,
                                    yerr=[y_err_min, y_err_max],
                                    # alpha=0.5,
                                    color=data_dict[modelname]["color"],
                                    marker=data_dict[modelname]["marker"],
                                    label=data_dict[modelname]["label_model"],
                                    markeredgewidth=data_dict[modelname]["markerwidth"],
                                    markersize=markersize,
                                    linewidth=2,
                                    ls='none',
                                    )


                        for pc in violin_parts['bodies']:
                            pc.set_facecolor(data_dict[modelname]["color"])
                            pc.set_edgecolor(data_dict[modelname]["color"])
                        for partname in ('cbars','cmins','cmaxes'):
                            vp = violin_parts[partname]
                            vp.set_edgecolor(data_dict[modelname]["color"])


                        iter_ += 1

                    ax.set_xticks([])
                    ax.set_xticklabels([])

                    ax.yaxis.grid(True)

                    ylabel = field2Label[field2plot]
                    plt.ylabel(r"Log(" + "{:}".format(ylabel) +")")

                    tf = utils.formatTime(prediction_horizon * dt)
                    title_str = "$T_{f}=" + "{:}".format(tf) + "$"

                    # print(title_str)
                    plt.title(r"{:}".format(title_str), pad=20)
                    if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                    # plt.show()
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    plt.close()




