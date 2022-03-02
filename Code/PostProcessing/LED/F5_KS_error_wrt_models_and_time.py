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

""" Selection of color pallete designed for colorblind people """
color_labels = [
(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
(0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
(0.8352941176470589, 0.3686274509803922, 0.0),
(0.8, 0.47058823529411764, 0.7372549019607844),
(0.792156862745098, 0.5686274509803921, 0.3803921568627451),
# (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
# (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
# (0.9254901960784314, 0.8823529411764706, 0.2),
(0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
]

# linestyles = ['-','--','-.',':','-','--','-.',':','-','--']
linestyles = ['-','-','-.',':','-','--','-.',':','-','--']
linemarkers = ["x","o","s","d",">","<",">","x","o","s"]
linemarkerswidth = [2,1, 1, 1, 1, 1, 1, 1, 1, 1]
linemarkerssize = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]


# FONTSIZE=26
# legend_str_list = [""]

FONTSIZE=12
legend_str_list = ["_legend"]


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
        "RMSE": "RMNSE",
        "MSE": "MSE",
        "mnad_act": "MNAD$(u, \\tilde{u})$",
        "mnad_in": "MNAD$(v, \\tilde{v})$",
    }
    return dict_

field2Label = field2LabelDictAverage()


if system_name == "KSGP64L22":
    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_13-C_lstm-R_1x256-SL_25-LFO_0-LFL_1",
    "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_8-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_2.0-REG_0.0-NS_10",
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-L2_0.0-NL_0.0-RS_7-LR_0.001-C_lstm-R_1x1024-SL_50-LFO_1-LFL_0",
    "CNN-RC-scaler_MinMaxZeroOne-SOLVER_pinv-SIZE_8000-DEG_10-R_0.99-S_1.0-REG_1e-05-NS_10",
    ]
    file_names = \
    [
    "results_iterative_latent_forecasting_test",
    "results_iterative_latent_forecasting_test",
    "results_iterative_state_forecasting_test",
    "results_iterative_latent_forecasting_test",
    ]

    labels = \
    [
    "CNN-LSTM",
    "CNN-RC",
    "LSTM",
    "RC",
    ]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["MSE", "RMSE", "NAD"]

    # steps_plot = 100
    steps_plot = 10000
    lyapunov_time = 20.83

elif system_name == "KSGP64L22Large":
    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x512-SL_50-LFO_0-LFL_1",
    "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_2.0-REG_0.0-NS_10",
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-L2_0.0-NL_0.0-RS_7-LR_0.001-C_lstm-R_1x1024-SL_50-LFO_1-LFL_0",
    "CNN-RC-scaler_MinMaxZeroOne-SOLVER_pinv-SIZE_8000-DEG_10-R_0.99-S_1.0-REG_1e-05-NS_10",
    ]
    file_names = \
    [
    "results_iterative_latent_forecasting_test",
    "results_iterative_latent_forecasting_test",
    "results_iterative_state_forecasting_test",
    "results_iterative_latent_forecasting_test",
    ]

    labels = \
    [
    "CNN-LSTM",
    "CNN-RC",
    "LSTM",
    "RC",
    ]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["MSE", "RMSE", "NAD"]

    steps_plot = 200
    # steps_plot = 10000
    lyapunov_time = 20.83
else:
    raise ValueError("system_name={:} not found.".format(system_name))

# for mdn in [6, 5, 4, 2, 0]:
for mdn in [4, 2, 0]:
    del color_labels[mdn]
    del linestyles[mdn]
    del linemarkers[mdn]
    del linemarkerswidth[mdn]
    del linemarkerssize[mdn]


saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)

print(system_name)

if MAKE_FIGURE:
    FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
    os.makedirs(FIGURES_PATH, exist_ok=True)


DATA_PATH = "./{:}/{:}/Data".format(system_name, Experiment_Name)
os.makedirs(DATA_PATH, exist_ok=True)


for field2plot in fields2plot:

    plot_name_base = "F5_{:}_wrt_models_in_time".format(field2plot)
    """ Create figure for specific field """

    data_dict = {}

    if SEARCH_FOR_DATA:

        for modelnum in range(len(model_names)):
            modelname = model_names[modelnum]
            print(modelname)
            

            filename = file_names[modelnum] + ".pickle"

            result_path = saving_path + os.sep + "/Evaluation_Data" + os.sep + modelname + os.sep + filename
            assert os.path.isfile(result_path), "File {:} not found".format(result_path)

            data_result = utils.loadDataPickle(result_path)
            data = data_result[field2plot]
            data = np.array(data)  
            print(np.shape(data))

            # Time is included in the metric
            assert len(np.shape(data))==2, "np.shape(data)={:}".format(np.shape(data))

            num_ics, prediction_horizon = np.shape(data)

            data = data[:, :steps_plot]
            print(np.shape(data))
            # Mean over initial conditions
            data_mean = np.mean(data, axis=(0))
            # min and max over initial conditions
            y_data_min = np.min(data, axis=(0))
            y_data_max = np.max(data, axis=(0))

            y_data = data_mean
            y_data_range = y_data_max - y_data_min


            y_data_log = np.log(y_data)
            # min and max over initial conditions
            y_data_min_log = np.min(np.log(data), axis=(0))
            y_data_max_log = np.max(np.log(data), axis=(0))

            data_dict.update({
                 modelname:{
                 "data":data.T,
                 "y_data":y_data,
                 "y_data_min":y_data_min,
                 "y_data_max":y_data_max,
                 "y_data_log":y_data_log,
                 "y_data_min_log":y_data_min_log,
                 "y_data_max_log":y_data_max_log,
                 "y_data_range":y_data_range,
                 "label_model":labels[modelnum],
                 }})

        data_dict.update({
                         "misc":{
                         "prediction_horizon":prediction_horizon,
                         }})

        utils.saveDataPickle(data_dict, DATA_PATH + "/{:}".format(plot_name_base), add_file_format=True)

    else:
        data_dict_load = utils.loadDataPickle(DATA_PATH + "/{:}".format(plot_name_base), add_file_format=True)
        data_dict.update(data_dict_load)



    if MAKE_FIGURE:

        for legend_str in legend_str_list:
            for log_str in ["", "_log"]:


                fig_path = FIGURES_PATH + "/{:}{:}{:}.pdf".format(plot_name_base, log_str, legend_str)
                fig, ax = plt.subplots()

                for iter_ in range(len(model_names)):
                    modelname = model_names[iter_]
                  
                    y_data = data_dict[modelname]["y_data"]
                    y_data_min = data_dict[modelname]["y_data_min"]
                    y_data_max = data_dict[modelname]["y_data_max"]

                    y_data_log = data_dict[modelname]["y_data_log"]
                    y_data_min_log = data_dict[modelname]["y_data_min_log"]
                    y_data_max_log = data_dict[modelname]["y_data_max_log"]

                    if log_str == "_log":
                        y_data_plot = y_data_log
                        y_data_min_plot = y_data_min_log
                        y_data_max_plot = y_data_max_log
                    else:
                        y_data_plot = y_data
                        y_data_min_plot = y_data_min
                        y_data_max_plot = y_data_max


                    time_evol = np.arange(len(y_data_plot))*dt
                    if lyapunov_time > 0:
                        time_evol = time_evol / lyapunov_time

                    ax.plot(time_evol, y_data_plot, color=color_labels[iter_], label=data_dict[modelname]["label_model"], marker=linemarkers[iter_], markevery=len(y_data_plot)//10, linewidth=2, markersize=linemarkerssize[iter_], markeredgewidth=linemarkerswidth[iter_], linestyle=linestyles[iter_])

                    # if not (log_str == "_log"): ax.fill_between(time_evol, y_data_max_plot,  y_data_min_plot, facecolor=color_labels[iter_], alpha=0.3)
                    ax.fill_between(time_evol, y_data_max_plot,  y_data_min_plot, facecolor=color_labels[iter_], alpha=0.3)

                ylabel = field2Label[field2plot]
                if log_str == "_log": ylabel = "$\\log($" + ylabel + "$)$"
                plt.ylabel(r"{:}".format(ylabel))

                if lyapunov_time > 0:
                    plt.xlabel(r"$t / \Lambda_1$")
                else:
                    plt.xlabel(r"$t$")

                # tf = utils.formatTime(data_dict["misc"]["prediction_horizon"] * dt)
                # title_str = "$T_{f}=" + "{:}".format(tf) + "$"
                # plt.title(r"{:}".format(title_str), pad=20)
                if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()




