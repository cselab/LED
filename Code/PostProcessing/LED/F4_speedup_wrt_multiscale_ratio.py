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
parser.add_argument('Experiment_Name', type=str)
args = parser.parse_args()
system_name = str(args.system_name)
Experiment_Name = str(args.Experiment_Name)

print("system_name = {:}".format(system_name))
print("Experiment_Name = {:}".format(Experiment_Name))

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
else:
    FONTSIZE=26
    legend_str_list = [""]

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
        "time_total_per_iter": "Speed-up",
    }
    return dict_

field2Label = field2LabelDictAverage()

if system_name == "FHN":
    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1",
    ]
    labels = \
    [
    "CNN-LSTM-end2end",
    ]

    color_idx = 0
    width = 0.8

    micro_step = 10
    micro_step_list = [micro_step]
    macro_step_list = [0, 10, 50, 100, 200, 1000, 9000]
    prediction_horizon = 8000

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

elif system_name == "KSGP64L22":
    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_13-C_lstm-R_1x256-SL_25-LFO_0-LFL_1",
    ]
    labels = \
    [
    "CNN-LSTM",
    ]
    color_idx = 1
    width = 0.8

    micro_step = 32
    micro_step_list = [micro_step]
    macro_step_list = [0, 16, 32, 64, 128, 256, 320, 512, 640, 3400]
    prediction_horizon = 3200

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

elif system_name == "KSGP64L22Large":
    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x512-SL_50-LFO_0-LFL_1",
    ]
    labels = \
    [
    "CNN-LSTM",
    ]
    color_idx = 1
    width = 0.8

    micro_step = 16
    micro_step_list = [micro_step]
    macro_step_list = [0, 4, 8, 16, 32, 64, 128, 256]
    prediction_horizon = 200

    # # VERY GOOD RESULTS
    # micro_step = 10
    # macro_step_list = [0, 5, 10, 20, 40, 80, 220]
    # prediction_horizon = 200
    
    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

else:
    raise ValueError("system_name={:} not found.".format(system_name))

saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)
logfile_path = saving_path + global_params.logfile_dir

print(system_name)
print(logfile_path)

FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
os.makedirs(FIGURES_PATH, exist_ok=True)




fields2plot = ["time_total_per_iter"]
field2plot = fields2plot[0]

for micro_step in micro_step_list:

    rho_plot = np.array([float(macro_step) for macro_step in macro_step_list]) / float(micro_step)
    rho_plot = [utils.formatTime(rho_) for rho_ in rho_plot]
    print(rho_plot)

    """ Create figure for specific field """
    data_dict = {}
    for modelnum in range(len(model_names)):
        modelname = model_names[modelnum]

        field_data = []
        for macro_step in macro_step_list:

            filename = "results_multiscale_forecasting_micro_{:}_macro_{:}_test.txt".format(micro_step, macro_step)

            logfile_path = saving_path + os.sep + "/Logfiles" + os.sep + modelname + os.sep + filename

            assert os.path.isfile(logfile_path), "File {:} not found".format(logfile_path)

            with open(logfile_path) as f:
                lines = f.readlines()
            if len(lines)>1:
                print("More than one line found. Taking into account last model.")
            # print(lines)
            """ Taking into account last model """
            lines = lines[-1]
            """ Discarding newline character """
            lines = lines[:-2]
            """ Splitting """
            lines = lines.split(":")
            # print(lines)
            line_dict = {lines[i]:lines[i+1] for i in range(0, len(lines)//2, 2)}
            for key in line_dict:
                is_number = utils.is_number(line_dict[key])
                print("key = {:}, is_number = {:}".format(key, is_number))
                if is_number:
                    print("Transforming to float.")
                    line_dict[key] = float(line_dict[key])

            data = line_dict[field2plot]

            if macro_step == 0 and field2plot == "time_total_per_iter":
                time_microdynamics = data
                macro_zero_found = True

            else:
                # Calculating speed-up
                data = float(time_microdynamics) / np.array(data)
                field_data.append(data)

        if macro_zero_found:
            # Removing the macro=0 (used to calculate the time for microdynamics)
            macro_step_list_ = macro_step_list[1:]
            rho_plot_ = rho_plot[1:]
        else:
            macro_step_list_ = macro_step_list
            rho_plot_ = rho_plot
        x_pos = 1.0 * np.arange(len(macro_step_list_))
        y_data = np.array(field_data)

        # print(macro_step_list_)
        labels_plot = []
        for case in range(len(macro_step_list_)):
            macro_step = macro_step_list_[case]
            if macro_step < prediction_horizon:
                labels_plot.append(rho_plot_[case])
            else:
                labels_plot.append(str("Latent"))
                x_pos[case] += 0.3


        data_dict.update({
                         modelname:{
                         "labels_plot":labels_plot,
                         "x_pos":x_pos,
                         "y_data":y_data,
                         "label_model":labels[modelnum],
                         "color":color_labels[color_idx],
                         "marker":linemarkers[modelnum],
                         "markerwidth":linemarkerswidth[modelnum],
                         }})





    if len(model_names) % 2 == 1 :
        x0 = - (len(model_names)-1)/2 * width
    else:
        x0 = - len(model_names)/2 * width + width / 2.0


    for legend_str in legend_str_list:


        fig_path = FIGURES_PATH + "/F4_{:}_wrt_rho_micro{:}{:}.pdf".format(field2plot, micro_step, legend_str)
        fig, ax = plt.subplots()
        iter_ = 0
        for modelname in model_names:
            print(data_dict[modelname]["y_data"])
            ax.bar(
                    data_dict[modelname]["x_pos"] + x0 + width * iter_,
                    data_dict[modelname]["y_data"],
                    color=data_dict[modelname]["color"],
                    label=data_dict[modelname]["label_model"],
                    width=width,
                    hatch='//',
                    alpha=0.8, ecolor="white", edgecolor="white",
                    )

            iter_ += 1

        ax.set_xticks(data_dict[modelname]["x_pos"])
        ax.set_xticklabels(data_dict[modelname]["labels_plot"])

        ax.yaxis.grid(True)

        ylabel = field2Label[field2plot]
        plt.ylabel(r"{:}".format(ylabel))

        plt.xlabel(r"$\rho = T_m / T_{\mu}$")
        # micro_step_time = micro_step * dt / lyapunov_time
        
        tm = utils.formatTime(micro_step * dt)
        title_str = "$T_{\mu}=" + "{:}".format(tm)

        tf = utils.formatTime(prediction_horizon * dt)
        title_str += ", \, T_{f}=" + "{:}".format(tf) + "$"
        # print(title_str)
        plt.title(r"{:}".format(title_str), pad=20)
        if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        # plt.show()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()


    for legend_str in legend_str_list:
        fig_path = FIGURES_PATH + "/F4_log_{:}_wrt_rho_micro{:}.pdf".format(field2plot, micro_step, legend_str)
        fig, ax = plt.subplots()
        iter_ = 0
        for modelname in model_names:

            ax.bar(
                    data_dict[modelname]["x_pos"] + x0 + width * iter_,
                    np.log10(data_dict[modelname]["y_data"]),
                    color=data_dict[modelname]["color"],
                    label=data_dict[modelname]["label_model"],
                    width=width,
                    hatch='//',
                    alpha=0.8, ecolor="white", edgecolor="white",
                    )

            iter_ += 1

        ax.set_xticks(data_dict[modelname]["x_pos"])
        ax.set_xticklabels(data_dict[modelname]["labels_plot"])

        ax.yaxis.grid(True)
        plt.ylabel(r"{:}".format("$\log_{10}($Speed-up$)$"))
        plt.xlabel(r"$\rho = T_m / T_{\mu}$")

        tm = utils.formatTime(micro_step * dt)
        title_str = "$T_{\mu}=" + "{:}".format(tm)

        tf = utils.formatTime(prediction_horizon * dt)
        title_str += ", \, T_{f}=" + "{:}".format(tf) + "$"

        plt.title(r"{:}".format(title_str), pad=20)
        if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        # plt.show()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()






