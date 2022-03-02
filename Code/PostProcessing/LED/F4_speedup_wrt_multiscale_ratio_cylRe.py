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



# system2Model = {
#     "cylRe100HR":"GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",
#     "cylRe1000HR":"GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",
#     "cylRe100HRDt005":"GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-OPT_adabelief-C_lstm-R_1x16-SL_10-RS_17-LR_0.01",
#     "cylRe1000HRDt005":"GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",    
# }


system2Model = {
    "cylRe100HR":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
    "cylRe1000HR":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
    "cylRe100HRDt005":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x64-SL_10-LFO_0-LFL_1",
    "cylRe1000HRDt005":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1", 
}


def field2LabelDictAverage():
    dict_ = {
        "time_total_per_iter": "Speed-up",
    }
    return dict_

field2Label = field2LabelDictAverage()

if system_name in [
"cylRe100HR",
"cylRe1000HR",
"cylRe100HRDt005",
"cylRe1000HRDt005",
]:
    model_names = \
    [
    system2Model[system_name],
    ]
    labels = \
    [
    "CNN-LSTM",
    ]
    color_idx = 1
    width = 0.8

    # micro_step = 5
    # macro_step_list = [0, 5, 10, 15, 20, 25, 50, 150]
    macro_step_list = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 64, 150]
    micro_step_list = [1, 2, 4, 8, 16]
    prediction_horizon = 100

    if system_name == "cylRe100HR":
        macro_step_list = [2, 4, 6, 10, 20, 150]
        micro_step_list = [2]

    if system_name == "cylRe1000HR":
        macro_step_list = [4, 8, 16, 32, 64, 150]
        micro_step_list = [8]

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


if "cylRe1000" in system_name:
    # Re=1000
    # Assuming reference run on 10 instead of 36 cores
    num_cores=3.6
    time_start=1491.200878
    time_end=1982.671927
    Deltat=(time_end-time_start)
    real_time=24*60*60 # 24 hours, transformed in seconds
    time_total_per_iter_micro=real_time/Deltat*dt
    time_microdynamics=time_total_per_iter_micro*num_cores
    macro_zero_found=False
elif "cylRe100" in system_name:
    # Re=1000
    # Assuming reference run on 10 instead of 36 cores
    num_cores=3.6
    time_start=0
    time_end=644.068309
    Deltat=(time_end-time_start)
    real_time=24*60*60 # 24 hours, transformed in seconds
    time_total_per_iter_micro=real_time/Deltat*dt
    time_microdynamics=time_total_per_iter_micro*num_cores
    macro_zero_found=False
else:
    raise ValueError("Bug.")

def getMultiscaleParams(multiscale_macro_steps, multiscale_micro_steps, prediction_horizon):
    macro_steps_per_round = []
    micro_steps_per_round = []
    steps = 0
    while (steps < prediction_horizon):
        steps_to_go = prediction_horizon - steps
        if steps_to_go >= multiscale_macro_steps:
            macro_steps_per_round.append(multiscale_macro_steps)
            steps += multiscale_macro_steps
        elif steps_to_go != 0:
            macro_steps_per_round.append(steps_to_go)
            steps += steps_to_go
        else:
            raise ValueError("This was not supposed to happen.")
        steps_to_go = prediction_horizon - steps
        if steps_to_go >= multiscale_micro_steps:
            micro_steps_per_round.append(multiscale_micro_steps)
            steps += multiscale_micro_steps
        elif steps_to_go != 0:
            micro_steps_per_round.append(steps_to_go)
            steps += steps_to_go
    return macro_steps_per_round, micro_steps_per_round



for micro_step in micro_step_list:

    for modelnum in range(len(model_names)):

        modelname = model_names[modelnum]
        macro_step = macro_step_list[-1]
        filename = "results_multiscale_forecasting_micro_{:}_macro_{:}_test.txt".format(micro_step, macro_step)
        logfile_path = saving_path + os.sep + "/Logfiles" + os.sep + modelname + os.sep + filename
        assert os.path.isfile(logfile_path), "File {:} not found".format(logfile_path)
        with open(logfile_path) as f: lines = f.readlines()
        if len(lines)>1: print("More than one line found. Taking into account last model.")
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

        time_macrodynamics = line_dict[field2plot]
        print(time_macrodynamics)
        print(time_microdynamics)


        rho_plot = np.array([float(macro_step) for macro_step in macro_step_list]) / float(micro_step)
        rho_plot = [utils.formatTime(rho_) for rho_ in rho_plot]

        """ Create figure for specific field """
        data_dict = {}

        field_data = []
        for macro_step in macro_step_list:
            # print(micro_step)
            # print(macro_step)
            macro_steps_per_round, micro_steps_per_round = getMultiscaleParams(macro_step, micro_step, prediction_horizon)

            # print(macro_steps_per_round)
            # print(micro_steps_per_round)

            total_macro_steps = np.sum(macro_steps_per_round)
            total_micro_steps = np.sum(micro_steps_per_round)

            # print(total_macro_steps)
            # print(total_micro_steps)

            time_rho = (total_macro_steps * time_macrodynamics + total_micro_steps * time_microdynamics) / (total_macro_steps + total_micro_steps)

            speed_up_rho = float(time_microdynamics) / time_rho
            field_data.append(speed_up_rho)


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






