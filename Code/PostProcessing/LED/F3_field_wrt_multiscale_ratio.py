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
        "NAD": "MNAD",
        "RMSE": "RMNSE",
        "MSE": "MSE",
        "CORR": "Correlation",
        "mnad_act": "MNAD$(u, \\tilde{u})$",
        "mnad_in": "MNAD$(v, \\tilde{v})$",
        "state_dist_L1_hist_error": "L1-NHD",
        "state_dist_wasserstein_distance": "WD",
        "drag_coef_error_rel_all":"$|C_d-\\tilde{C_d}|/|\\tilde{C_d}|$",
        # "NRMSE": "NRMSE",
        "NRMSE": "MNAD",
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

# Experiment_Name="Experiment_Daint_Large"
# Experiment_Name="Experiment_Barry"
# Experiment_Name=None


system2Model = {
    # "cylRe100HR":"GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",
    # "cylRe1000HR":"GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",
    # "cylRe100HRDt005":"GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_4-OPT_adabelief-C_lstm-R_1x16-SL_10-RS_17-LR_0.01",
    # "cylRe1000HRDt005":"GPU-RNN-DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_10-OPT_adabelief-C_lstm-R_1x64-SL_10-RS_17-LR_0.01",
    "cylRe100HR":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
    # "cylRe1000HR":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_10-LFO_0-LFL_1", # WRONG SUBOPTIMAL RESULT
    "cylRe1000HR":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
    "cylRe100HRDt005":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x64-SL_10-LFO_0-LFL_1",
    "cylRe1000HRDt005":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1", 
}

# "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",


if system_name == "FHN":

    # model_names = \
    # [
    # "ARNN-scaler_MinMaxZeroOne-OPT_adam-LR_0.001-NL_0.0-L2_0.0-RS_114-CHANNELS_2-8-16-32-4-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_6-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_0-LFL_1" \
    # ]
    # labels = \
    # [
    # "CNN-LSTM",
    # ]

    # micro_step = 10
    # macro_step_list = [40, 80, 1000]



    model_names = \
    [
    "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1",
    # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_40-LFO_0-LFL_1",
    # # "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_1.0-REG_1e-05-NS_10",
    # "GPU-CNN-RC-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-SOLVER_pinv-SIZE_1000-DEG_10-R_0.99-S_1.0-REG_0.001-NS_10",
    # "GPU-CNN-SINDy-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-TYPE_continuous-PO_3-THRES_1e-05-LIB_poly-INT_5",
    # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_mlp-R_3x100-SL_25-LFO_1-LFL_1",
    ]
    labels = \
    [
    "CNN-LSTM-end2end",
    # "CNN-LSTM",
    # "CNN-RC",
    # "CNN-SINDy",
    # "CNN-MLP",
    ]
    color_idx = 0

    micro_step = 10
    macro_step_list = [10, 50, 100, 200, 1000, 9000]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["MSE", "mnad_act", "mnad_in"]
    fields2plotLimBottom = [True, True, True]

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

    micro_step = 32
    macro_step_list = [0, 16, 32, 64, 128, 256, 320, 512, 640, 3400]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["MSE", "RMSE", "NAD", "state_dist_L1_hist_error", "state_dist_wasserstein_distance"]
    fields2plotLimBottom = [True, True, True, True, True]

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

    micro_step = 16
    macro_step_list = [4, 8, 16, 32, 64, 128, 256]

    # hyper_params_dict["multiscale_micro_steps"]     = [4, 8, 16, 32]
    # multiscale_macro_step_list_list=[0, 4, 8, 16, 32, 64, 128, 256]

    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["MSE", "RMSE", "NAD", "CORR", "state_dist_L1_hist_error", "state_dist_wasserstein_distance"]
    fields2plotLimBottom = [True, True, True, False, True, True]

elif system_name in [
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

    # micro_step = 5
    # macro_step_list = [0, 5, 10, 15, 20, 25, 50, 150]
    macro_step_list = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 64, 150]
    micro_step_list = [1, 2, 4, 8, 16]

    if system_name == "cylRe100HR":
        macro_step_list = [2, 4, 6, 10, 20, 150]
        micro_step_list = [2]

    if system_name == "cylRe1000HR":
        macro_step_list = [4, 8, 16, 32, 64, 150]
        micro_step_list = [4]
        
    dt = np.array([np.loadtxt(global_params.data_path_gen.format(system_name) + "/dt.txt")])[0]

    fields2plot = ["NRMSE", "drag_coef_error_rel_all"]
    fields2plotLimBottom = [False, False]

else:
    raise ValueError("system_name={:} not found.".format(system_name))


saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)

print(system_name)

if MAKE_FIGURE:
    FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
    os.makedirs(FIGURES_PATH, exist_ok=True)


DATA_PATH = "./{:}/{:}/Data".format(system_name, Experiment_Name)
os.makedirs(DATA_PATH, exist_ok=True)


# widths
widths = 0.8 / len(model_names)



# field2plot = fields2plot[0]

for micro_step in micro_step_list:

    rho_plot = np.array([float(macro_step) for macro_step in macro_step_list]) / float(micro_step)
    rho_plot = [utils.formatTime(rho_) for rho_ in rho_plot]

    for field2plotNr in range(len(fields2plot)):

        field2plot = fields2plot[field2plotNr]
        field2plotLimBottom = fields2plotLimBottom[field2plotNr]

        print(field2plot)

        plot_name_base = "F3_{:}_wrt_rho_micro{:}".format(field2plot, micro_step)
        """ Create figure for specific field """

        data_dict = {}

        if SEARCH_FOR_DATA:

            for modelnum in range(len(model_names)):
                modelname = model_names[modelnum]
                print(modelname)
                
                field_data = []
                for macro_step in macro_step_list:

                    filename = "results_multiscale_forecasting_micro_{:}_macro_{:}_test.pickle".format(micro_step, macro_step)

                    result_path = saving_path + os.sep + "/Evaluation_Data" + os.sep + modelname + os.sep + filename

                    assert os.path.isfile(result_path), "File {:} not found".format(result_path)



                    data_result = utils.loadDataPickle(result_path)

                    assert field2plot in data_result, "field {:} not found in {:}.".format(field2plot, result_path)
                    data = data_result[field2plot]
                    field_data.append(data)
                    
                data = np.array(field_data)

                print(np.shape(data))

                if len(np.shape(data))==3:
                    # Time is included in the metric
                    assert len(np.shape(data))==3
                    num_macro_runs, num_ics, prediction_horizon = np.shape(data)
                    # Mean over time
                    data = np.mean(data, axis=(2))

                elif len(np.shape(data))==2:
                    prediction_horizon = np.shape(data_result["MSE"])[1]
                    num_macro_runs, num_ics = np.shape(data)

                else:
                    raise ValueError("Not implemented.")

                # Mean over initial conditions
                data_mean = np.mean(data, axis=(1))
                # min and max over initial conditions
                data_min = np.min(data, axis=(1))
                data_max = np.max(data, axis=(1))

                x_pos = 1.0 * np.arange(len(macro_step_list))
                y_data = data_mean
                y_data_range = data_max - data_min

                labels_plot = []
                for case in range(len(macro_step_list)):
                    macro_step = macro_step_list[case]
                    if macro_step < prediction_horizon:
                        labels_plot.append(rho_plot[case])
                    else:
                        labels_plot.append(str("Latent"))
                        x_pos[case] += 0.3


                data_dict.update({
                                 modelname:{
                                 "data":data.T,
                                 "labels_plot":labels_plot,
                                 "x_pos":x_pos,
                                 "y_data":y_data,
                                 "y_data_min":data_min,
                                 "y_data_max":data_max,
                                 "y_data_range":y_data_range,
                                 "label_model":labels[modelnum],
                                 "color":color_labels[color_idx],
                                 "marker":linemarkers[modelnum],
                                 "markerwidth":linemarkerswidth[modelnum],
                                 }})

            data_dict.update({
                             "misc":{
                             "prediction_horizon":prediction_horizon,
                             "labels_plot":labels_plot,
                             "x_pos":x_pos,
                             }})

            utils.saveDataPickle(data_dict, DATA_PATH + "/{:}".format(plot_name_base), add_file_format=True)

        else:
            data_dict_load = utils.loadDataPickle(DATA_PATH + "/{:}".format(plot_name_base), add_file_format=True)
            data_dict.update(data_dict_load)



        if MAKE_FIGURE:

            for legend_str in legend_str_list:
                fig_path = FIGURES_PATH + "/{:}_{:}.pdf".format(plot_name_base, legend_str)
                fig, ax = plt.subplots()
                iter_ = 0
                for modelname in model_names:

                    violin_parts = ax.violinplot(data_dict[modelname]["data"],
                                  positions =data_dict[modelname]["x_pos"] + 0.1 * iter_,
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
                                ls='none',
                                )

                    for violin_part_key in violin_parts.keys():
                        violin_part_list = violin_parts[violin_part_key]
                        if isinstance(violin_part_list, list):
                            for violin_part in violin_part_list:
                                violin_part.set_facecolor(data_dict[modelname]["color"])
                                violin_part.set_edgecolor(data_dict[modelname]["color"])
                        else:
                            violin_part = violin_part_list
                            violin_part.set_facecolor(data_dict[modelname]["color"])
                            violin_part.set_edgecolor(data_dict[modelname]["color"])

                    # for pc in violin_parts['bodies']:
                    #     pc.set_facecolor(data_dict[modelname]["color"])
                    #     pc.set_edgecolor(data_dict[modelname]["color"])

                    # ax.errorbar(data_dict[modelname]["x_pos"] + 0.1 * iter_,
                    #             data_dict[modelname]["y_data"],
                    #             yerr=[data_dict[modelname]["y_data"]-data_dict[modelname]["y_data_min"], data_dict[modelname]["y_data_max"]-data_dict[modelname]["y_data"]],
                    #             # alpha=0.5,
                    #             color=data_dict[modelname]["color"],
                    #             marker=data_dict[modelname]["marker"],
                    #             label=data_dict[modelname]["label_model"],
                    #             markeredgewidth=data_dict[modelname]["markerwidth"],
                    #             linewidth=2,
                    #             ls='none',
                    #             )
                    iter_ += 1

                ax.set_xticks(data_dict["misc"]["x_pos"])
                ax.set_xticklabels(data_dict["misc"]["labels_plot"])

                ax.yaxis.grid(True)

                ylabel = field2Label[field2plot]
                plt.ylabel(r"{:}".format(ylabel))

                if field2plotLimBottom: plt.ylim(bottom=0)

                plt.xlabel(r"$\rho = T_m / T_{\mu}$")

                tm = utils.formatTime(micro_step * dt)
                title_str = "$T_{\mu}=" + "{:}".format(tm)

                tf = utils.formatTime(data_dict["misc"]["prediction_horizon"] * dt)
                title_str += ", \, T_{f}=" + "{:}".format(tf) + "$"

                plt.title(r"{:}".format(title_str), pad=20)
                if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                # plt.show()
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()




