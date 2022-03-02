
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
from Utils.utils import *

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

system_name = "cylRe1000HRLarge"
Experiment_Name = "Experiment_Daint_Large"

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

FONTSIZE=26
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


# model_names = \
# [
# "GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_10088-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_2",
# "GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_101616-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_2",
# "GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_103232-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_2",
# "GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_106464-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_2",
# "GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_1128128-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_2",
# # "GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_1256256-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_2",
# ]

model_names = \
[
"GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_94-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10",
"GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_98-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10",
"GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_916-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10",
"GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_932-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10",
"GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_964-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10",
"GPU-ARNN-SC_MinMaxZeroOne-OPT_adam-PREC_single-LR_0.0001-NL_0.0-L2_0.0-RS_9128-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10",
]



ranks = [1, 2, 4, 8, 16, 32]
# ranks = [1, 2, 4, 8, 16]

saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)

print(system_name)

FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
os.makedirs(FIGURES_PATH, exist_ok=True)


plot_name_base = "scaling_plot"

data_dict = {}


fieldlist = \
[
"model_name",
"total_training_time",
]
typelist = \
[
str,
float,
]
filename = "train.txt"


logfile_path = saving_path + os.sep + "/Logfiles"
model_list, model_dict = parseModelFields(logfile_path, fieldlist, typelist, filename)

rank_array = []
speedup_array = []
time_epoch_array = []
for modelnum in range(len(model_names)):
    modelname = model_names[modelnum]
    rank_num = ranks[modelnum]

    data = model_dict[modelname]
    max_epochs = 2.
    time_epoch_array.append(data[0] / (max_epochs+1.))

    if rank_num == 1: time_rank_1 = data[0]

    speedup = time_rank_1 / data[0] 
    rank_array.append(rank_num)
    speedup_array.append(speedup)

print(rank_array)
# print(speedup_array)
print(time_epoch_array)



# fig_path = FIGURES_PATH + "/stron_scaling_plot.pdf"
# fig, ax = plt.subplots()

# ax.plot(rank_array,
#         speedup_array,
#         color=color_labels[0],
#         linewidth=3,
#         marker = "x",
#         markersize = 20,
#         markeredgewidth=3,
#         )
# ax.plot(rank_array,
#         rank_array,
#         "k--",
#         linewidth=3,
#         )
# ax.plot(rank_array,
#         0.5 * np.array(rank_array),
#         "r--",
#         linewidth=3,
#         )
# ax.set_aspect(1.0)
# plt.ylabel(r"Speedup")
# plt.xlabel(r"GPU Nodes")

# plt.tight_layout()
# plt.savefig(fig_path, dpi=100)
# plt.close()




