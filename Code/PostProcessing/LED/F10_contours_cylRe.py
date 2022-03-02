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

from Codebase.Systems.KS import utils_processing_ks

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


FONTSIZE=26
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

if system_name in ["cylRe100"]:
    modelname = "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_20-LFO_1-LFL_1"
    filename_iter =  "results_multiscale_forecasting_micro_20_macro_2000_test.pickle"
else:
    raise ValueError("system_name={:} not found.".format(system_name))

legend_str_list = ["", "_legend"]

saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)

print(system_name)

if MAKE_FIGURE:
    FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
    os.makedirs(FIGURES_PATH, exist_ok=True)


DATA_PATH = "./{:}/{:}/Data".format(system_name, Experiment_Name)
os.makedirs(DATA_PATH, exist_ok=True)



plot_name_data = "F10_contours.pdf"

data_dict = {}


if SEARCH_FOR_DATA:



    result_path = saving_path + os.sep + "/Evaluation_Data" + os.sep + modelname + os.sep + filename_iter
    with open(result_path, "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        data = pickle.load(file)
        # for key in data: print(key)
        targets_all = np.array(data["targets_all"])
        predictions_all = np.array(data["predictions_all"])
        dt = data["dt"]
        del data

    print(np.shape(targets_all))
    print(np.shape(predictions_all))

    targ_to_plot = targets_all[-1]
    pred_to_plot = predictions_all[-1]

    targ_to_plot = utils_plotting.loadDataArray(targ_to_plot)
    pred_to_plot = utils_plotting.loadDataArray(pred_to_plot)

    print(np.shape(targ_to_plot))
    print(np.shape(pred_to_plot))


    data_dict.update({
                     "targ_to_plot":targ_to_plot,
                     "pred_to_plot":pred_to_plot,
                     "dt":dt,
                     })

    utils.saveDataPickle(data_dict, DATA_PATH + "/{:}".format(plot_name_data), add_file_format=True)

else:
    data_dict_load = utils.loadDataPickle(DATA_PATH + "/{:}".format(plot_name_data), add_file_format=True)
    data_dict.update(data_dict_load)



# if MAKE_FIGURE: 
#     FIGTYPE = "pdf"



#         mp1 = axes[0][n].imshow(target[tInarray, rgb_channel],
#                                 vmin=vmin,
#                                 vmax=vmax,
#                                 cmap=plt.get_cmap(cmap),
#                                 aspect=1.0,
#                                 interpolation=interpolation)






