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

from Codebase.Utils import utils_data


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

# FIGTYPE="png"
FIGTYPE="pdf"


system2Model = {
    "cylRe100HR":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_4-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
    "cylRe1000HR":"GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_10-PRETRAIN-AE_1-RS_21-C_lstm-R_1x32-SL_25-LFO_0-LFL_1",
}



if system_name == "KSGP64L22Large":

    modelname = "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x512-SL_50-LFO_0-LFL_1"

    filename_iter =  "results_iterative_latent_forecasting_test.pickle"
    filename_tf =  "results_teacher_forcing_forecasting_test.pickle"

    is_structured = False

elif system_name in ["cylRe100HR", "cylRe1000HR"]:

    modelname = system2Model[system_name] 

    filename_iter =  "results_iterative_latent_forecasting_test.pickle"
    filename_tf =  "results_teacher_forcing_forecasting_test.pickle"

    is_structured = True

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



plot_name_data = "F9_latent_dynamics.pdf"

data_dict = {}

if SEARCH_FOR_DATA:



    result_path = saving_path + os.sep + "/Evaluation_Data" + os.sep + modelname + os.sep + filename_iter
    with open(result_path, "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        data = pickle.load(file)
        # for key in data: print(key)
        latent_states_all_iter = np.array(data["latent_states_all"])
        dt = data["dt"]
        del data


    result_path = saving_path + os.sep + "/Evaluation_Data" + os.sep + modelname + os.sep + filename_tf
    with open(result_path, "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        data = pickle.load(file)
        # for key in data: print(key)
        latent_states_all_tf = np.array(data["latent_states_all"])
        dt = data["dt"]
        del data



    if is_structured:
        latent_states_all_iter_data = []
        latent_states_all_tf_data = []
        for ic in [0]:
            latent_states_all_iter_ic = latent_states_all_iter[0]
            latent_states_all_tf_ic = latent_states_all_tf[0]

            latent_states_all_iter_ic = np.array(latent_states_all_iter_ic)
            latent_states_all_iter_ic = utils_data.getDataHDF5Fields(latent_states_all_iter_ic[0, 0], latent_states_all_iter_ic[:, 1])   

            latent_states_all_tf_ic = np.array(latent_states_all_tf_ic)
            latent_states_all_tf_ic = utils_data.getDataHDF5Fields(latent_states_all_tf_ic[0, 0], latent_states_all_tf_ic[:, 1])

            latent_states_all_iter_data.append(latent_states_all_iter_ic)
            latent_states_all_tf_data.append(latent_states_all_tf_ic)

        latent_states_all_iter_data = np.array(latent_states_all_iter_data)
        latent_states_all_tf_data = np.array(latent_states_all_tf_data)

        latent_states_all_iter = latent_states_all_iter_data
        latent_states_all_tf = latent_states_all_tf_data

    data_dict.update({
                     "latent_states_all_tf":latent_states_all_tf,
                     "latent_states_all_iter":latent_states_all_iter,
                     })

    utils.saveDataPickle(data_dict, DATA_PATH + "/{:}".format(plot_name_data), add_file_format=True)

else:
    data_dict_load = utils.loadDataPickle(DATA_PATH + "/{:}".format(plot_name_data), add_file_format=True)
    data_dict.update(data_dict_load)


def getFrequency(temp):
    # SAMPLE_RATE = 44100  # Hertz
    SAMPLE_RATE = 1  # Hertz
    DURATION = len(temp)  # Seconds

    from scipy.fft import fft, fftfreq
    # Number of samples in temp
    N = SAMPLE_RATE * DURATION
    yf = fft(temp)
    xf = fftfreq(N, 1 / SAMPLE_RATE)
    N = int(len(yf)/2)
    xf = xf[:N]
    yf = yf[:N]
    xf = xf[1:]
    yf = yf[1:]
    idx_ = np.argmax(yf)
    freq_ = xf[idx_]
    Dt = 0.2
    f = freq_ / Dt
    L = 0.075
    U = 0.15
    St = f * L / U
    DT = 1/f
    print("Frequency is ={:}".format(f))
    print("ST number is ={:}".format(St))
    print("Period is ={:}".format(DT))
    return f, St


if MAKE_FIGURE:

    latent_states_all_tf = data_dict["latent_states_all_tf"]
    latent_states_all_iter = data_dict["latent_states_all_iter"]

    print("Target:")
    f_targ, st_targ = getFrequency(latent_states_all_tf[0, :, 0])
    print("Predicted:")
    f_pred, st_pred = getFrequency(latent_states_all_iter[0, :, 0])




