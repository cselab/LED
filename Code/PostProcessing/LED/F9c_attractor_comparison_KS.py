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


if system_name == "KSGP64L22Large":

    modelname = "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_7-C_lstm-R_1x512-SL_50-LFO_0-LFL_1"

    filename_iter =  "results_iterative_latent_forecasting_test.pickle"
    filename_tf =  "results_teacher_forcing_forecasting_test.pickle"


elif system_name == "KSGP64L22":

    modelname = "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_8-PRETRAIN-AE_1-RS_13-C_lstm-R_1x256-SL_25-LFO_0-LFL_1"

    filename_iter =  "results_iterative_latent_forecasting_test.pickle"
    filename_tf =  "results_teacher_forcing_forecasting_test.pickle"

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



plot_name_data = "F9c_attractor_KS.pdf"

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

    state_dist_statistics = {}
    state_dist_statistics = utils_processing_ks.computeStateDistributionStatisticsSystemKSUxUxx(state_dist_statistics, targets_all, predictions_all, rule="rice", nbins=100)

    data_dict.update({
                     "state_dist_statistics":state_dist_statistics,
                     })

    utils.saveDataPickle(data_dict, DATA_PATH + "/{:}".format(plot_name_data), add_file_format=True)

else:
    data_dict_load = utils.loadDataPickle(DATA_PATH + "/{:}".format(plot_name_data), add_file_format=True)
    data_dict.update(data_dict_load)




if MAKE_FIGURE: 
    FIGTYPE = "pdf"



    state_dist_statistics = data_dict["state_dist_statistics"]
    ux_uxx_density_target = state_dist_statistics["ux_uxx_density_target"]
    ux_uxx_density_predicted = state_dist_statistics["ux_uxx_density_predicted"]
    ux_uxx_l1_hist_error = state_dist_statistics["ux_uxx_l1_hist_error"]
    ux_uxx_l1_hist_error_vec = state_dist_statistics["ux_uxx_l1_hist_error_vec"]
    ux_uxx_wasserstein_distance = state_dist_statistics["ux_uxx_wasserstein_distance"]
    ux_uxx_mesh = state_dist_statistics["ux_uxx_mesh"]

    with np.errstate(divide='ignore', invalid='ignore'):
        ux_uxx_density_target = np.log(ux_uxx_density_target)
        ux_uxx_density_predicted = np.log(ux_uxx_density_predicted)
        ux_uxx_l1_hist_error_vec = np.log(ux_uxx_l1_hist_error_vec)
    ncols = 2
    nrows = 1

    ux_uxx_density_target[ux_uxx_density_target<-6] = -6
    ux_uxx_density_predicted[ux_uxx_density_predicted<-6] = -6

    tick_locs = [-1, -2, -3, -4, -5]
    tick_labels = ["-1", "-2", "-3", "-4", "-5"]

    vmin = np.nanmin(ux_uxx_density_target[ux_uxx_density_target != -np.inf])
    vmax = np.nanmax(ux_uxx_density_target) 


    fig, ax = plt.subplots(figsize=(5, 4.5))
    contours_vec = []
    ax.set_title("Reference Density", pad=20)
    mp = ax.contourf(ux_uxx_mesh[0],
                             ux_uxx_mesh[1],
                             ux_uxx_density_target.T,
                             60,
                             cmap=plt.get_cmap("Reds"),
                             levels=np.linspace(vmin, vmax, 60),
                             )
    bar = fig.colorbar(mp, ax=ax)
    contours_vec.append(mp)
    ax.set_ylabel(r"$u_{xx}$")
    ax.set_xlabel(r"$u_{x}$")
    bar.locator     = matplotlib.ticker.FixedLocator(tick_locs)
    bar.formatter   = matplotlib.ticker.FixedFormatter(tick_labels)
    bar.update_ticks()
    for contours in contours_vec:
        for pathcoll in contours.collections:
            pathcoll.set_rasterized(True)
    fig.tight_layout()
    fig_path = FIGURES_PATH + "/F9c_KS_attractor_target.{:}".format(FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()



    fig, ax = plt.subplots(figsize=(5, 4.5))
    contours_vec = []
    ax.set_title("Predicted Density", pad=20)
    mp = ax.contourf(ux_uxx_mesh[0],
                    ux_uxx_mesh[1],
                    ux_uxx_density_predicted.T,
                    60,
                    cmap=plt.get_cmap("Reds"),
                    levels=np.linspace(vmin, vmax, 60),
                    )
    bar = fig.colorbar(mp, ax=ax)
    contours_vec.append(mp)
    ax.set_ylabel(r"$u_{xx}$")
    ax.set_xlabel(r"$u_{x}$")
    bar.locator     = matplotlib.ticker.FixedLocator(tick_locs)
    bar.formatter   = matplotlib.ticker.FixedFormatter(tick_labels)
    bar.update_ticks()
    for contours in contours_vec:
        for pathcoll in contours.collections:
            pathcoll.set_rasterized(True)
    fig.tight_layout()
    fig_path = FIGURES_PATH + "/F9c_KS_attractor_predicted.{:}".format(FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()
 




    # fig, axes = plt.subplots(nrows=nrows,
    #                          ncols=ncols,
    #                          figsize=(5 * ncols, 4 * nrows),
    #                          squeeze=False)


    # contours_vec = []
    # axes[0, 0].set_title("Reference Density")
    # mp = axes[0, 0].contourf(ux_uxx_mesh[0],
    #                          ux_uxx_mesh[1],
    #                          ux_uxx_density_target.T,
    #                          60,
    #                          cmap=plt.get_cmap("Reds"),
    #                          levels=np.linspace(vmin, vmax, 60),
    #                          )

    # bar = fig.colorbar(mp, ax=axes[0, 0])
    # contours_vec.append(mp)
    # axes[0, 0].set_ylabel(r"$u_{xx}$")
    # axes[0, 0].set_xlabel(r"$u_{x}$")

    # bar.locator     = matplotlib.ticker.FixedLocator(tick_locs)
    # bar.formatter   = matplotlib.ticker.FixedFormatter(tick_labels)
    # bar.update_ticks()


    # axes[0, 1].set_title("Predicted Density")
    # mp = axes[0, 1].contourf(ux_uxx_mesh[0],
    #                          ux_uxx_mesh[1],
    #                          ux_uxx_density_predicted.T,
    #                          60,
    #                          cmap=plt.get_cmap("Reds"),
    #                          levels=np.linspace(vmin, vmax, 60),
    #                          )
    # bar = fig.colorbar(mp, ax=axes[0, 1])
    # contours_vec.append(mp)
    # axes[0, 1].set_ylabel(r"$u_{xx}$")
    # axes[0, 1].set_xlabel(r"$u_{x}$")

    # bar.locator     = matplotlib.ticker.FixedLocator(tick_locs)
    # bar.formatter   = matplotlib.ticker.FixedFormatter(tick_labels)
    # bar.update_ticks()

    # for contours in contours_vec:
    #     for pathcoll in contours.collections:
    #         pathcoll.set_rasterized(True)

    # fig.tight_layout()

    # plt.savefig(fig_path, dpi=300)
    # plt.close()






