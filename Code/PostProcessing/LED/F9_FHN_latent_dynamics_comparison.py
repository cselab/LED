#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import glob, os, sys
import numpy as np
import argparse
from Utils import utils
from decimal import Decimal

import socket
hostname=socket.gethostname()
print(hostname)
sys.path.append('../../Methods')
from Config.global_conf import global_params

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)


# plt.rcParams["text.usetex"] = True
# plt.rcParams['xtick.major.pad']='10'
# plt.rcParams['ytick.major.pad']='10'
# font = {'weight':'normal', 'size':16}
# plt.rc('font', **font)


FONTSIZE=26
# FONTSIZE=28
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

system_name="FHN"
Experiment_Name="Local"


FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
os.makedirs(FIGURES_PATH, exist_ok=True)
print(FIGURES_PATH)

if system_name == "FHN":
    modelname = "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1"
    result_path = "./FHN/FHN_result_prediction_ic1_horizon1000/Evaluation_Data"


set_name = "test"
filename =  "results_iterative_latent_forecasting_{:}.pickle".format(set_name)
datadir = result_path + "/{:}/{:}".format(modelname, filename)
with open(datadir, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    # for key in data: print(key)
    latent_states_all_iter = np.array(data["latent_states_all"])
    dt = data["dt"]
    del data

filename =  "results_teacher_forcing_forecasting_{:}.pickle".format(set_name)
datadir = result_path + "/{:}/{:}".format(modelname, filename)
with open(datadir, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    # for key in data: print(key)
    latent_states_all_tf = np.array(data["latent_states_all"])
    dt = data["dt"]
    del data


print(np.shape(latent_states_all_iter))
print(np.shape(latent_states_all_tf))

ic=0
latent_states_iter=latent_states_all_iter[ic]
latent_states_tf=latent_states_all_tf[ic]

latent_states_iter = latent_states_iter[200:]
latent_states_tf = latent_states_iter[200:]


linewidth=3
subsample = 1
latent_states_iter = latent_states_iter[::subsample]
latent_states_tf = latent_states_tf[::subsample]

for legend_str in ["", "_legend"]:
    fig, ax = plt.subplots(figsize=(6,6))
    # fig, ax = plt.subplots()
    fig_path = FIGURES_PATH + "/F9_FHN_ic{:}_latent_dynamics{:}.pdf".format(ic, legend_str)
    plt.title("Latent dynamics in {:}".format(set_name), pad=10)

    plt.plot(latent_states_tf[:, 0],
            latent_states_tf[:, 1],
            color=(0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
            linestyle="--",
            marker='o',
            markevery=77,
            label="Data",
            markersize=10,
            linewidth=linewidth,
            )

    plt.plot(latent_states_iter[:, 0],
            latent_states_iter[:, 1],
            color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
            linestyle=":",
            marker='x',
            markevery=44,
            markersize=10,
            label="LED",
            linewidth=linewidth,
            markeredgewidth=linewidth,
            )


    plt.xlabel(r"$z_{1}$", labelpad=20)
    plt.ylabel(r"$z_{2}$", labelpad=20)

    if legend_str=="_legend":
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


