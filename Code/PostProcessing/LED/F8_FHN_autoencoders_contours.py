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


# FONTSIZE=26
FONTSIZE=28
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

filename =  "results_iterative_latent_forecasting_test.pickle"
modelname = "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1"

result_path = "./FHN/FHN_result_prediction_ic1_horizon451/Evaluation_Data"
datadir = result_path + "/{:}/{:}".format(modelname, filename)


with open(datadir, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    targets_all = np.array(data["targets_all"])
    predictions_all = np.array(data["predictions_all"])
    rho_act_all_pred = predictions_all[:,:,0]
    rho_in_all_pred = predictions_all[:,:,1]
    rho_act_all_target = targets_all[:,:,0]
    rho_in_all_target = targets_all[:,:,1]
    dt = data["dt"]
    del data

T_end = np.shape(rho_act_all_pred)[1]

N = 100
L = 20
x = np.linspace(0, L, N+1)
dx = x[1]-x[0]

t_vec = np.arange(T_end) * dt


for ic in range(len(rho_act_all_pred)):
    fig_path_base = FIGURES_PATH + "/F8_FHN_ic{:}".format(ic)

    rho_act_pred = rho_act_all_pred[ic]
    rho_act_target = rho_act_all_target[ic]
    rho_in_pred = rho_in_all_pred[ic]
    rho_in_target = rho_in_all_target[ic]

    X = x
    Y = t_vec

    X, Y = np.meshgrid(X, Y)

    error = np.abs(rho_act_target-rho_act_pred)
    data = [rho_act_target, rho_act_pred, error]
    vmin = np.min([np.min(rho_act_target), np.min(rho_act_pred)])
    vmax = np.max([np.max(rho_act_target), np.max(rho_act_pred)])
    vmin_err = np.min(error)
    vmax_err = np.max(error)
    labels = ["$u(x,t)$", "$\\tilde{u}(x,t)$", "$|u(x,t) - \\tilde{u}(x,t)|$"]
    figname = "ACT"

    # error = np.abs(rho_in_pred-rho_in_target)
    # data = [rho_in_target, rho_in_pred, error]
    # vmin = np.min([np.min(rho_in_target), np.min(rho_in_pred)])
    # vmax = np.max([np.max(rho_in_target), np.max(rho_in_pred)])
    # vmin_err = np.min(error)
    # vmax_err = np.max(error)
    # labels = ["$v(x,t)$", "$\\tilde{v}(x,t)$", "$|v(x,t) - \\tilde{v}(x,t)|$"]
    # figname = "IN"


    titles = ["Reference", "Prediction", "Absolute Error"]
    cmaps = [cm.coolwarm, cm.coolwarm, cm.Reds]
    mins = [vmin, vmin, vmin_err]
    maxs = [vmax, vmax, vmax_err]

    # fig = plt.figure(figsize=(20,6))
    # for i in range(len(data)):
    #     Z = data[i]
    #     title = titles[i]
    #     label = labels[i]
    #     ax = fig.add_subplot(1, 3, i+1, projection='3d')
    #     print(np.shape(X))
    #     print(np.shape(Y))
    #     print(np.shape(Z))
    #     # Plot the surface.
    #     surf = ax.plot_surface(X, Y, Z, cmap=cmaps[i], rasterized=True)
    #     ax.set_xlabel(r"$x$", labelpad=20)
    #     ax.set_ylabel(r"$t$", labelpad=20)
    #     ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    #     ax.set_zlabel(r"{:}".format(label), rotation=0, labelpad=20 if i<2 else 50)
    #     # Add a color bar which maps values to colors.
    #     fig.colorbar(surf, orientation="horizontal")
    #     ax.invert_xaxis()
    #     ax.view_init(elev=34., azim=-48.)
    #     ax.set_title(r"{:}".format(title), pad=20)
    #     plt.subplots_adjust(wspace=0.3, left=0.01, right=0.85)

    # # plt.tight_layout()
    # plt.savefig(fig_path_base + "_surface_{:}.pdf".format(figname))
    # # plt.show()
    # plt.close()

    # from matplotlib.ticker import FuncFormatter
    # fmt = lambda x, pos: '{:.2f}'.format(x)

    # fig = plt.figure(figsize=(20,6))
    fig = plt.figure(figsize=(6,15))
    # fig = plt.figure()
    titles = ["Reference", "Prediction", "Absolute Error"]
    cmaps = [plt.get_cmap("seismic"), plt.get_cmap("seismic"), cm.Reds]
    for i in range(len(data)):
        Z = data[i]
        title = titles[i]
        label = labels[i]
        # ax = fig.add_subplot(1, 3, i+1)
        ax = fig.add_subplot(3, 1, i+1)
        print(np.shape(X))
        print(np.shape(Y))
        print(np.shape(Z))
        mp = ax.contourf(X, Y, Z, 100, cmap=cmaps[i],zorder=-9, levels=np.linspace(mins[i], maxs[i], 100))
        # mp = ax.contourf(X, Y, Z, 100, cmap=cmaps[i], levels=np.linspace(mins[i], maxs[i], 10))
        ax.set_xlabel(r"$x$", labelpad=20)
        ax.set_ylabel(r"$t$", labelpad=20)
        # cbar = fig.colorbar(mp, format=FuncFormatter(fmt))
        # cbar = fig.colorbar(mp, format='%.0e')
        
        if title == "Absolute Error":
            cbar = fig.colorbar(mp, format='%.0e')
        else:
            cbar = fig.colorbar(mp, format='%.2f')

        cbar.ax.locator_params(nbins=5)

        # cbar = plt.colorbar()
        # cbar.formatter.set_powerlimits((0, 0))

        plt.gca().set_rasterization_zorder(-1)
        ax.set_title(r"{:}".format(label), pad=20)

    plt.tight_layout()
    plt.savefig(fig_path_base + "_contour_{:}.pdf".format(figname))
    plt.close()

