#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np
import os

import socket
import matplotlib

hostname = socket.gethostname()
print("[utils_cylRe_plotting] PLOTTING HOSTNAME: {:}".format(hostname))
""" check if running on a cluster """
CLUSTER = True if (
                   (hostname[:len('local')]=='local') \
                    or (hostname[:len('eu')]=='eu') \
                    or (hostname[:len('daint')]=='daint') \
                    or (hostname[:len('barry')]=='barry') \
                    or (hostname[:len('barrycontainer')]=='barrycontainer') \
                    or (hostname[:len('nid')]=='nid') \
                    ) else False

if (hostname[:len('local')] == 'local'):
    CLUSTER_NAME = "local"
elif (hostname[:len('daint')] == 'daint') or (hostname[:len('nid')] == 'nid'):
    CLUSTER_NAME = "daint"
elif (hostname[:len('barrycontainer')]
      == 'barrycontainer') or (hostname[:len('barry')] == 'barry'):
    CLUSTER_NAME = "barry"
else:
    CLUSTER_NAME = "local"

print("[utils_cylRe_plotting] CLUSTER={:}, CLUSTER_NAME={:}".format(
    CLUSTER, CLUSTER_NAME))

if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

FONTSIZE = 18
font = {'size': FONTSIZE, 'family': 'Times New Roman'}
matplotlib.rc('xtick', labelsize=FONTSIZE)
matplotlib.rc('ytick', labelsize=FONTSIZE)
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

if CLUSTER_NAME in ["local", "barry"]:
    # Plotting parameters
    rc('text', usetex=True)
    plt.rcParams["text.usetex"] = True
    plt.rcParams['xtick.major.pad'] = '10'
    plt.rcParams['ytick.major.pad'] = '10'

from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ... import Utils as utils

# FIGTYPE = "pdf"
FIGTYPE = "png"


def removeTicksFromPlots(ax):
    ax.axis('off')
    ax.tick_params(axis='both',
                   left='off',
                   top='off',
                   right='off',
                   bottom='off',
                   labelleft='off',
                   labeltop='off',
                   labelright='off',
                   labelbottom='off')
    return ax


def addShape(model, ax):
    # Adding the characteristic function
    if "chi" in model.data_info_dict["sim_micro_data"].keys():
        chi = model.data_info_dict["sim_micro_data"]["chi_sub"][0, :, :, 0]
        chi = np.ma.masked_where(chi < 0.9, chi)
        ax.imshow(chi,
                  aspect=1.0,
                  cmap=plt.get_cmap("Greys"),
                  vmin=0.0,
                  vmax=1.0,
                  interpolation="bilinear",
                  )
    return ax


def plotDrugCoefficient(model, results, set_name, testing_mode):
    drag_coef_targ = results["drag_coef_targ"]
    drag_coef_pred = results["drag_coef_pred"]
    dt = results["dt"]

    num_ics, T = np.shape(drag_coef_targ)

    for ic_idx in range(num_ics):
        Cd_target = drag_coef_targ[ic_idx]
        Cd_pred = drag_coef_pred[ic_idx]
        Cd_error = np.abs(Cd_target - Cd_pred)

        fig_path = utils.getFigureDir(
            model) + "/{:}_{:}_{:}_Drag_Coefficient.{:}".format(
                testing_mode, set_name, ic_idx, FIGTYPE)
        plt.plot(np.arange(np.shape(Cd_target)[0]) * dt,
                 Cd_target[:],
                 linewidth=3.0,
                 label='Reference',
                 color="tab:green")
        plt.plot(np.arange(np.shape(Cd_pred)[0]) * dt,
                 Cd_pred[:],
                 linewidth=3.0,
                 label='Prediction',
                 color="tab:blue")
        plt.plot(np.arange(np.shape(Cd_error)[0]) * dt,
                 Cd_error[:],
                 linewidth=3.0,
                 label='Error',
                 color="tab:red")
        plt.xlabel(r"Time $t$")
        plt.ylabel(r"Drag coefficient $C_{d}$")
        plt.ylim(bottom=0.0)
        plt.legend(loc="upper center",
                   bbox_to_anchor=(0.5, 1.1),
                   borderaxespad=0.,
                   ncol=3,
                   frameon=False)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()


def plotSystemCUP2D(model, results, set_name, testing_mode):
    if "autoencoder" in testing_mode:
        targets_all = results["inputs_all"]
        predictions_all = results["outputs_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    latent_states_all = results["latent_states_all"]

    dt = results["dt"]





    ic_idx = 0


    target = targets_all[ic_idx]
    prediction = predictions_all[ic_idx]

    target = np.array(target)
    prediction = np.array(prediction)

    # frames_to_plot = [10]

    # target = np.array(target)[frames_to_plot]
    # prediction = np.array(prediction)[frames_to_plot]

    # target = utils.correctStructuredDataPaths(model, target)
    # prediction = utils.correctStructuredDataPaths(model, prediction)

    # if model.data_info_dict["structured"]:
    #     target = utils.getDataHDF5Fields(target[0, 0], target[:, 1])
    #     prediction = utils.getDataHDF5Fields(prediction[0, 0], prediction[:, 1])


    # print(np.shape(target))
    # print(np.shape(prediction))


    T = len(target)

    rgb_channel = 3

    data_max = model.data_info_dict["data_max"]
    data_min = model.data_info_dict["data_min"]
    data_std = model.data_info_dict["data_std"]
    data_range = data_max - data_min
    data_range = data_range[rgb_channel]
    data_norm = np.power(data_range, 2.0)


    fields = ["prediction", "reference", "NNAD"]

    for field in fields:
        video_folder = "{:}_{:}_C{:}_IC{:}_{:}".format(
            testing_mode, set_name, rgb_channel, ic_idx, field)
        n_frames_max, frame_path_python, frame_path_bash, video_path = utils.makeVideoPaths(
            model, video_folder, field=field)

        n_frames = np.min([n_frames_max, T])
        # n_frames = 2

        for t in range(n_frames):
            fig_path = frame_path_python.format(t)
            fig, axes = plt.subplots()
            if model.data_info_dict["structured"]:
                target = utils.correctStructuredDataPaths(model, target)
                prediction = utils.correctStructuredDataPaths(model, prediction)

                target_plot = utils.getDataHDF5Field(
                    target[t, 0], target[t, 1])
                target_plot = target_plot[rgb_channel]

                prediction_plot = utils.getDataHDF5Field(
                    prediction[t, 0], prediction[t, 1])
                prediction_plot = prediction_plot[rgb_channel]

            else:
                target_plot = target[t, rgb_channel]
                prediction_plot = prediction[t, rgb_channel]

            # # For Re=100
            # percent = 0.2

            # For Re=1000
            percent = 0.5
            
            if field == "reference":
                data_plot = target_plot
                vmin = data_min[rgb_channel]
                vmax = data_max[rgb_channel]
                vrange = vmax - vmin
                vmean = 0.0
                vmax = vmean + percent * vrange
                vmin = vmean - percent * vrange

                cmap =  plt.get_cmap("seismic")

            elif field == "prediction":
                data_plot = prediction_plot

                vmin = data_min[rgb_channel]
                vmax = data_max[rgb_channel]
                vrange = vmax - vmin
                vmean = 0.0
                vmax = vmean + percent * vrange
                vmin = vmean - percent * vrange

                cmap = plt.get_cmap("seismic")

            elif field == "NNAD":
                data_plot = np.sqrt(np.power(target_plot - prediction_plot, 2.0) / data_norm)
                vmin = 0
                vmax = None
                cmap = plt.get_cmap("Reds")


            else:
                raise ValueError("Bug here.")

            mp1 = axes.imshow(
                data_plot,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                aspect=1.0,
                interpolation="bilinear",
            )


            axes.axis('off')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(mp1, cax=cax)

            axes = addShape(model, axes)

            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()

        utils.makeVideo(model, video_path, frame_path_bash, n_frames_max)





