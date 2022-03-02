#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import socket
import os
import subprocess
import warnings

# Plotting parameters
import matplotlib

hostname = socket.gethostname()
print("[utils_plotting] PLOTTING HOSTNAME: {:}".format(hostname))
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

print("[utils_plotting] CLUSTER={:}, CLUSTER_NAME={:}".format(
    CLUSTER, CLUSTER_NAME))

if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

print("[utils_plotting] Matplotlib Version = {:}".format(
    matplotlib.__version__))

from matplotlib import colors
import six

color_dict = dict(six.iteritems(colors.cnames))

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

from scipy.interpolate import interpn

from . import utils_processing
from . import utils_statistics
from . import utils_data
from . import utils_time
from . import utils_networks

# FIGTYPE = "pdf"
FIGTYPE = "png"

color_labels = [
    'tab:blue',
    'tab:red',
    'tab:green',
    'tab:brown',
    'tab:orange',
    'tab:cyan',
    'tab:olive',
    'tab:pink',
    'tab:gray',
    'tab:purple',
]


def plotScheduleKLLoss(model, learning_rate_vec):

    fig_path = utils_networks.getFigureDir(
        model) + "/schedule_beta_vae_kl_loss_weight." + FIGTYPE
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.semilogy(
        np.arange(len(learning_rate_vec)),
        learning_rate_vec,
        "o-",
        color='tab:blue',
        label="LR",
        linewidth=3,
    )
    ax.set_ylabel(r"($\beta$-VAE) KL Loss Weight")
    ax.set_xlabel(r"Epoch")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plotScheduleLearningRate(model, learning_rate_vec):

    fig_path = utils_networks.getFigureDir(
        model) + "/schedule_learn_rate." + FIGTYPE
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.semilogy(
        np.arange(len(learning_rate_vec)),
        learning_rate_vec,
        "o-",
        color='tab:blue',
        label="LR",
        linewidth=3,
    )
    ax.set_ylabel(r"Learning Rate")
    ax.set_xlabel(r"Epoch")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plotScheduleLoss(model, ifp_train_vec, ifp_val_vec):
    fig_path = utils_networks.getFigureDir(model) + "/schedule_loss." + FIGTYPE
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.plot(
        np.arange(len(ifp_train_vec)),
        ifp_train_vec,
        "o-",
        color='tab:blue',
        label="train",
        linewidth=3,
    )
    plt.plot(
        np.arange(len(ifp_val_vec)),
        ifp_val_vec,
        "x-",
        color='tab:red',
        label="val",
        linewidth=3,
    )
    ax.set_ylabel(r"Iterative forecasting propability")
    ax.set_xlabel(r"Epoch")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plotAllLosses(model,
                  losses_train,
                  time_train,
                  losses_val,
                  time_val,
                  min_val_error,
                  name_str=""):
    loss_labels = model.losses_labels
    idx1 = np.nonzero(losses_train[0])[0]
    idx2 = np.nonzero(losses_train[-1])[0]
    idx = np.union1d(idx1, idx2)
    losses_val = np.array(losses_val)
    losses_train = np.array(losses_train)

    min_val_epoch = np.argmin(
        np.abs(np.array(losses_val[:, 0]) - min_val_error))
    min_val_time = time_val[min_val_epoch]

    losses_train = losses_train[:, idx]
    losses_val = losses_val[:, idx]
    loss_labels = [loss_labels[i] for i in idx]
    # Ignore first two colors
    color_labels_idx = [color_labels[i + 3] for i in idx]

    time_train = np.array(time_train)
    time_val = np.array(time_val)

    if np.all(np.array(losses_train) > 0.0) and np.all(
            np.array(losses_val) > 0.0):
        losses_train = np.log10(losses_train)
        losses_val = np.log10(losses_val)
        min_val_error_log = np.log10(min_val_error)
        if len(time_train) > 1:
            for time_str in ["", "_time"]:
                fig_path = utils_networks.getFigureDir(
                    model
                ) + "/losses_all_log" + time_str + name_str + "." + FIGTYPE
                fig, ax = plt.subplots(figsize=(20, 10))
                title = "MIN LOSS-VAL={:.4f}".format(min_val_error)
                plt.title(title)
                max_i = np.min([np.shape(losses_train)[1], len(loss_labels)])
                for i in range(max_i):
                    if time_str != "_time":
                        x_axis_train = np.arange(
                            np.shape(losses_train[:, i])[0])
                        x_axis_val = np.arange(np.shape(losses_val[:, i])[0])
                        min_val_axis = min_val_epoch
                        ax.set_xlabel(r"Epoch")
                    else:
                        dt = time_train[1] - time_train[0]
                        x_axis_train = time_train + i * dt
                        x_axis_val = time_val + i * dt
                        min_val_axis = min_val_time
                        ax.set_xlabel(r"Time")
                    plt.plot(x_axis_train,
                             losses_train[:, i],
                             color=color_labels_idx[i],
                             label=loss_labels[i] + " Train")
                    plt.plot(x_axis_val,
                             losses_val[:, i],
                             color=color_labels_idx[i],
                             label=loss_labels[i] + " Val",
                             linestyle="--")
                plt.plot(min_val_axis,
                         min_val_error_log,
                         "o",
                         color='tab:red',
                         label="optimal")
                ax.set_ylabel(r"Log${}_{10}$(Loss)")
                plt.legend(loc="upper left",
                           bbox_to_anchor=(1.05, 1),
                           borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()
    else:
        if len(time_train) > 1:
            for time_str in ["", "_time"]:
                fig_path = utils_networks.getFigureDir(
                    model
                ) + "/losses_all" + time_str + name_str + "." + FIGTYPE
                fig, ax = plt.subplots(figsize=(20, 10))
                title = "MIN LOSS-VAL={:.4f}".format(min_val_error)
                plt.title(title)
                max_i = np.min([np.shape(losses_train)[1], len(loss_labels)])
                for i in range(max_i):
                    if time_str != "_time":
                        x_axis_train = np.arange(
                            np.shape(losses_train[:, i])[0])
                        x_axis_val = np.arange(np.shape(losses_val[:, i])[0])
                        min_val_axis = min_val_epoch
                        ax.set_xlabel(r"Epoch")
                    else:
                        dt = time_train[1] - time_train[0]
                        x_axis_train = time_train + i * dt
                        x_axis_val = time_val + i * dt
                        min_val_axis = min_val_time
                        ax.set_xlabel(r"Time")
                    plt.plot(x_axis_train,
                             losses_train[:, i],
                             color=color_labels_idx[i],
                             label=loss_labels[i] + " Train")
                    plt.plot(x_axis_val,
                             losses_val[:, i],
                             color=color_labels_idx[i],
                             label=loss_labels[i] + " Val",
                             linestyle="--")
                plt.plot(min_val_axis,
                         min_val_error,
                         "o",
                         color='tab:red',
                         label="optimal")
                ax.set_ylabel(r"Loss")
                plt.legend(loc="upper left",
                           bbox_to_anchor=(1.05, 1),
                           borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()


def plotTrainingLosses(model,
                       loss_train,
                       loss_val,
                       min_val_error,
                       name_str=""):

    w, h = plt.figaspect(0.4)

    label_optimal = "Optimal {:.8f}".format(min_val_error)
    if (len(loss_train) != 0) and (len(loss_val) != 0):
        min_val_epoch = np.argmin(np.abs(np.array(loss_val) - min_val_error))
        fig_path = utils_networks.getFigureDir(
            model) + "/loss_total" + name_str + "." + FIGTYPE
        fig, ax = plt.subplots(figsize=(w, h))
        # plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(
            np.arange(np.shape(loss_train)[0]),
            loss_train,
            linewidth=2,
            color='tab:green',
            label="Train",
        )
        plt.plot(
            np.arange(np.shape(loss_val)[0]),
            loss_val,
            color='tab:blue',
            linewidth=2,
            label="Validation",
        )
        plt.plot(
            min_val_epoch,
            min_val_error,
            "o",
            color='tab:red',
            linewidth=2,
            label=label_optimal,
        )
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Loss")
        plt.legend(loc="upper left",
                   bbox_to_anchor=(1.05, 1),
                   borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        loss_train = np.array(loss_train)
        loss_val = np.array(loss_val)
        if (np.all(loss_train[~np.isnan(loss_train)] > 0.0)
                and np.all(loss_val[~np.isnan(loss_val)] > 0.0)):
            fig_path = utils_networks.getFigureDir(
                model) + "/loss_total_log" + name_str + "." + FIGTYPE
            fig, ax = plt.subplots(figsize=(w, h))
            # plt.title("Validation error {:.10f}".format(min_val_error))
            plt.plot(
                np.arange(np.shape(loss_train)[0]),
                np.log10(loss_train),
                color='tab:green',
                linewidth=2,
                label="Train",
            )
            plt.plot(
                np.arange(np.shape(loss_val)[0]),
                np.log10(loss_val),
                color='tab:blue',
                linewidth=2,
                label="Validation",
            )
            plt.plot(
                min_val_epoch,
                np.log10(min_val_error),
                "o",
                color='tab:red',
                linewidth=2,
                label=label_optimal,
            )
            ax.set_xlabel(r"Epoch")
            ax.set_ylabel(r"Log${}_{10}$(Loss)")
            plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
    else:
        print("[utils_plotting] ## Empty losses. Not printing... ##")


def loadDataArray(list_of_paths):
    data = []
    for i in range(len(list_of_paths)):
        data.append(utils_data.loadData(list_of_paths[i], "pickle"))
    data = np.array(data)
    return data


def correctStructuredDataPaths(model, array):
    # print(model.params["results_dir"])
    # print(model.params["saving_path"])
    results_dir = model.params["results_dir"]
    saving_path = model.params["saving_path"]
    for t in range(len(array)):
        path_ = array[t, 0]
        idx_ = path_.find(results_dir)
        path_ = path_[idx_:]
        path_ = saving_path + path_
        array[t, 0] = path_
    return array


def plotLatentDynamics(model,
                       set_name,
                       latent_states,
                       ic_idx,
                       testing_mode,
                       warm_up=0):

    if model.data_info_dict["structured"]:
        latent_states = np.array(latent_states)
        latent_states = correctStructuredDataPaths(model, latent_states)
        latent_states = utils_data.getDataHDF5Fields(latent_states[0, 0],
                                                     latent_states[:, 1])

    T = np.shape(latent_states)[0]
    if len(np.shape(latent_states)) > 2:
        latent_states = np.reshape(latent_states, (T, -1))
    # print(np.shape(latent_states))

    T, D = np.shape(latent_states)
    fig_path = utils_networks.getFigureDir(
        model) + "/{:}_{:}_{:}_latent_states.{:}".format(
            testing_mode, set_name, ic_idx, FIGTYPE)
    D_PLOT = np.min([D, 10])
    length = 10
    height = D_PLOT * 4
    fig, axes = plt.subplots(figsize=(length, height),
                             nrows=D_PLOT,
                             ncols=1,
                             sharex=True)
    for d in range(D_PLOT):
        data_plot = latent_states[:, d]
        ax_ = axes if D_PLOT == 1 else axes[d]
        ax_.plot(np.arange(T), data_plot, linewidth=2.0, color="tab:blue")
        label_str = "Latent state $z$" if D_PLOT == 1 else "Latent state $" + "z_" + "{:}".format(
            d) + "$"
        ax_.set_ylabel(label_str)
        if warm_up > 0:
            ax_.plot(np.ones((100, 1)) * warm_up,
                     np.linspace(np.min(data_plot), np.max(data_plot), 100),
                     'g--',
                     linewidth=2.0,
                     label='warm-up')

    ax_ = axes if D_PLOT == 1 else axes[-1]
    ax_.set_xlabel("Timestep")
    ax_ = axes if D_PLOT == 1 else axes[0]
    ax_.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Latent dynamics in {:}".format(set_name))
    if D > 1:
        plt.plot(latent_states[:, 0],
                 latent_states[:, 1],
                 "--",
                 color="tab:blue",
                 linewidth=2.0)
        plt.xlabel(r"$\mathbf{z}_{0}$")
        plt.ylabel(r"$\mathbf{z}_{1}$")
    else:
        plt.plot(latent_states[:-1, 0],
                 latent_states[1:, 0],
                 "--",
                 color="tab:blue",
                 linewidth=2.0)
        plt.xlabel(r"$\mathbf{z}_{t}$")
        plt.ylabel(r"$\mathbf{z}_{t+1}$")
    plt.tight_layout()
    fig_path = utils_networks.getFigureDir(
        model) + "/{:}_latent_space_{:}_{:}.{:}".format(
            testing_mode, set_name, ic_idx, FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # plotLatentDynamicsScatteringPlots(model, set_name, latent_states, ic_idx, testing_mode)


def plotLatentDynamicsScatteringPlots(model, set_name, latent_states, ic_idx,
                                      testing_mode):

    if np.shape(latent_states)[1] >= 2:
        # print(np.shape(latent_states))
        latent_states = np.reshape(latent_states,
                                   (np.shape(latent_states)[0], -1))
        # print(np.shape(latent_states))
        # print(ark)
        latent_dim = np.shape(latent_states)[1]
        latent_dim_max_comp = 3
        latent_dim_max_comp = np.min([latent_dim_max_comp, latent_dim])

        for idx1 in range(latent_dim_max_comp):
            for idx2 in range(idx1 + 1, latent_dim_max_comp):
                fig, ax = plt.subplots()
                plt.title("Latent dynamics in {:}".format(set_name))
                X = latent_states[:, idx1]
                Y = latent_states[:, idx2]
                # arrowplot(ax, X, Y, nArrs=100)
                scatterDensityLatentDynamicsPlot(X, Y, ax=ax)
                plt.xlabel("State {:}".format(idx1 + 1))
                plt.ylabel("State {:}".format(idx2 + 1))
                plt.tight_layout()
                fig_path = utils_networks.getFigureDir(
                    model) + "/{:}_latent_dynamics_{:}_{:}_{:}_{:}.{:}".format(
                        testing_mode, set_name, ic_idx, idx1, idx2, FIGTYPE)
                plt.savefig(fig_path, dpi=300)
                plt.close()

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        latent_tsne_results = tsne.fit_transform(latent_states)
        print(np.shape(latent_tsne_results))
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        X = latent_tsne_results[:, 0]
        Y = latent_tsne_results[:, 1]
        # arrowplot(ax, X, Y, nArrs=100)
        scatterDensityLatentDynamicsPlot(X, Y, ax=ax)
        plt.xlabel("TSNE mode {:}".format(0 + 1))
        plt.ylabel("TSNE mode {:}".format(1 + 1))
        plt.tight_layout()
        fig_path = utils_networks.getFigureDir(
            model) + "/{:}_latent_dynamics_TSNE_{:}_{:}_{:}_{:}.{:}".format(
                testing_mode, set_name, ic_idx, 0, 1, FIGTYPE)
        plt.savefig(fig_path, dpi=300)
        plt.close()

        # print(np.shape(latent_states_plot))
        pca = PCA(n_components=latent_dim_max_comp)
        pca.fit(latent_states[:5000])
        latent_states_pca = pca.transform(latent_states)
        print(np.shape(latent_states))
        for idx1 in range(latent_dim_max_comp):
            for idx2 in range(idx1 + 1, latent_dim_max_comp):
                fig, ax = plt.subplots()
                plt.title("Latent dynamics in {:}".format(set_name))
                X = latent_states_pca[:, idx1]
                Y = latent_states_pca[:, idx2]
                # arrowplot(ax, X, Y, nArrs=100)
                scatterDensityLatentDynamicsPlot(X, Y, ax=ax)
                plt.xlabel("PCA mode {:}".format(idx1 + 1))
                plt.ylabel("PCA mode {:}".format(idx2 + 1))
                plt.tight_layout()
                fig_path = utils_networks.getFigureDir(
                    model
                ) + "/{:}_latent_dynamics_PCA_{:}_{:}_{:}_{:}.{:}".format(
                    testing_mode, set_name, ic_idx, idx1, idx2, FIGTYPE)
                plt.savefig(fig_path, dpi=300)
                plt.close()
    else:
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        latent_states_plot_x = np.reshape(np.array(latent_states[:-1]), (-1))
        latent_states_plot_y = np.reshape(np.array(latent_states[1:]), (-1))
        scatterDensityLatentDynamicsPlot(latent_states_plot_x,
                                         latent_states_plot_y,
                                         ax=ax)
        plt.xlabel(r"$\mathbf{z}_{t}$")
        plt.ylabel(r"$\mathbf{z}_{t+1}$")
        plt.tight_layout()
        fig_path = model.getFigureDir(
        ) + "/{:}_latent_dynamics_{:}_{:}.{:}".format(testing_mode, set_name,
                                                      ic_idx, FIGTYPE)
        plt.savefig(fig_path, dpi=300)
        plt.close()


def plotErrorsInTime(model, results, set_name, testing_mode):

    print("[utils_plotting] # plotErrorsInTime() #")

    error_mean_dict_in_time = results["error_mean_dict_in_time"]
    error_std_dict_in_time = results["error_std_dict_in_time"]

    dt = results["dt"]

    for error_key in error_mean_dict_in_time:
        assert (error_key in error_std_dict_in_time)

        fig_path = utils_networks.getFigureDir(
            model) + "/{:}_{:}_{:}_over_time.{:}".format(
                testing_mode, set_name, error_key, FIGTYPE)
        error_mean = np.array(error_mean_dict_in_time[error_key])
        error_std = np.array(error_std_dict_in_time[error_key])

        time = np.arange(np.shape(error_mean)[0]) * dt

        fig, axes = plt.subplots(figsize=(10, 6))

        axes.plot(time, error_mean, 'b-', marker='^', label=r"$\mu$")
        plt.fill_between(time,
                         error_mean - error_std,
                         error_mean + error_std,
                         alpha=0.3,
                         label=r"+/- $\sigma$")

        plt.ylim(bottom=0.0)
        axes.legend(loc="upper left",
                    bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.)
        # axes.errorbar(time, error_mean, yerr=error_std, fmt='-o')

        axes.set_xlabel(r"Timestep")
        axes.set_ylabel(r"{:}".format(error_key))
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        # plt.show()
        plt.close()




def createIterativePredictionPlots(model, target, prediction, dt, ic_idx, set_name, \
    testing_mode="", latent_states=None, hist_data=None, wasserstein_distance_data=None, \
    warm_up=None, target_augment=None, prediction_augment=None):
    print("[utils_plotting] # createIterativePredictionPlots() #")
    # if error is not None:
    #      fig_path = utils_networks.getFigureDir(model) + "/{:}_{:}_{:}_error.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
    #      plt.plot(error, label='error')
    #      plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    #      plt.tight_layout()
    #      plt.savefig(fig_path, dpi=300)
    #      plt.close()

    #      fig_path = utils_networks.getFigureDir(model) + "/{:}_{:}_{:}_log_error.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
    #      plt.plot(np.log10(np.arange(np.shape(error)[0])), np.log10(error), label='Log${}_{10}$(Loss)')
    #      plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    #      plt.tight_layout()
    #      plt.savefig(fig_path, dpi=300)
    #      plt.close()

    if model.params["Dx"] > 0 and model.params["Dy"] > 0:
        createIterativePredictionPlotsForImageData(model, target, prediction,
                                                   dt, ic_idx, set_name,
                                                   testing_mode)

    elif len(np.shape(prediction)) == 2 or len(np.shape(prediction)) == 3:

        # if latent_states is not None:
        #     plotLatentDynamics(model, set_name, latent_states, ic_idx, testing_mode)

        if ((target_augment is not None) and (prediction_augment is not None)):
            # prediction_augment_plot = prediction_augment[:, 0] if len(
            #     np.shape(prediction_augment)
            # ) == 2 else prediction_augment[:, 0, 0] if len(
            #     np.shape(prediction_augment)) == 3 else None
            # target_augment_plot = target_augment[:, 0] if len(
            #     np.shape(
            #         target_augment)) == 2 else target_augment[:, 0, 0] if len(
            #             np.shape(target_augment)) == 3 else None

            # fig_path = utils_networks.getFigureDir(model) + "/{:}_augmented_{:}_{:}.{:}".format(testing_mode, set_name,
            #                                         ic_idx, FIGTYPE)
            # plt.plot(np.arange(np.shape(prediction_augment_plot)[0]),
            #          prediction_augment_plot,
            #          'b',
            #          linewidth=2.0,
            #          label='output')
            # plt.plot(np.arange(np.shape(target_augment_plot)[0]),
            #          target_augment_plot,
            #          'r',
            #          linewidth=2.0,
            #          label='target')
            # plt.plot(np.ones((100, 1)) * warm_up,
            #          np.linspace(np.min(target_augment_plot),
            #                      np.max(target_augment_plot), 100),
            #          'g--',
            #          linewidth=2.0,
            #          label='warm-up')
            # plt.legend(loc="upper left",
            #            bbox_to_anchor=(1.05, 1),
            #            borderaxespad=0.)
            # plt.tight_layout()
            # plt.savefig(fig_path, dpi=300)
            # plt.close()

            # prediction_plot = prediction[:, 0] if len(
            #     np.shape(prediction)) == 2 else prediction[:, 0, 0] if len(
            #         np.shape(prediction)) == 3 else None
            # target_plot = target[:, 0] if len(
            #     np.shape(target)) == 2 else target[:, 0, 0] if len(
            #         np.shape(target)) == 3 else None

            # fig_path = utils_networks.getFigureDir(model) + "/{:}_{:}_{:}.{:}".format(
            #     testing_mode, set_name, ic_idx, FIGTYPE)
            # plt.plot(prediction_plot, 'r--', label='prediction')
            # plt.plot(target_plot, 'g--', label='target')
            # plt.legend(loc="upper left",
            #            bbox_to_anchor=(1.05, 1),
            #            borderaxespad=0.)
            # plt.tight_layout()
            # plt.savefig(fig_path, dpi=300)
            # plt.close()

            # if model.input_dim >= 2:
            plotTestingContours(
                model,
                target,
                prediction,
                dt,
                ic_idx,
                set_name,
                latent_states=latent_states,
                testing_mode=testing_mode,
                hist_data=hist_data,
                wasserstein_distance_data=wasserstein_distance_data,
                with_multiscale_bar=isMultiscale(testing_mode))


def isMultiscale(testing_mode):
    if "multiscale" in testing_mode:
        return True
    else:
        return False


def createIterativePredictionPlotsForImageData(
    model,
    target,
    prediction,
    dt,
    ic_idx,
    set_name,
    testing_mode,
):
    if not model.data_info_dict["structured"]:
        assert (len(np.shape(prediction)) == 4)
    N_ = np.shape(prediction)[0]
    N_MAX = np.min([5, N_])

    # print(np.shape(prediction))
    # RGB_CHANNELS = np.shape(prediction)[1]
    # RGB_CHANNELS = 1
    RGB_CHANNELS = model.input_dim

    frames_to_plot = np.arange(N_MAX)
    for frames_to_plot in [
            np.arange(N_MAX),
            np.linspace(0, N_ - 1, num=5).astype(int)
    ]:

        for rgb_channel in range(RGB_CHANNELS):
            createIterativePredictionPlotsForImageData_(
                model, target, prediction, dt, ic_idx, set_name, testing_mode,
                rgb_channel, frames_to_plot)

    rgb_channel = RGB_CHANNELS-1
    createIterativePredictionVideoForImageData(model, target, prediction, dt,
                                               ic_idx, set_name, testing_mode,
                                               rgb_channel, N_)


def getColormap(vmin, vmax, cmap):
    if vmin < 0.0 and vmax > 0.0:
        dd = np.max([np.abs(vmin), np.abs(vmax)])
        vmin = -dd
        vmax = dd
        cmap = "seismic"
        return vmin, vmax, cmap
    else:
        return vmin, vmax, cmap


def createIterativePredictionPlotsForImageData_(
    model,
    target,
    prediction,
    dt,
    ic_idx,
    set_name,
    testing_mode,
    rgb_channel,
    frames_to_plot,
):

    # IMAGE DATA
    if not model.data_info_dict["structured"]:
        assert (len(np.shape(prediction)) == 4)

    # rgb_channel = 0
    interpolation = "bilinear"

    # print(frames_to_plot)

    N_MAX = len(frames_to_plot)
    frames_to_plot_list = "".join(
        ["T{:}-".format(str(frame)) for frame in frames_to_plot])[:-1]
    fig_path = utils_networks.getFigureDir(
        model) + "/{:}_{:}_{:}_C{:}_{:}.{:}".format(
            testing_mode, set_name, ic_idx, rgb_channel, frames_to_plot_list,
            FIGTYPE)

    # fig, axes = plt.subplots(nrows=4, ncols=N_MAX+1,figsize=(16, 12))

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig = plt.figure(constrained_layout=True, figsize=(16, 12))
    widths = N_MAX * [5] + [1]
    heights = 4 * [1]
    # print(widths)
    # print(heights)
    ncols = N_MAX + 1
    nrows = 4
    assert (nrows == len(heights))
    assert (ncols == len(widths))
    spec = fig.add_gridspec(ncols=ncols,
                            nrows=nrows,
                            width_ratios=widths,
                            height_ratios=heights)

    # for row in range(nrows):
    #     for col in range(ncols):
    #         ax = fig.add_subplot(spec[row, col])
    #         label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
    #         ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center', fontsize=1)

    if model.data_info_dict["structured"]:
        frames_to_plot_in_data = frames_to_plot

        prediction = np.array(prediction)[frames_to_plot]
        target = np.array(target)[frames_to_plot]

        prediction = correctStructuredDataPaths(model, prediction)
        target = correctStructuredDataPaths(model, target)

        target = utils_data.getDataHDF5Fields(target[0, 0], target[:, 1])
        prediction = utils_data.getDataHDF5Fields(prediction[0, 0],
                                                  prediction[:, 1])

        frames_to_plot = np.arange(len(target))
    else:
        frames_to_plot_in_data = frames_to_plot

    vmin = target[:, rgb_channel].min()
    vmax = target[:, rgb_channel].max()

    abserror = np.abs(target - prediction)
    # relerror = abserror / (1e-6 + np.abs(target))
    with np.errstate(divide='ignore', invalid='ignore'):
        relerror = abserror / (np.abs(target) + 1e-6)
    relerror[relerror > 100.0] = 100.0

    vmax_rel = relerror[:, rgb_channel].max()
    vmaxerror = abserror[:, rgb_channel].max()

    axes = []
    for row in range(nrows):
        axes_row = []
        for col in range(ncols):
            # col is the timestep
            ax = fig.add_subplot(spec[row, col])
            axes_row.append(ax)
        axes.append(axes_row)

    vmin, vmax, cmap = getColormap(vmin, vmax,
                                   model.data_info_dict["colormap"])

    # print(len(axes))
    for n in range(N_MAX):
        tInarray = frames_to_plot[n]
        tInData = frames_to_plot_in_data[n]

        mp1 = axes[0][n].imshow(target[tInarray, rgb_channel],
                                vmin=vmin,
                                vmax=vmax,
                                cmap=plt.get_cmap(cmap),
                                aspect=1.0,
                                interpolation=interpolation)
        mp2 = axes[1][n].imshow(prediction[tInarray, rgb_channel],
                                vmin=vmin,
                                vmax=vmax,
                                cmap=plt.get_cmap(cmap),
                                aspect=1.0,
                                interpolation=interpolation)
        mp3 = axes[2][n].imshow(abserror[tInarray, rgb_channel],
                                vmin=0.0,
                                vmax=vmaxerror,
                                cmap=plt.get_cmap("Reds"),
                                aspect=1.0,
                                interpolation=interpolation)
        mp4 = axes[3][n].imshow(relerror[tInarray, rgb_channel],
                                vmin=0,
                                vmax=vmax_rel,
                                cmap=plt.get_cmap("Reds"),
                                aspect=1.0,
                                interpolation=interpolation)

        axes[0][n].set_title("Target T={:}".format(tInData))
        axes[1][n].set_title("Prediction T={:}".format(tInData))
        axes[2][n].set_title("Absolute error T={:}".format(tInData))
        axes[3][n].set_title("Relative error T={:}".format(tInData))

    # fig.subplots_adjust(hspace=0.4, wspace = 0.4)

    cbar = plt.colorbar(mp1, cax=axes[0][N_MAX], fraction=0.046, pad=0.04)
    cbar = plt.colorbar(mp2, cax=axes[1][N_MAX], fraction=0.046, pad=0.04)
    cbar = plt.colorbar(mp3, cax=axes[2][N_MAX], fraction=0.046, pad=0.04)
    cbar = plt.colorbar(mp4, cax=axes[3][N_MAX], fraction=0.046, pad=0.04)

    # cbar=fig.colorbar(mp1, cax=axes[0, -1], format='%.0f')

    # cbar_ax = fig.add_axes([0.93, 0.76, 0.015, 0.15]) #[left, bottom, width, height]
    # cbar=fig.colorbar(mp1, cax=cbar_ax, format='%.0f')
    # cbar_ax = fig.add_axes([0.93, 0.54, 0.015, 0.15]) #[left, bottom, width, height]
    # cbar=fig.colorbar(mp2, cax=cbar_ax, format='%.0f')
    # cbar_ax = fig.add_axes([0.93, 0.32, 0.015, 0.15]) #[left, bottom, width, height]
    # cbar=fig.colorbar(mp3, cax=cbar_ax, format='%.0f')
    # cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.15]) #[left, bottom, width, height]
    # cbar=fig.colorbar(mp3, cax=cbar_ax, format='%.0f')

    # plt.tight_layout()

    plt.savefig(fig_path, dpi=300)
    plt.close()


def createIterativePredictionVideoForImageData(
    model,
    target,
    prediction,
    dt,
    ic_idx,
    set_name,
    testing_mode,
    rgb_channel,
    N_,
):
    interpolation = "bilinear"
    # IMAGE DATA
    if not model.data_info_dict["structured"]:
        assert (len(np.shape(prediction)) == 4)

    data_max = model.data_info_dict["data_max"]
    data_min = model.data_info_dict["data_min"]
    # data_std = model.data_info_dict["data_std"]
    data_range = data_max - data_min
    data_range = data_range[rgb_channel]
    data_norm = np.power(data_range, 2.0)

    # Iterating through the files to find vmin and vmax
    if model.data_info_dict["structured"]:
        vmin_array = []
        vmax_array = []
        vmax_rel_array = []
        vmax_nrmse_array = []
        for t in range(len(target)):
            target = np.array(target)
            prediction = np.array(prediction)

            target = correctStructuredDataPaths(model, target)
            prediction = correctStructuredDataPaths(model, prediction)

            target_data = utils_data.getDataHDF5Field(target[t, 0], target[t, 1])
            prediction_data = utils_data.getDataHDF5Field(prediction[t, 0], prediction[t, 1])
            # target_data = utils_data.loadData(target[t], "pickle")
            # prediction_data = utils_data.loadData(prediction[t], "pickle")
            vmin_array.append(target_data[rgb_channel].min())
            vmax_array.append(target_data[rgb_channel].max())
            abserror = np.abs(target_data - prediction_data)
            with np.errstate(divide='ignore', invalid='ignore'):
                relerror = abserror / (np.abs(target_data) + 1e-6)
            vmax_rel_array.append(relerror[rgb_channel].max())

            nrmse = np.sqrt(np.power(target_data[rgb_channel] - prediction_data[rgb_channel], 2.0) / data_norm)

            vmax_nrmse_array.append(np.max(nrmse))
            vmax_rel_array.append(relerror[rgb_channel].max())

        vmin = np.min(np.array(vmin_array))
        vmax = np.max(np.array(vmax_array))
        # vmax_rel = np.max(np.array(vmax_rel_array))
        vmax_rel = 100
        vmax_nrmse = np.max(np.array(vmax_nrmse_array))
    else:

        vmin = target[:, rgb_channel].min()
        vmax = target[:, rgb_channel].max()
        abserror = np.abs(target - prediction)
        # relerror = abserror / (1e-6 + np.abs(target))
        with np.errstate(divide='ignore', invalid='ignore'):
            relerror = abserror / (np.abs(target) + 1e-6)
        relerror[relerror > 100.0] = 100.0
        # vmax_rel = relerror[:, rgb_channel].max()
        vmax_rel = 100

        nrmse = np.sqrt(np.power(target[:, rgb_channel] - prediction[:, rgb_channel], 2.0) / data_norm)
        vmax_nrmse = np.max(np.array(nrmse))

    # setting appropriate vmin, vmax, and colormap
    vmin, vmax, cmap = getColormap(vmin, vmax,
                                   model.data_info_dict["colormap"])

    if model.params["make_videos"]:

        video_folder = "{:}_image_data_video_{:}_C{:}_IC{:}".format(
            testing_mode, set_name, rgb_channel, ic_idx)
        n_frames_max, frame_path_python, frame_path_bash, video_path = makeVideoPaths(
            model, video_folder)

        n_frames = np.min([n_frames_max, N_])
        # n_frames = 2
        for t in range(n_frames):
            fig_path = frame_path_python.format(t)
            # fig, axes = plt.subplots(figsize=(15, 3), ncols=4)
            fig, axes = plt.subplots(figsize=(10, 3), ncols=3)

            if model.data_info_dict["structured"]:
                target = correctStructuredDataPaths(model, target)
                prediction = correctStructuredDataPaths(model, prediction)

                # target_plot = utils_data.loadData(target[t], "pickle")
                target_plot = utils_data.getDataHDF5Field(
                    target[t, 0], target[t, 1])
                target_plot = target_plot[rgb_channel]
                # prediction_plot = utils_data.loadData(prediction[t], "pickle")
                prediction_plot = utils_data.getDataHDF5Field(
                    prediction[t, 0], prediction[t, 1])
                prediction_plot = prediction_plot[rgb_channel]

            else:
                target_plot = target[t, rgb_channel]
                prediction_plot = prediction[t, rgb_channel]

            mp1 = axes[0].imshow(
                target_plot,
                vmin=vmin,
                vmax=vmax,
                cmap=plt.get_cmap(cmap),
                aspect=1.0,
                interpolation=interpolation,
            )
            mp2 = axes[1].imshow(
                prediction_plot,
                vmin=vmin,
                vmax=vmax,
                cmap=plt.get_cmap(cmap),
                aspect=1.0,
                interpolation=interpolation,
            )
            # mp3 = axes[2].imshow(
            #     np.abs(target_plot - prediction_plot),
            #     vmin=0,
            #     vmax=vmax,
            #     cmap=plt.get_cmap("Reds"),
            #     aspect=1.0,
            #     interpolation=interpolation,
            # )
            # relerror = np.abs(target_plot -
            #                   prediction_plot) / (np.abs(target_plot) + 1e-6)
            # relerror[relerror > 100.0] = 100.0
            # mp4 = axes[3].imshow(
            #     relerror,
            #     vmin=0,
            #     vmax=vmax_rel,
            #     cmap=plt.get_cmap("Reds"),
            #     aspect=1.0,
            #     interpolation=interpolation,
            # )

            nrmse = np.sqrt(np.power(target_plot - prediction_plot, 2.0) / data_norm)

            mp3 = axes[2].imshow(
                nrmse,
                vmin=0,
                # vmax=1.0,
                vmax=vmax_nrmse,
                cmap=plt.get_cmap("Reds"),
                aspect=1.0,
                interpolation=interpolation,
            )

            axes[0].axis('off')
            axes[1].axis('off')
            axes[2].axis('off')

            axes[0].set_title("Reference")
            axes[1].set_title("Prediction")
            axes[2].set_title("NNAD")
            shrink = 0.5
            fig.colorbar(mp1, ax=axes[0], shrink=shrink, aspect=20*shrink)
            fig.colorbar(mp2, ax=axes[1], shrink=shrink, aspect=20*shrink)
            fig.colorbar(mp3, ax=axes[2], shrink=shrink, aspect=20*shrink)

            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()

        makeVideo(model, video_path, frame_path_bash, n_frames_max)

        # fields = ["reference", "prediction", "abserror", "relerror"]
        # fields = ["reference", "prediction", "NNAD", "NAD"]
        fields = ["reference", "prediction", "NNAD"]

        for field in fields:
            video_folder = "{:}_image_data_video_{:}_C{:}_IC{:}_{:}".format(
                testing_mode, set_name, rgb_channel, ic_idx, field)
            n_frames_max, frame_path_python, frame_path_bash, video_path = makeVideoPaths(
                model, video_folder, field=field)

            n_frames = np.min([n_frames_max, N_])
            # n_frames = 2

            for t in range(n_frames):
                fig_path = frame_path_python.format(t)
                fig, axes = plt.subplots()
                if model.data_info_dict["structured"]:
                    target = correctStructuredDataPaths(model, target)
                    prediction = correctStructuredDataPaths(model, prediction)
                    # target_plot = utils_data.loadData(target[t], "pickle")
                    target_plot = utils_data.getDataHDF5Field(
                        target[t, 0], target[t, 1])
                    target_plot = target_plot[rgb_channel]
                    # prediction_plot = utils_data.loadData(prediction[t], "pickle")
                    prediction_plot = utils_data.getDataHDF5Field(
                        prediction[t, 0], prediction[t, 1])
                    prediction_plot = prediction_plot[rgb_channel]

                else:
                    target_plot = target[t, rgb_channel]
                    prediction_plot = prediction[t, rgb_channel]

                if field == "reference":
                    data_plot = target_plot
                    vmin = vmin
                    vmax = vmax
                    cmap = plt.get_cmap(cmap)

                elif field == "prediction":
                    data_plot = prediction_plot
                    vmin = vmin
                    vmax = vmax
                    cmap = plt.get_cmap(cmap)

                elif field == "abserror":
                    data_plot = np.abs(target_plot - prediction_plot)
                    vmin = 0
                    vmax = vmax
                    cmap = plt.get_cmap("Reds")

                elif field == "relerror":
                    data_plot = np.abs(target_plot - prediction_plot) / (
                        np.abs(target_plot) + 1e-6)
                    vmin = 0
                    vmax = vmax_rel
                    cmap = plt.get_cmap("Reds")

                elif field == "NNAD":
                    data_plot = np.sqrt(np.power(target_plot - prediction_plot, 2.0) / data_norm)
                    vmin = 0
                    vmax = vmax_nrmse
                    cmap = plt.get_cmap("Reds")

                elif field == "NAD":
                    data_plot = np.abs(target_plot - prediction_plot) / data_range
                    vmin = 0
                    vmax = vmax_nrmse
                    cmap = plt.get_cmap("Reds")

                else:
                    raise ValueError("Bug here.")

                mp1 = axes.imshow(
                    data_plot,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    aspect=1.0,
                    interpolation=interpolation,
                )
                axes.axis('off')
                divider = make_axes_locatable(axes)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(mp1, cax=cax)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()

            makeVideo(model, video_path, frame_path_bash, n_frames_max)


def makeVideoPaths(model, video_folder, field=""):
    n_frames_max = 100
    # n_frames_max = 1000
    video_base_dir = utils_networks.getFigureDir(model)
    video_path = "{:}/{:}".format(video_base_dir, video_folder)
    os.makedirs(video_path + "/", exist_ok=True)
    frame_path_python = video_path + "/" + "{:}".format(field) + "frame_N{:04d}.png"
    frame_path_bash = video_path + "/" + "{:}".format(field) + "frame_N%04d.png"
    return n_frames_max, frame_path_python, frame_path_bash, video_path


def makeVideo(model, video_path, frame_path_bash, n_frames_max):
    # MAKING VIDEO
    command_str = "ffmpeg -y -r 5 -f image2 -s 1342x830 -i {:} -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' {:}.mp4".format(
        frame_path_bash, video_path)
    print("[utils_plotting] COMMAND TO MAKE VIDEO:")
    print(command_str)

    # Write video command to a .sh file in the Figures folder
    print("[utils_plotting] Writing the command to file...")
    with open(
            utils_networks.getFigureDir(model) + "/video_commands_abs.sh",
            "a+") as file:
        file.write(command_str)
        file.write("\n")

    # Remove duplicates
    filename = utils_networks.getFigureDir(model) + "/video_commands_abs.sh"
    utils_processing.removeDuplicates(filename)

    temp = frame_path_bash.split("/")
    temp = temp[-2:]
    frame_path_bash_rel = "./" + temp[0] + "/" + temp[1]

    temp = video_path.split("/")
    video_path_rel = "./" + temp[-1]
    command_str_rel = "ffmpeg -y -r 5 -f image2 -s 1342x830 -i {:} -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' {:}.mp4".format(
        frame_path_bash_rel, video_path_rel)
    print("[utils_plotting] COMMAND TO MAKE VIDEO (RELATIVE):")
    print(command_str_rel)
    print(
        "[utils_plotting] Writing the command (with relative paths) to file..."
    )
    with open(
            utils_networks.getFigureDir(model) + "/video_commands_rel.sh",
            "a+") as file:
        file.write(command_str_rel)
        file.write("\n")

    # os.system(command_str)
    # ffmpeg -y -r 5 -f image2 -s 1342x830 -i ./Iterative_Prediction_Video_TEST_IC108479/frame_N%04d.{:} -vcodec libx264 -crf 1  -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" video_TEST_IC108479.mp4

    # if not CLUSTER:
    #     return_value = subprocess.call([
    #         'ffmpeg',
    #         '-y',
    #         '-r', '20',
    #         '-f', 'image2',
    #         '-s', '1342x830',
    #         '-i', '{:}'.format(frame_path_bash),
    #         '-vcodec', 'libx264',
    #         '-crf', '1',
    #         '-pix_fmt', 'yuv420p',
    #         '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
    #         '-frames:v', '{:}'.format(n_frames_max),
    #         '-shortest', "{:}.mp4".format(video_path),
    #     ])
    #     if return_value:
    #         print("[utils_plotting] Failure: FFMPEG probably not installed.")
    #     else:
    #         print("[utils_plotting] Sucess: Video ready!")

    # Remove duplicates
    filename = utils_networks.getFigureDir(model) + "/video_commands_rel.sh"
    utils_processing.removeDuplicates(filename)


def plotTestingContourEvolution(
    model,
    target,
    output,
    dt,
    ic_idx,
    set_name,
    latent_states=None,
    channel="",
    testing_mode="",
    with_multiscale_bar=False,
):
    print("[utils_plotting] # plotTestingContourEvolution() #")
    error = np.abs(target - output)
    vmin = target.min()
    vmax = target.max()
    vmin_error = 0.0
    vmax_error = vmax - vmin

    prediction_horizon = np.shape(target)[0]
    if latent_states is None:
        fig, axes = plt.subplots(nrows=1,
                                 ncols=4,
                                 figsize=(14, 6),
                                 sharey=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        axes[0].set_ylabel(r"Time $t$")
        contours_vec = []
        mp = createContour_(fig,
                            axes[0],
                            target,
                            "Reference",
                            vmin,
                            vmax,
                            model.data_info_dict["colormap"],
                            dt,
                            xlabel="State")

        contours_vec.append(mp)
        mp = createContour_(fig,
                            axes[1],
                            output,
                            "Output",
                            vmin,
                            vmax,
                            model.data_info_dict["colormap"],
                            dt,
                            xlabel="State")
        contours_vec.append(mp)
        mp = createContour_(fig,
                            axes[2],
                            error,
                            "Error",
                            vmin_error,
                            vmax_error,
                            plt.get_cmap("Reds"),
                            dt,
                            xlabel="State")
        contours_vec.append(mp)
        corr = [pearsonr(target[i], output[i])[0] for i in range(len(target))]
        time_vector = np.arange(target.shape[0]) * dt
        axes[3].plot(corr, time_vector)
        axes[3].set_title("Correlation")
        axes[3].set_xlabel(r"Correlation")
        axes[3].set_xlim((-1, 1))
        axes[3].set_ylim((time_vector.min(), time_vector.max()))
        for contours in contours_vec:
            for pathcoll in contours.collections:
                pathcoll.set_rasterized(True)
        fig_path = utils_networks.getFigureDir(
            model) + "/{:}_{:}_IC{:}_C{:}_contour.{:}".format(
                testing_mode, set_name, ic_idx, channel, FIGTYPE)
        plt.savefig(fig_path, dpi=300)
        plt.close()
    elif len(np.shape(latent_states)) == 2:
        # Plotting the contour plot
        ncols = 6 if with_multiscale_bar else 5
        fig, axes = plt.subplots(nrows=1,
                                 ncols=ncols,
                                 figsize=(3.6 * ncols, 6),
                                 sharey=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        axes[0].set_ylabel(r"Time $t$")
        contours_vec = []
        mp = createContour_(fig,
                            axes[0],
                            target,
                            "Reference",
                            vmin,
                            vmax,
                            model.data_info_dict["colormap"],
                            dt,
                            xlabel="State")

        contours_vec.append(mp)
        time_vector = np.arange(target.shape[0]) * dt
        vmin_latent = np.min(latent_states)
        vmax_latent = np.max(latent_states)
        if np.shape(latent_states)[1] > 1:
            mp = createLatentContour_(
                fig,
                axes[1],
                latent_states,
                None,
                vmin_latent,
                vmax_latent,
                model.data_info_dict["colormap"],
                dt,
                xlabel="Latent state",
                numxaxisticks=8,
            )
            # contours_vec.append(mp)
        else:
            axes[1].plot(latent_states[:, 0], time_vector)
            axes[1].set_xlabel(r"Latent state")
            axes[1].set_ylim((time_vector.min(), time_vector.max()))

        mp = createContour_(
            fig,
            axes[2],
            output,
            # "Output",
            "Prediction",
            vmin,
            vmax,
            model.data_info_dict["colormap"],
            dt,
            xlabel="State",
            numxaxisticks=4,
        )
        contours_vec.append(mp)
        mp = createContour_(
            fig,
            axes[3],
            error,
            "Error",
            vmin_error,
            vmax_error,
            plt.get_cmap("Reds"),
            dt,
            xlabel="State",
            numxaxisticks=4,
        )
        contours_vec.append(mp)

        corr = [pearsonr(target[i], output[i])[0] for i in range(len(target))]
        axes[4].plot(corr, time_vector)
        axes[4].set_xlabel(r"Correlation")
        axes[4].set_xlim((-1, 1))
        axes[4].set_ylim((time_vector.min(), time_vector.max()))
        axes[4].set_title("Correlation")
        if with_multiscale_bar:
            # Add a bar plot demonstrating where it is with_multiscale_bar and where not
            axes[5].set_title("Multiscale?")
            multiscale_rounds, macro_steps_per_round, micro_steps_per_round, _, _ = model.parent.getMultiscaleParams(
                testing_mode, prediction_horizon)

            start_idx = 0
            for round_ in range(multiscale_rounds):
                end_idx = start_idx + macro_steps_per_round[round_]
                start_t = start_idx * dt
                end_t = end_idx * dt
                axes[5].axhspan(start_t,
                                end_t,
                                color="orange",
                                alpha=0.7,
                                label=None if round_ == 0 else None)

                start_idx = end_idx
                if round_ < len(micro_steps_per_round):
                    end_idx = start_idx + micro_steps_per_round[round_]
                    start_t = start_idx * dt
                    end_t = end_idx * dt
                    axes[5].axhspan(start_t,
                                    end_t,
                                    color="green",
                                    alpha=0.7,
                                    label=None if round_ == 0 else None)
                    start_idx = end_idx

                plt.legend(loc="upper left",
                           bbox_to_anchor=(1.05, 1),
                           borderaxespad=0.)
                plt.axis('off')
            axes[5].set_ylim((time_vector.min(), time_vector.max()))

        for contours in contours_vec:
            for pathcoll in contours.collections:
                pathcoll.set_rasterized(True)

        plt.tight_layout()
        fig_path = utils_networks.getFigureDir(
            model) + "/{:}_{:}_IC{:}_C{:}_contour.{:}".format(
                testing_mode, set_name, ic_idx, channel, FIGTYPE)
        plt.savefig(fig_path, dpi=300)
        plt.close()


def plotTestingContours(model,
                        target,
                        output,
                        dt,
                        ic_idx,
                        set_name,
                        latent_states=None,
                        testing_mode="",
                        with_multiscale_bar=False,
                        quantity="Positions",
                        xlabel="Positions",
                        hist_data=None,
                        wasserstein_distance_data=None):

    print(
        "[utils_plotting] # plotTestingContours() # - {:}, {:}, Multiscale bar? {:}"
        .format(testing_mode, set_name, with_multiscale_bar))

    if model.data_info_dict["contour_plots"]:

        if model.data_info_dict["structured"]:
            latent_states = utils_data.getDataFromHDF5PathArraySingleIC(
                latent_states)
            target = utils_data.getDataFromHDF5PathArraySingleIC(target)
            output = utils_data.getDataFromHDF5PathArraySingleIC(output)

        T, channels, D = np.shape(target)

        for channel in range(channels):

            plotTestingContourEvolution(
                model,
                target[:, channel],
                output[:, channel],
                dt,
                ic_idx,
                set_name,
                latent_states=latent_states,
                testing_mode=testing_mode,
                channel="{:d}".format(channel),
                with_multiscale_bar=with_multiscale_bar)

    if model.data_info_dict["density_plots"]:
        plotTestingContourDensity(
            model,
            target,
            output,
            dt,
            ic_idx,
            set_name,
            latent_states=latent_states,
            testing_mode=testing_mode,
            with_multiscale_bar=with_multiscale_bar,
            quantity=quantity,
            xlabel=xlabel,
            hist_data=hist_data,
            wasserstein_distance_data=wasserstein_distance_data)

    if not model.data_info_dict["structured"] and len(np.shape(target)) == 2:
        N_PLOT_MAX = 1000
        target = target[:N_PLOT_MAX]
        output = output[:N_PLOT_MAX]
        # PLOTTING 10 SIGNALS FOR REFERENCE
        plot_max = np.min([np.shape(target)[1], 10])
        fig_path = utils_networks.getFigureDir(
            model) + "/{:}_{:}_{:}_signals.{:}".format(testing_mode, set_name,
                                                       ic_idx, FIGTYPE)
        for idx in range(plot_max):
            plt.plot(np.arange(np.shape(output)[0]),
                     output[:, idx],
                     color='blue',
                     linewidth=1.0,
                     label='Output' if idx == 0 else None)
            plt.plot(np.arange(np.shape(target)[0]),
                     target[:, idx],
                     color='red',
                     linewidth=1.0,
                     label='Target' if idx == 0 else None)
        plt.legend(loc="upper left",
                   bbox_to_anchor=(1.05, 1),
                   borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        fig_path = model.getFigureDir(
        ) + "/{:}_{:}_{:}_signals_target.{:}".format(testing_mode, set_name,
                                                     ic_idx, FIGTYPE)
        for idx in range(plot_max):
            plt.plot(np.arange(np.shape(target)[0]),
                     target[:, idx],
                     linewidth=1.0,
                     label='Target' if idx == 0 else None)
        plt.legend(loc="upper left",
                   bbox_to_anchor=(1.05, 1),
                   borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        fig_path = model.getFigureDir(
        ) + "/{:}_{:}_{:}_signals_output.{:}".format(testing_mode, set_name,
                                                     ic_idx, FIGTYPE)
        for idx in range(plot_max):
            plt.plot(np.arange(np.shape(output)[0]),
                     output[:, idx],
                     linewidth=1.0,
                     label='Output' if idx == 0 else None)
        plt.legend(loc="upper left",
                   bbox_to_anchor=(1.05, 1),
                   borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

    elif model.params["Dx"] > 0 and model.params["Dy"] > 0:
        createIterativePredictionPlotsForImageData(model, target, output, dt,
                                                   ic_idx, set_name,
                                                   testing_mode)


def createDensityContour_(fig,
                          ax,
                          density,
                          bins,
                          title,
                          vmin,
                          vmax,
                          cmap,
                          dt,
                          xlabel="Value",
                          scale=None):
    ax.set_title(title)
    t, s = np.meshgrid(np.arange(density.shape[0]) * dt, bins)
    if scale is None:
        mp = ax.contourf(s,
                         t,
                         np.transpose(density),
                         cmap=cmap,
                         levels=np.linspace(vmin, vmax, 60),
                         extend="both")
    elif scale == "log":
        from matplotlib import ticker
        mp = ax.contourf(s,
                         t,
                         np.transpose(density),
                         cmap=cmap,
                         locator=ticker.LogLocator(),
                         extend="both")
    fig.colorbar(mp, ax=ax)
    ax.set_xlabel(r"{:}".format(xlabel))
    return mp


def createContour_(
    fig,
    ax,
    data,
    title,
    vmin,
    vmax,
    cmap,
    dt,
    mask_where=None,
    xlabel=None,
    numxaxisticks=None,
):
    ax.set_title(title)
    time_vec = np.arange(data.shape[0]) * dt
    state_vec = np.arange(data.shape[1])
    if mask_where is not None:
        # print(mask_where)
        mask = [
            mask_where[i] * np.ones(data.shape[1])
            for i in range(np.shape(mask_where)[0])
        ]
        mask = np.array(mask)
        data = np.ma.array(data, mask=mask)

    t, s = np.meshgrid(time_vec, state_vec)
    mp = ax.contourf(
        s,
        t,
        np.transpose(data),
        15,
        cmap=cmap,
        levels=np.linspace(vmin, vmax, 60),
        extend="both",
    )
    cbar = fig.colorbar(mp, ax=ax)

    tick_locator = matplotlib.ticker.MaxNLocator(nbins=5, symmetric=True)
    cbar.locator = tick_locator
    cbar.update_ticks()

    if not (numxaxisticks == None):
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=numxaxisticks,
                                          symmetric=False))

    ax.set_xlabel(r"{:}".format(xlabel))
    return mp


def createLatentContour_(
    fig,
    ax,
    data,
    title,
    vmin,
    vmax,
    cmap,
    dt,
    mask_where=None,
    xlabel=None,
    numxaxisticks=None,
):
    """ Automatically detect colorbar """
    vmin, vmax, cmap = getColormap(vmin, vmax, cmap)

    ax.set_title(title)
    time_vec = np.arange(data.shape[0]) * dt
    state_vec = np.arange(data.shape[1])
    if mask_where is not None:
        # print(mask_where)
        mask = [
            mask_where[i] * np.ones(data.shape[1])
            for i in range(np.shape(mask_where)[0])
        ]
        mask = np.array(mask)
        data = np.ma.array(data, mask=mask)

    t, s = np.meshgrid(time_vec, state_vec)

    mp = ax.imshow(data,
                   interpolation='nearest',
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax)
    ax.set_aspect('auto')
    fig.colorbar(mp, ax=ax)
    ax.set_xlabel(r"{:}".format(xlabel))

    if not (numxaxisticks == None):
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=numxaxisticks,
                                          symmetric=False))

    return mp


# def plotStateDistributions(model, results, set_name, testing_mode):
#     if model.data_info_dict["structured"]:
#         warnings.warn("[plotStateDistributions()] Warning: structured data (memory intensive). No plotting of state distribution.")
#         return 0

#     hist_data = results["state_dist_hist_data"]
#     LL = results["state_dist_LL"]
#     if type(LL) is list:
#         # Iterating over all states
#         N = len(LL)
#         for n in range(N):
#             _, _, density_target, density_pred, bin_centers = hist_data[n]
#             bounds = results["state_dist_bounds"][n]
#             plotStateDistribution(model, density_target, density_pred,
#                                   bin_centers, bounds, n, set_name,
#                                   testing_mode)
#     else:
#         # CUMMULATIVE STATE DISTRIBUTION
#         _, _, density_target, density_pred, bin_centers = hist_data
#         bounds = results["state_dist_bounds"]
#         plotStateDistribution(model, density_target, density_pred, bin_centers,
#                               bounds, 0, set_name, testing_mode)

# def plotStateDistribution(model, density_target, density_pred, bin_centers,
#                           bounds, state_num, set_name, testing_mode):

#     ################################
#     ### BAR PLOTS - (.bar())
#     ################################

#     fig_path = model.getFigureDir(
#     ) + "/{:}_state_dist_bar_S{:}_{:}.{:}".format(testing_mode, state_num,
#                                                   set_name, FIGTYPE)
#     bin_width = bin_centers[1] - bin_centers[0]
#     vmax_density = np.max([np.max(density_target), np.max(density_pred)])
#     vmin_density = 0.0
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.bar(bin_centers,
#            density_target,
#            width=bin_width,
#            color='tab:green',
#            alpha=0.5,
#            label="Target Density")
#     ax.bar(bin_centers,
#            density_pred,
#            width=bin_width,
#            color='tab:blue',
#            alpha=0.5,
#            label="Predicted Density")
#     ax.legend(loc="upper center",
#               bbox_to_anchor=(0.5, 1.1),
#               borderaxespad=0.,
#               ncol=2,
#               frameon=False)
#     ax.set_ylim((vmin_density, vmax_density))
#     ax.set_xlim(bounds)
#     # state_label = "s_{" + "{:}".format(state_num+1) + "}"
#     state_label = "x_{" + "{:}".format(state_num + 1) + "}"
#     xlabel = "$" + state_label + "$"
#     ax.set_xlabel(r"{:}".format(xlabel))
#     ylabel = "$" + "f_{" + state_label + "}(" + state_label + ")$"
#     ax.set_ylabel(r"{:}".format(ylabel))
#     # plt.show()
#     plt.tight_layout()
#     plt.savefig(fig_path, dpi=300)
#     plt.close()

#     ################################
#     ### PLOTS - (.plt())
#     ################################

#     # fig_path = utils_networks.getFigureDir(model) + "/{:}_state_dist_S{:}_{:}.{:}".format(testing_mode, state_num, set_name, FIGTYPE)
#     # bin_width = bin_centers[1] - bin_centers[0]
#     # vmax_density = np.max([np.max(density_target), np.max(density_pred)])
#     # vmin_density = 0.0
#     # fig, ax = plt.subplots(figsize=(6, 6))
#     # ax.plot(bin_centers, density_target, color="tab:green",label="Target Density")
#     # ax.plot(bin_centers, density_pred, color="tab:blue", label="Predicted Density")
#     # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), borderaxespad=0., ncol=2, frameon=False)
#     # ax.set_ylim((vmin_density, vmax_density))
#     # ax.set_xlim(bounds)
#     # state_label = "s_{" + "{:}".format(state_num) + "}"
#     # xlabel = "$" + state_label + "$"
#     # ax.set_xlabel(r"{:}".format(xlabel))
#     # ylabel = "$" + "f_{" + state_label + "}(" + state_label +")$"
#     # ax.set_ylabel(r"{:}".format(ylabel))
#     # # plt.show()
#     # plt.tight_layout()
#     # plt.savefig(fig_path, dpi=300)
#     # plt.close()


def plotSpectrum(model, results, set_name, testing_mode=""):
    if model.data_info_dict["structured"]:
        warnings.warn(
            "[plotSpectrum()] Warning: structured data (memory intensive). No plotting of spectrum."
        )
        return 0

    assert ("sp_true" in results)
    assert ("sp_pred" in results)
    assert ("freq_true" in results)
    assert ("freq_pred" in results)
    sp_true = results["sp_true"]
    sp_pred = results["sp_pred"]
    freq_true = results["freq_true"]
    freq_pred = results["freq_pred"]
    fig_path = utils_networks.getFigureDir(
        model) + "/{:}_{:}_frequencies.{:}".format(testing_mode, set_name,
                                                   FIGTYPE)
    spatial_dims = len(np.shape(sp_pred))
    if spatial_dims == 1:
        # plt.title("Frequency error={:.4f}".format(np.mean(np.abs(sp_true-sp_pred))))
        plt.plot(freq_pred, sp_pred, '--', color="tab:red", label="prediction")
        plt.plot(freq_true,
                 sp_true,
                 '--',
                 color="tab:green",
                 label="Reference")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectrum [dB]')
        # plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.legend(loc="upper center",
                   bbox_to_anchor=(0.5, 1.1),
                   borderaxespad=0.,
                   ncol=2,
                   frameon=False)
    elif spatial_dims == 2:
        fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
        plt.suptitle("Frequency error = {:.4f}".format(
            np.mean(np.abs(sp_true - sp_pred))))
        mp1 = axes[0].imshow(sp_true,
                             cmap=plt.get_cmap("plasma"),
                             aspect=1.0,
                             interpolation='lanczos')
        axes[0].set_title("True Spatial FFT2D")
        mp2 = axes[1].imshow(sp_pred,
                             cmap=plt.get_cmap("plasma"),
                             aspect=1.0,
                             interpolation='lanczos')
        axes[1].set_title("Predicted Spatial FFT2D")
        fig.colorbar(mp1, ax=axes[0])
        fig.colorbar(mp2, ax=axes[1])
    else:
        raise ValueError("Not implemented.")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plotLatentDynamicsComparison(model, latent_states_dict, set_name):
    max_plot = 3
    iter_ = 0
    for key1, value1 in latent_states_dict.items():
        for key2, value2 in latent_states_dict.items():
            if key1 != key2:
                plotLatentDynamicsComparison_(model, value1[0], value2[0],
                                              key1, key2, set_name)
                iter_ += 1
            if iter_ > max_plot:
                break


def plotLatentDynamicsComparison_(model,
                                  latent_states1,
                                  latent_states2,
                                  label1,
                                  label2,
                                  set_name,
                                  latent_states3=None,
                                  label3=None):
    shape_ = np.shape(latent_states1)
    if len(shape_) == 2:
        T, D = shape_
        if D >= 2:
            fig, ax = plt.subplots()
            plt.title("Latent dynamics in {:}".format(set_name))
            arrowplot(ax,
                      latent_states1[:, 0],
                      latent_states1[:, 1],
                      nArrs=100,
                      color="blue",
                      label=label1)
            arrowplot(ax,
                      latent_states2[:, 0],
                      latent_states2[:, 1],
                      nArrs=100,
                      color="green",
                      label=label2)
            if label3 is not None:
                arrowplot(ax,
                          latent_states3[:, 0],
                          latent_states3[:, 1],
                          nArrs=100,
                          color="tab:red",
                          label=label3)
            plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.)
            fig_path = utils_networks.getFigureDir(
                model
            ) + "/Comparison_latent_dynamics_{:}_{:}_{:}_{:}.{:}".format(
                set_name, label1, label2, label3, FIGTYPE)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
        else:
            fig, ax = plt.subplots()
            plt.title("Latent dynamics in {:}".format(set_name))
            plt.plot(latent_states1[:-1, 0],
                     latent_states1[1:, 0],
                     'b',
                     linewidth=1.0,
                     label=label1)
            plt.plot(latent_states2[:-1, 0],
                     latent_states2[1:, 0],
                     'g',
                     linewidth=1.0,
                     label=label2)
            if label3 is not None:
                plt.plot(latent_states3[:-1, 0],
                         latent_states3[1:, 0],
                         'r',
                         linewidth=1.0,
                         label=label3)
            fig_path = utils_networks.getFigureDir(
                model
            ) + "/Comparison_latent_dynamics_{:}_{:}_{:}_{:}.{:}".format(
                set_name, label1, label2, label3, FIGTYPE)
            plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()


def scatterDensityLatentDynamicsPlot(x,
                                     y,
                                     ax=None,
                                     sort=True,
                                     bins=20,
                                     cmap=plt.get_cmap("Reds"),
                                     with_colorbar=True,
                                     log_norm=True):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data,
                np.vstack([x, y]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=0.0)

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    if ax is None:
        fig, ax = plt.subplots()

    if log_norm:
        mp = ax.scatter(x,
                        y,
                        c=z,
                        cmap=cmap,
                        norm=matplotlib.colors.LogNorm(),
                        rasterized=True)
    else:
        mp = ax.scatter(x, y, c=z, cmap=cmap, rasterized=True)
    if with_colorbar: plt.colorbar(mp)
    return ax
