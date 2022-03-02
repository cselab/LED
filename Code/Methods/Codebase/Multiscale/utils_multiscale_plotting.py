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

from .. import Utils as utils

# Plotting parameters
import matplotlib

hostname = socket.gethostname()
print("[utils_multiscale_plotting] PLOTTING HOSTNAME: {:}".format(hostname))
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

print("[utils_multiscale_plotting] CLUSTER={:}, CLUSTER_NAME={:}".format(
    CLUSTER, CLUSTER_NAME))

if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

print("[utils_multiscale_plotting] Matplotlib Version = {:}".format(
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

# FIGTYPE = "pdf"
FIGTYPE = "png"

color_labels = [
    'tab:blue',
    'tab:green',
    'tab:brown',
    'tab:orange',
    'tab:cyan',
    'tab:olive',
    'tab:pink',
    'tab:gray',
    'tab:purple',
    # 'tab:red',
]

linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
linemarkers = ["s", "d", "o", ">", "*", "x", "<", ">"]
linemarkerswidth = [3, 2, 2, 2, 4, 2, 2, 2]


def error2LabelDictAverage():
    dict_ = {
        "CORR": "Correlation",
        "NAD": "MNAD",
        "RMSE": "RMSE",
        "mnad_act": "MNAD$(u, \\tilde{u})$",
        "mnad_in": "MNAD$(v, \\tilde{v})$",
        "state_dist_L1_hist_error": "L1-NHD",
        "state_dist_wasserstein_distance": "WD",
        # "mse_avg": "MSE",
        # "rmse_avg": "RMSE",
        # "abserror_avg": "ABS",
        # "state_dist_L1_hist_error_all": "L1-NHD",
        # "state_dist_wasserstein_distance_all": "WD",
        # "state_dist_L1_hist_error_avg": "L1-NHD",
        # "state_dist_wasserstein_distance_avg": "WD",
        # "rmnse_avg_over_ics": "NNAD",
        # "mnad_avg_over_ics_act": "NAD$(u, \\tilde{u})$",
        # "mnad_avg_over_ics_in": "NAD$(v, \\tilde{v})$",
        # "mnad_avg_over_ics": "NAD",
    }
    return dict_


def error2LabelDictTime():
    dict_ = {
        "CORR": "Correlation",
        "NAD": "NAD",
        "RMSE": "RMSE",
        "mnad_act": "NAD$(u, \\tilde{u})$",
        "mnad_in": "NAD$(v, \\tilde{v})$",
        "state_dist_L1_hist_error": "L1-NHD",
        "state_dist_wasserstein_distance": "WD",
        # "mse_avg": "MSE",
        # "rmse_avg": "RMSE",
        # "abserror_avg": "ABS",
        # "state_dist_L1_hist_error_all": "L1-NHD",
        # "state_dist_wasserstein_distance_all": "WD",
        # "state_dist_L1_hist_error_avg": "L1-NHD",
        # "state_dist_wasserstein_distance_avg": "WD",
        # "rmnse_avg_over_ics": "NNAD",
        # "mnad_avg_over_ics_act": "NAD$(u, \\tilde{u})$",
        # "mnad_avg_over_ics_in": "NAD$(v, \\tilde{v})$",
        # "mnad_avg_over_ics": "NAD",
    }
    return dict_


def loadDataArray(list_of_paths):
    data = []
    for i in range(len(list_of_paths)):
        data.append(utils.loadData(list_of_paths[i], "pickle"))
    data = np.array(data)
    return data


def getMacroAndMicroSteps(model,
                          dicts_to_compare,
                          field,
                          prediction_horizon=2000):
    macro_steps, micro_steps, result = [], [], []
    result_iterative_found = False
    result_equations_found = False

    for key, values in dicts_to_compare.items():

        if "iterative" in key:
            result_iterative = values[field]
            result_iterative_found = True
        elif "_macro_0" in key:
            # Only used to track the time !
            result_equations = values[field]
            result_equations_found = True

        elif "multiscale" in key:
            temp = key.split("_")
            macro_steps_ = int(float(temp[-1]))
            micro_steps_ = int(float(temp[-3]))
            result_ = values[field]

            if macro_steps_ >= prediction_horizon:
                result_iterative = values[field]
                result_iterative_found = True
            else:
                macro_steps.append(macro_steps_)
                micro_steps.append(micro_steps_)
                result.append(result_)
        elif "teacher_forcing_forecasting" in key:
            pass
        else:
            raise ValueError(
                "[utils_multiscale_plotting] I don't know how to process {:}.".
                format(key))
    macro_steps = np.array(macro_steps)
    micro_steps = np.array(micro_steps)
    result = np.array(result)
    if not result_iterative_found:
        raise ValueError(
            "[utils_multiscale_plotting] Result from iterative forecasting, or result from macro_steps>prediction_horizon not found."
        )
    if not result_equations_found:
        warnings.warn(
            "[utils_multiscale_plotting] Result from equations not found.")
        result_equations = None

    result_iterative = np.array(result_iterative)
    return macro_steps, micro_steps, result, result_iterative, result_equations


def makeBarPlot(
    model,
    field,
    micro_steps,
    micro_step,
    macro_steps,
    result,
    result_iterative,
    dt,
    set_name,
):
    error2Label = error2LabelDictAverage()

    if (field not in error2Label):
        raise ValueError("Field {:} not in error2Label.".format(field))

    indexes = micro_steps == micro_step
    macro_steps_plot = macro_steps[indexes]
    result_plot = result[indexes]

    idx_sort = np.argsort(macro_steps_plot)
    macro_steps_plot = macro_steps_plot[idx_sort]
    result_plot = result_plot[idx_sort]

    macro_steps_plot_float = np.array(
        [float(temp) for temp in macro_steps_plot])
    rho_plot = macro_steps_plot_float / float(micro_step)

    # plt.plot(rho_plot, result_plot)

    ######## BAR PLOT
    result_plot = np.concatenate((result_plot, result_iterative[np.newaxis]),
                                 axis=0)
    rho_plot = ["{:.2f}".format(temp) for temp in rho_plot]
    rho_plot.append(str("Latent"))

    # Mean over time
    result_plot = np.mean(result_plot, axis=(2))
    # Mean over initial conditions
    result_plot_mean = np.mean(result_plot, axis=(1))
    # min and max over initial conditions
    result_plot_min = np.min(result_plot, axis=(1))
    result_plot_max = np.max(result_plot, axis=(1))

    # print(result_plot_mean)
    # print(result_plot_min)
    # print(result_plot_max)

    labels = rho_plot
    x_pos = np.arange(len(labels))
    y_mean = result_plot_mean
    error = result_plot_max - result_plot_min
    yerr = np.array([
        result_plot_mean - result_plot_min, result_plot_max - result_plot_mean
    ])

    # Build the plot
    fig, ax = plt.subplots()
    barlist = ax.bar(
        x_pos,
        y_mean,
        yerr=yerr,
        align='center',
        alpha=0.5,
        ecolor='black',
        capsize=10,
    )

    for i_color in range(len(barlist[:-1])):
        barlist[i_color].set_color(color_labels[i_color])
    barlist[-1].set_color("tab:red")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)

    ax.yaxis.grid(True)

    ylabel = error2Label[field]
    plt.ylabel(r"{:}".format(ylabel))

    plt.xlabel(r"$\rho = T_m / T_{\mu}$")
    # micro_step_time = micro_step * dt / lyapunov_time
    title_str = "$T_{\mu}=" + "{:.2f}".format(micro_step * dt)
    # +"="+"{:.2f}".format(micro_step_time) + "\, \Lambda_1"
    title_str += ", \, T_{f}=" + "{:.2f}".format(
        model.prediction_horizon * dt) + "$"
    # print(title_str)
    plt.title(r"{:}".format(title_str), pad=10)
    fig_path = utils.getFigureDir(
        model) + "/Comp_multiscale_barplot_{:}_micro{:}_{:}.{:}".format(
            set_name, int(micro_step), field, FIGTYPE)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def makeErrorTimePlot(
    model,
    field,
    micro_steps,
    micro_step,
    macro_steps,
    result,
    result_iterative,
    dt,
    set_name,
    with_legend,
):
    error2Label = error2LabelDictTime()

    legend_str = "_legend" if with_legend else ""

    if model.params["system_name"] in ["KSGP64L22", "KSGP64L22Large"]:
        lyapunov_time = 20.83
        dt_scaled = dt / lyapunov_time
        vpt_label = "VPT ($\\times \Lambda_1$)"
        # vpt_label = "VPT"
        time_label = "$t /\Lambda_1$"
    else:
        dt_scaled = dt
        vpt_label = "VPT"
        time_label = "$t$"

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    # print("micro_step\n{:}".format(micro_step))
    indexes = micro_steps == micro_step
    macro_steps_plot = macro_steps[indexes]
    result_plot = result[indexes]
    result_iterative = np.array(result_iterative)
    result_plot = np.array(result_plot)

    idx_sort = np.argsort(macro_steps_plot)
    macro_steps_plot = macro_steps_plot[idx_sort]
    result_plot = result_plot[idx_sort]

    # Mean over initial conditions
    result_iterative = np.mean(result_iterative, axis=(0))
    result_plot = np.mean(result_plot, axis=(1))

    prediction_horizon = len(result_iterative)
    num_runs, T = np.shape(result_plot)
    time_vector = np.arange(T) * dt_scaled

    i_color = -1
    markevery = int(np.max([int(T / 12), 1]))
    plt.plot(
        time_vector,
        result_iterative,
        label=r"{:}".format("Iterative Latent Forecasting"),
        linestyle=linestyles[i_color],
        marker=linemarkers[i_color],
        markeredgewidth=linemarkerswidth[i_color],
        markersize=10,
        markevery=markevery,
        color="tab:red",
        linewidth=2,
    )

    for i_color in range(num_runs):
        # label = "T_{m}=" + "{:.2f}".format(float(macro_steps_plot[i]) * dt)
        # if micro_step>0: label = label + ", \, \\rho=" + "{:.2f}".format(float(macro_steps_plot[i]/micro_step))
        # label = "$" + label + "$"
        label = "Multiscale Forecasting $T_{\mu}=" + "{:.0f}".format(
            float(micro_step) * dt) + "$, $T_{m}=" + "{:.0f}".format(
                float(macro_steps_plot[i_color]) *
                dt) + "$" + ", $\\rho=" + "{:.2f}".format(
                    float(macro_steps_plot[i_color] / micro_step)) + "$"

        plt.plot(time_vector,
                 result_plot[i_color],
                 label=r"{:}".format(label),
                 linestyle=linestyles[i_color],
                 marker=linemarkers[i_color],
                 markeredgewidth=linemarkerswidth[i_color],
                 markersize=10,
                 markevery=markevery,
                 color=color_labels[i_color],
                 linewidth=2)

    plt.xlim([np.min(time_vector), np.max(time_vector)])
    plt.ylim([
        np.min(np.array(result_iterative)),
        1.1 * np.max(np.array(result_iterative))
    ])

    ylabel = error2Label[field]
    plt.ylabel(r"{:}".format(ylabel))

    plt.xlabel(r"{:}".format(time_label))

    title_str = "$T_{\mu}=" + "{:.2f}".format(micro_step * dt)
    title_str += ", \, T_{f}=" + "{:.2f}".format(prediction_horizon * dt) + "$"

    plt.title(r"{:}".format(title_str), pad=10)
    if with_legend:
        plt.legend(loc="upper left",
                   bbox_to_anchor=(1.05, 1),
                   borderaxespad=0.,
                   frameon=False)
    plt.tight_layout()
    fig_path = utils.getFigureDir(
        model) + "/Comp_multiscale_timeplot_{:}_micro{:}_{:}{:}.{:}".format(
            set_name, int(micro_step), field, legend_str, FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    # plt.show()
    plt.close()


def makeTimeBarPlot(
    model,
    field,
    micro_steps,
    micro_step,
    macro_steps,
    result,
    result_iterative,
    dt,
    set_name,
    result_equations,
):
    # CREATING BAR PLOT
    # print("micro_step\n{:}".format(micro_step))
    indexes = micro_steps == micro_step
    macro_steps_plot = macro_steps[indexes]
    result_plot = result[indexes]

    idx_sort = np.argsort(macro_steps_plot)
    macro_steps_plot = macro_steps_plot[idx_sort]
    result_plot = result_plot[idx_sort]

    rho_plot = [
        "{:.2f}".format((temp / float(micro_step)))
        for temp in macro_steps_plot
    ]
    rho_plot.append(str("Latent"))
    result_plot = list(result_plot) + list([result_iterative])
    barlist = plt.bar(rho_plot, result_plot)

    for i_color in range(len(barlist[:-1])):
        barlist[i_color].set_color(color_labels[i_color])
    barlist[-1].set_color("tab:red")

    plt.ylabel(r"${:}$".format("T_{iter}"))
    plt.xlabel(r"$ \rho=T_m / T_{\mu}$")
    title_str = "{:.2f}".format(micro_step * dt)
    # micro_step_time = micro_step * dt / lyapunov_time
    title_str = "$T_{\mu}=" + title_str
    title_str += ", \, T_{f}=" + "{:.2f}".format(
        model.prediction_horizon * dt) + "$"
    plt.title(r"{:}".format(title_str), pad=10)
    fig_path = utils.getFigureDir(
        model) + "/Comp_multiscale_{:}_micro{:}_{:}.{:}".format(
            set_name, int(micro_step), field, FIGTYPE)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    field_second = "speed_up"
    # CREATING BAR PLOT
    # print("micro_step\n{:}".format(micro_step))
    # Computing the speed-up
    result_plot = float(result_equations) / np.array(result_plot)
    barlist = plt.bar(rho_plot, result_plot)
    barlist[-1].set_color("tab:red")

    for i_color in range(len(barlist[:-1])):
        barlist[i_color].set_color(color_labels[i_color])
    barlist[-1].set_color("tab:red")

    plt.ylabel(r"{:}".format("Speed-up"))
    plt.xlabel(r"$ \rho=T_m / T_{\mu}$")
    title_str = "{:.2f}".format(micro_step * dt)
    # micro_step_time = micro_step * dt / lyapunov_time
    title_str = "$T_{\mu}=" + title_str
    title_str += ", \, T_{f}=" + "{:.2f}".format(
        model.prediction_horizon * dt) + "$"
    plt.title(r"{:}".format(title_str), pad=10)
    fig_path = utils.getFigureDir(
        model) + "/Comp_multiscale_{:}_micro{:}_{:}.{:}".format(
            set_name, int(micro_step), field_second, FIGTYPE)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    field_third = "speed_up_log"
    # CREATING BAR PLOT
    # print("micro_step\n{:}".format(micro_step))
    # Computing the speed-up
    result_plot = np.log10(result_plot)
    barlist = plt.bar(rho_plot, result_plot)
    barlist[-1].set_color("tab:red")

    for i_color in range(len(barlist[:-1])):
        barlist[i_color].set_color(color_labels[i_color])
    barlist[-1].set_color("tab:red")

    plt.ylabel(r"{:}".format("$\log_{10}($Speed-up$)$"))
    plt.xlabel(r"$ \rho=T_m / T_{\mu}$")
    title_str = "{:.2f}".format(micro_step * dt)
    # micro_step_time = micro_step * dt / lyapunov_time
    title_str = "$T_{\mu}=" + title_str
    title_str += ", \, T_{f}=" + "{:.2f}".format(
        model.prediction_horizon * dt) + "$"
    plt.title(r"{:}".format(title_str), pad=10)
    fig_path = utils.getFigureDir(
        model) + "/Comp_multiscale_{:}_micro{:}_{:}.{:}".format(
            set_name, int(micro_step), field_third, FIGTYPE)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plotMultiscaleResultsComparison(model, dicts_to_compare, set_name,
                                    fields_to_compare, dt):
    # print(fields_to_compare)
    # print(ark)
    # FIGTYPE = "pdf"
    FIGTYPE = "png"

    for field in fields_to_compare:
        macro_steps, micro_steps, result, result_iterative, result_equations = getMacroAndMicroSteps(
            model, dicts_to_compare, field, model.params["prediction_horizon"])

        micro_steps_set = set(micro_steps)
        for micro_step in micro_steps_set:

            if field in [
                    "state_dist_L1_hist_error",
                    "state_dist_wasserstein_distance",
            ]:
                """ Adding dummy extra space dimension (will be averaged inside the function) """
                result = np.expand_dims(result, axis=2)
                result_iterative = np.expand_dims(result_iterative, axis=1)

                makeBarPlot(
                    model,
                    field,
                    micro_steps,
                    micro_step,
                    macro_steps,
                    result,
                    result_iterative,
                    dt,
                    set_name,
                )

            if field in [
                    "CORR",
                    "RMSE",
                    "mnad_in",
                    "mnad_act",
            ]:

                makeBarPlot(
                    model,
                    field,
                    micro_steps,
                    micro_step,
                    macro_steps,
                    result,
                    result_iterative,
                    dt,
                    set_name,
                )

            if field in [
                    "CORR",
                    "RMSE",
                    "mnad_act",
                    "mnad_in",
            ]:

                for with_legend in [True, False]:

                    makeErrorTimePlot(
                        model,
                        field,
                        micro_steps,
                        micro_step,
                        macro_steps,
                        result,
                        result_iterative,
                        dt,
                        set_name,
                        with_legend,
                    )

            if field in ["time_total_per_iter"]:
                makeTimeBarPlot(model, field, micro_steps, micro_step,
                                macro_steps, result, result_iterative, dt,
                                set_name, result_equations)

            if field in ["num_accurate_pred_050_avg"]:
                pass

                # # print("micro_step\n{:}".format(micro_step))
                # indexes = micro_steps == micro_step
                # macro_steps_plot = macro_steps[indexes]
                # result_plot = result[indexes]

                # idx_sort = np.argsort(macro_steps_plot)
                # macro_steps_plot = macro_steps_plot[idx_sort]
                # result_plot = result_plot[idx_sort]

                # rho_plot = [
                #     "{:.2f}".format((temp / float(micro_step)))
                #     for temp in macro_steps_plot
                # ]
                # rho_plot.append(str("Latent"))
                # result_plot = np.concatenate(
                #     (result_plot, result_iterative[np.newaxis]), axis=0)
                # result_plot = result_plot * dt_scaled
                # # print(rho_plot)
                # # print(result_plot)
                # barlist = plt.bar(rho_plot, result_plot)
                # barlist[-1].set_color("tab:red")

                # for i_color in range(len(barlist[:-1])):
                #     barlist[i_color].set_color(color_labels[i_color])
                # barlist[-1].set_color("tab:red")

                # plt.ylabel(r"{:}".format(vpt_label))
                # plt.xlabel(r"$ \rho=T_m / T_{\mu}$")
                # title_str = "{:.2f}".format(micro_step * dt)
                # # micro_step_time = micro_step * dt / lyapunov_time
                # title_str = "$T_{\mu}=" + title_str
                # title_str += ", \, T_{f}=" + "{:.2f}".format(
                #     model.prediction_horizon * dt) + "$"
                # plt.title(r"{:}".format(title_str), pad=10)
                # fig_path = utils.getFigureDir(model) + "/Comp_multiscale_{:}_micro{:}_{:}.{:}".format(
                #     set_name, int(micro_step), field, FIGTYPE)
                # plt.tight_layout()
                # plt.savefig(fig_path, dpi=300)
                # plt.close()
