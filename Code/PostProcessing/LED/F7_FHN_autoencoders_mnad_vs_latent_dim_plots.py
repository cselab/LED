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

font = {'size'   : 28, 'family':'Times New Roman'}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)


system_name="FHN"
Experiment_Name="Local"

saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)
logfile_path = "./FHN/FHN_result_prediction_ic1_horizon451/Logfiles"

print(system_name)
print(logfile_path)

FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
os.makedirs(FIGURES_PATH, exist_ok=True)
print(FIGURES_PATH)

filename =  "results_iterative_latent_forecasting_test.txt"
casename = "test"

modelname = "ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_7-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-C_lstm-R_1x32-SL_20-LFO_1-LFL_1"

FIELDS=[
"model_name",
"mnad_act_avg",
"mnad_in_avg",
]


PYTHON_TYPES=[str, float, float]
test_model_list, test_model_dict = utils.parseModelFields(logfile_path, FIELDS, PYTHON_TYPES, filename)
print("Number of {:} files processed {:}.".format(filename, len(test_model_list)))
# print(test_model_list)
test_model = test_model_list[0]


field2Label = {
    "mnad_act_avg":"MNAD$(u, \\tilde{u})$",
    "mnad_in_avg":"MNAD$(v, \\tilde{v})$",
}

# labels_led = ["CNN-LSTM (LED) $d_z=2$"]
labels_led = ["LED $d_z=2$"]


labels_kevr = [
"CSPDE-GP",
"CSPDE-NN",
"CSPDE-GP-F1",
"CSPDE-NN-F1",
"CSPDE-GP-F2",
"CSPDE-NN-F2",
# "CSPDE-GP-F3",
# "CSPDE-NN-F3",
]

# labels_kevr = ["Lee et. al. (2020), " + labels_kevr[i] for i in range(len(labels_kevr))]


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%.2E' % Decimal(height), ha='center', va='bottom')


def autolabelHorizontal(rects):
    """
    Attach a text label above each bar displaying its height
    """
    # margin = 0.03
    margin = 5 *1e-3
    # margin = 4 *1e-3
    # margin = 3.5 *1e-3
    for rect in rects:
        width = rect.get_width()
        # print(rect)
        y = rect.get_y()
        # y = rect.get_y() + rect.get_height()/4.
        # y = rect.get_y() + rect.get_height()/8.
        ax.text(width + margin, y, '%.2E' % Decimal(width), ha='center', va='bottom')



for field in field2Label:
    if field == "mnad_act_avg":
        data_kevr = [ 1.59E-02, 
                    1.53E-02,
                    1.58E-02,
                    1.54E-02,
                    2.39E-02,
                    2.00E-02,
                    # 3.20E-02,
                    # 2.08E-02
                    ]
    elif field == "mnad_in_avg":
        data_kevr = [ 1.62E-02, 
                    1.56E-02,
                    1.62E-02,
                    1.57E-02,
                    2.20E-02,
                    2.11E-02,
                    # 3.31E-02,
                    # 2.16E-02
                    ]

    idx = FIELDS.index(field)
    data_led = test_model[idx]
    label_var = field2Label[field]

    WIDTH = 1
    yfactor = 1

    fig_path = FIGURES_PATH + "/F7_FHN_CSPDE_comparison_{:}.pdf".format(field)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel(r"{:}".format(label_var))
     
    xdata_led = yfactor*np.arange(1, len(labels_led)+1)
    # print(xdata_led)

    xdata_kevr = yfactor*np.arange(len(labels_led)+1, len(labels_led)+1+len(labels_kevr))


    # print(xdata_kevr)
    # print(ark)
    # print(len(data_kevr))
    rects2 = ax.barh(xdata_kevr, data_kevr, color='green', hatch='', height=WIDTH, ecolor="white", edgecolor="white", label="Lee et. al. (2020)")
    rects1 = ax.barh(xdata_led, data_led, color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745), hatch='//', height=WIDTH, ecolor="white", edgecolor="white", label=labels_led[0])
    rects_all = [rects1, rects2]

    # rects1_all = []
    # rects2_all = []
    # xdataall = []
    # xlabelsall = []
    # for i in range(len(xdata)):
    #     rects2 = ax.barh(xdata[i]+WIDTH/2.0, interesting_loss_var[i], color='tab:blue', hatch='\\', height=WIDTH, ecolor="white", edgecolor="white", label="LED-VAE" if i==0 else None)
    #     rects2_all.append(rects2)
    #     xdataall.append(xdata[i]+WIDTH/2.0)
    #     xlabelsall.append(xlabels_var[i])

        
    #     rects1_all.append(rects1)
    #     xdataall.append(xdata[i]-WIDTH/2.0)
    #     xlabelsall.append(xlabels[i])

    
    xdata_all = list(xdata_led) + list(xdata_kevr) 
    xlabels_all = labels_led + labels_kevr
    ax.set_yticks(xdata_all)
    ax.set_yticklabels(xlabels_all)

    # ylims = [-1.5 * WIDTH, len(xdataall_kevr)-WIDTH/2.0]
    # ax.set_ylim(ylims)
    max_ = np.max([np.max(data_kevr), np.max(data_led)])
    # xlims = [0.0, 1.3 * max_]
    xlims = [0.0, 1.45 * max_]
    ax.set_xlim(xlims)
    
    # ax.set_axisbelow(True)
    # ax.grid(color='gray', linestyle='dashed')

    for bar in rects_all:
        autolabelHorizontal(bar)

    # plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_path)
    plt.close()
    # print(ark)