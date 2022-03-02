
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

print("-V- Matplotlib Version = {:}".format(matplotlib.__version__))


""" Selection of color pallete designed for colorblind people """
color_labels = [
# (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
# (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
# (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
# (0.8352941176470589, 0.3686274509803922, 0.0),
# (0.8, 0.47058823529411764, 0.7372549019607844),
(0.792156862745098, 0.5686274509803921, 0.3803921568627451),
(0.984313725490196, 0.6862745098039216, 0.8941176470588236),
(0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
(0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
(0.9254901960784314, 0.8823529411764706, 0.2),
]

linestyles = ['-','--','-.',':','-','--','-.',':']
linemarkers = ["x","o","s","d",">","<",">"]
linemarkerswidth = [4,2,2,2,2,2,2]
linemarkerssize = list(14 * np.ones_like(linemarkerswidth))

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



Experiment_Name="Experiment_Daint_Large"
# Experiment_Name="Experiment_Barry"


# set_ = "train"
# set_ = "test"

def field2Label(field):
    dict_ = {
        "MSE_avg": "\\mathrm{MSE}",
        "drag_coef_error_rel": "|C_d-\\tilde{C_d}|/|\\tilde{C_d}|",
    }
    assert field in dict_
    return dict_[field]

# for system_name in ["FHN"]:
for system_name in [
    # "FHN",
    # "KSGP64L22",
    # "KSGP64L22Large",
    # "cylRe100",
    # "cylRe1000",
    "cylRe100HR",
    "cylRe1000HR",
    "cylRe100HRDt005",
    "cylRe1000HRDt005",
    ]:

    if system_name == "FHN":
        model_names = \
        [
        "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        # "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_5.0-DMN_10-LD_(.*)",
        "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_1.0-DMN_10-LD_(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_2-8-16-32-4-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        ]

        labels = \
        [
        "PCA",
        # "DiffMaps w=5.0",
        # "DiffMaps",
        "DiffMaps w=1.0",
        "AE",
        "CNN-AE",
        ]


    elif system_name == "KSGP64L22Large":
        model_names = \
        [
        "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        # "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_5.0-DMN_10-LD_(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        ]
        labels = \
        [
        "PCA",
        # "DiffMaps w=1.0",
        "AE",
        "CNN-AE",
        # "CNN-AE T",
        ]
        del color_labels[1]
        del linestyles[1]
        del linemarkers[1]
        del linemarkerswidth[1]
        del linemarkerssize[1]

    elif system_name == "KSGP64L22":
        model_names = \
        [
        "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        "GPU-DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_5.0-DMN_10-LD_(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_30-CHANNELS_1-16-32-64-8-KERNELS_5-5-5-5-BN_0-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        ]

        labels = \
        [
        "PCA",
        "DiffMaps w=1.0",
        "AE",
        "CNN-AE",
        ]

    elif system_name in ["cylRe100"]:

        model_names = \
        [
        "DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        "DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_1.0-DMN_5-LD_(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        ]

        labels = \
        [
        "PCA",
        "DiffMaps",
        "CNN-AE",
        ]

    elif system_name in ["cylRe1000"]:

        model_names = \
        [
        "DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        "DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_1.0-DMN_5-LD_(.*)",
        "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-32-8-KERNELS_11-9-7-5-3-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        ]

        labels = \
        [
        "PCA",
        "DiffMaps",
        "CNN-AE",
        ]
    elif system_name in [
    "cylRe100HRDt005",
    "cylRe1000HRDt005",
    ]:

        model_names = \
        [
        "DimRed-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        # "DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        # "DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_1.0-DMN_5-LD_(.*)",
        "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        ]

        labels = \
        [
        "PCA",
        # "DiffMaps",
        "CNN-AE",
        ]

    elif system_name in [
    "cylRe100HR",
    "cylRe1000HR",
    ]:

        model_names = \
        [
        "DimRedRNN-scaler_MinMaxZeroOne-METHOD_pca-LD_(.*)",
        # "DimRedRNN-scaler_MinMaxZeroOne-METHOD_diffmaps-DMW_1.0-DMN_5-LD_(.*)",
        "GPU-ARNN-SC_MinMaxZeroOne-OPT_adabelief-PREC_single-LR_0.001-NL_0.0-L2_0.0-RS_10-CNL_4-20-20-20-20-20-2-KRN_13-13-13-13-13-13-BN_1-TR_1-SI_2-PL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        ]

        labels = \
        [
        "PCA",
        # "DiffMaps",
        "CNN-AE",
        ]

        # model_names = \
        # [
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-8-8-8-8-1-KERNELS_13-13-13-13-13-13-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-8-8-8-8-1-KERNELS_13-13-13-13-13-13-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-8-8-8-8-1-KERNELS_13-13-13-13-13-13-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-8-8-8-8-1-KERNELS_13-13-13-13-13-13-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        # # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-8-8-8-8-1-KERNELS_13-13-13-13-13-13-BN_1-AF_1-TRANS_0-POOL_avg-ACTOUT_identity-ACT_celu-DKP_1.0-LD_(.*)",
        # # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-8-8-8-8-1-KERNELS_13-13-13-13-13-13-BN_1-AF_1-TRANS_0-POOL_avg-ACTOUT_identity-ACT_celu-DKP_1.0-LD_(.*)",
        # # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-8-8-8-8-1-KERNELS_13-13-13-13-13-13-BN_1-AF_1-TRANS_0-POOL_avg-ACTOUT_identity-ACT_celu-DKP_1.0-LD_(.*)",
        # ]

        
        # labels = \
        # [
        # "CNN-AE 13 BN1 T1 celu",
        # "CNN-AE 13 BN0 T1 celu",
        # "CNN-AE 13 BN1 T0 celu",
        # "CNN-AE 13 BN1 T1 celu",
        # ]
    elif system_name in ["cylRe1000HR"]:

        # model_names = \
        # [
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-8-KERNELS_11-11-7-5-BN_1-TRANS_0-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        # "GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_4-8-16-16-8-KERNELS_11-11-7-5-BN_1-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_(.*)",
        # ]

        # labels = \
        # [
        # "CNN-AE",
        # "CNN-AE-T",
        # ]
        pass
    else:
        raise ValueError("Unknown system {:}.".format(system_name))


    saving_path = utils.getSavingPath(Experiment_Name, system_name, global_params)
    logfile_path = saving_path + global_params.logfile_dir

    print(system_name)
    print(logfile_path)

    FIGURES_PATH = "./{:}/{:}/Figures".format(system_name, Experiment_Name)
    os.makedirs(FIGURES_PATH, exist_ok=True)


    FIELDS=[
    "model_name",
    "MSE_avg",
    "drag_coef_error_rel",
    ]

    PYTHON_TYPES=[
    str,
    float,
    float,
    ]
    max_x = 20

    # set_ = "test"
    # set_ = "val"

    for field2plot in ["MSE_avg", "drag_coef_error_rel"]:

        idx_field = FIELDS.index(field2plot)
        label_field = field2Label(field2plot)
        # for set_ in ["test", "val", "train"]:
        for set_ in ["test", "val"]:
            

            data_dict = {}
            for modelnum in range(len(model_names)):
                modelname = model_names[modelnum]


                if "DimRedRNN" in modelname or "DimRed" in modelname:
                    filename = "results_dimred_testing_{:}".format(set_)
                else:
                    filename = "results_autoencoder_testing_{:}".format(set_)

                test_model_list, test_model_dict = utils.parseModelFields(logfile_path, FIELDS, PYTHON_TYPES, filename)

                model_regex            = re.compile(modelname)

                x_data = []
                y_data = []
                for model in test_model_list:
                    modelname_ = model[0]
                    if model_regex.search(modelname_):
                        # Get latent dimension
                        latent_dim = int(modelname_.split("-LD_")[1].split("-")[0])
                        if latent_dim < max_x:

                            x_data.append(latent_dim)
                            value_ = model[idx_field]
                            y_data.append(value_)

                assert modelname not in data_dict
                data_dict.update({
                                 modelname:{"x_data":x_data,
                                 "y_data":y_data,
                                 "label":labels[modelnum],
                                 "color":color_labels[modelnum],
                                 "marker":linemarkers[modelnum],
                                 "markerwidth":linemarkerswidth[modelnum],
                                 "markersize":linemarkerssize[modelnum],
                                 }})

            fig_path = FIGURES_PATH + "/F1_DimRed_MSE_wrt_latent_dim.pdf"
            fig, ax = plt.subplots()
            ax.set_xlabel(r"Latent dimension")
            ax.set_ylabel(r"{:}".format("MSE"))
            for modelname in model_names:
                plt.plot(data_dict[modelname]["x_data"], 
                         data_dict[modelname]["y_data"],
                         data_dict[modelname]["marker"],                
                         color=data_dict[modelname]["color"],
                         markeredgewidth=data_dict[modelname]["markerwidth"],
                         label=data_dict[modelname]["label"],
                         )
            # if legend_str=="_legend":
            plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

            for legend_str in ["", "_legend"]:
                fig_path = FIGURES_PATH + "/F1_DimRed_{:}_wrt_latent_dim{:}_{:}.pdf".format(field2plot, legend_str, set_)
                fig, ax = plt.subplots()
                ax.set_xlabel(r"Latent dimension")
                ax.set_ylabel(r"{:}".format("$" + label_field + "$"))
                plt.grid()
                for modelname in model_names:
                    plt.plot(data_dict[modelname]["x_data"], 
                             data_dict[modelname]["y_data"],
                             data_dict[modelname]["marker"],                
                             color=data_dict[modelname]["color"],
                             markeredgewidth=data_dict[modelname]["markerwidth"],
                             markersize=data_dict[modelname]["markersize"],
                             label=data_dict[modelname]["label"],
                             )
                if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()

            for legend_str in ["", "_legend"]:
                fig_path = FIGURES_PATH + "/F1_DimRed_{:}_wrt_latent_dim_log{:}_{:}.pdf".format(field2plot, legend_str, set_)
                fig, ax = plt.subplots()
                ax.set_xlabel(r"Latent dimension")
                ax.set_ylabel(r"{:}".format("$\log_{10}(" + label_field + ")$"))
                plt.grid()
                label_field
                for modelname in model_names:
                    plt.plot(data_dict[modelname]["x_data"], 
                             np.log10(data_dict[modelname]["y_data"]),
                             data_dict[modelname]["marker"],                
                             color=data_dict[modelname]["color"],
                             markeredgewidth=data_dict[modelname]["markerwidth"],
                             markersize=data_dict[modelname]["markersize"],
                             label=data_dict[modelname]["label"],
                             )

                if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()


            # for legend_str in ["", "_legend"]:
            #     fig_path = FIGURES_PATH + "/DimRed_MSE_wrt_latent_dim_log{:}_{:}_e.pdf".format(legend_str, set_)
            #     fig, ax = plt.subplots()
            #     ax.set_xlabel(r"Latent dimension")
            #     ax.set_ylabel(r"{:}".format("$\log_{e}(\mathrm{MSE})$"))
            #     plt.grid()
            #     for modelname in model_names:
            #         plt.plot(data_dict[modelname]["x_data"], 
            #                  np.log(data_dict[modelname]["y_data"]),
            #                  # np.log(data_dict[modelname]["y_data"]),
            #                  data_dict[modelname]["marker"],                
            #                  color=data_dict[modelname]["color"],
            #                  markeredgewidth=data_dict[modelname]["markerwidth"],
            #                  label=data_dict[modelname]["label"],
            #                  )
            #     if legend_str=="_legend": plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            #     plt.tight_layout()
            #     plt.savefig(fig_path)
            #     plt.close()

