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
print("[utils_plotting_ks] PLOTTING HOSTNAME: {:}".format(hostname))
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

print("[utils_plotting_ks] CLUSTER={:}, CLUSTER_NAME={:}".format(
    CLUSTER, CLUSTER_NAME))

if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

print("[utils_plotting_ks] Matplotlib Version = {:}".format(
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

FIGTYPE = "png"
# FIGTYPE="pdf"

from .. import Utils as utils


def plotStateDistributionsSystemKS(model, results, set_name, testing_mode):

    ux_uxx_density_target = results["ux_uxx_density_target"]
    ux_uxx_density_predicted = results["ux_uxx_density_predicted"]
    ux_uxx_l1_hist_error = results["ux_uxx_l1_hist_error"]
    ux_uxx_l1_hist_error_vec = results["ux_uxx_l1_hist_error_vec"]
    ux_uxx_wasserstein_distance = results["ux_uxx_wasserstein_distance"]
    ux_uxx_mesh = results["ux_uxx_mesh"]

    with np.errstate(divide='ignore', invalid='ignore'):
        ux_uxx_density_target = np.log(ux_uxx_density_target)
        ux_uxx_density_predicted = np.log(ux_uxx_density_predicted)
        ux_uxx_l1_hist_error_vec = np.log(ux_uxx_l1_hist_error_vec)
    ncols = 2
    nrows = 1
    vmin = np.nanmin(ux_uxx_density_target[ux_uxx_density_target != -np.inf])
    vmax = np.nanmax(ux_uxx_density_target)
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    contours_vec = []
    axes[0, 0].set_title("Target Density")
    mp = axes[0, 0].contourf(ux_uxx_mesh[0],
                             ux_uxx_mesh[1],
                             ux_uxx_density_target.T,
                             60,
                             cmap=plt.get_cmap("Reds"),
                             levels=np.linspace(vmin, vmax, 60))
    fig.colorbar(mp, ax=axes[0, 0])
    contours_vec.append(mp)
    axes[0, 0].set_ylabel(r"$u_{xx}$")
    axes[0, 0].set_xlabel(r"$u_{x}$")

    axes[0, 1].set_title("Predicted Density")
    mp = axes[0, 1].contourf(ux_uxx_mesh[0],
                             ux_uxx_mesh[1],
                             ux_uxx_density_predicted.T,
                             60,
                             cmap=plt.get_cmap("Reds"),
                             levels=np.linspace(vmin, vmax, 60))
    fig.colorbar(mp, ax=axes[0, 1])
    contours_vec.append(mp)
    axes[0, 1].set_ylabel(r"$u_{xx}$")
    axes[0, 1].set_xlabel(r"$u_{x}$")

    for contours in contours_vec:
        for pathcoll in contours.collections:
            pathcoll.set_rasterized(True)

    fig.tight_layout()
    fig_path = utils.getFigureDir(model) + "/{:}_ux_uxx_distr_{:}.{:}".format(
        testing_mode, set_name, FIGTYPE)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    # plt.hist2d(targets_ux, targets_uxx, (nbins,nbins), norm=LogNorm(), cmap=plt.get_cmap("Reds"), normed=True)
    # plt.hist2d(targets_ux, targets_uxx, bins=(100,100), normed=True)
