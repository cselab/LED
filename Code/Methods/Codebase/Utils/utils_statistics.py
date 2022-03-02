#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

# Integration for the errors on the densities
import scipy
from scipy import stats
from scipy.integrate import simps
import numpy as np


def getNumberOfBins(N, L=1.0, rule="rice"):
    ######################
    ## Diakonis rule
    ######################
    # IQR = 1
    # nbins = L / (2 * IQR) * np.power(N, 1.0/3.0)
    ######################
    ## Sturges rule
    ######################
    if rule == "sturges":
        nbins = int(1 + np.log2(N))
    ######################
    ## Rice rule
    ######################
    if rule == "rice":
        nbins = 2.0 * np.power(N, 1.0 / 3.0)
        nbins = int(np.ceil(nbins))
        nbins = int(np.max([2, nbins]))
    return nbins


def evaluateWassersteinDistance(data1, data2):
    if len(np.shape(data1)) == 1:
        error = stats.wasserstein_distance(data1, data2)
    elif len(np.shape(data1)) == 2:
        n, Dx = np.shape(data1)
        errors = []
        for d in range(Dx):
            error = stats.wasserstein_distance(data1[:, d], data2[:, d])
            errors.append(error)
        error = np.mean(np.array(errors))
    return error


def evaluateKSError(data1, data2):
    if len(np.shape(data1)) == 1:
        error = stats.ks_2samp(data1, data2)[0]
    elif len(np.shape(data1)) == 2:
        n, Dx = np.shape(data1)
        errors = []
        for d in range(Dx):
            error = stats.ks_2samp(data1[:, d], data2[:, d])[0]
            errors.append(error)
        error = np.mean(np.array(errors))
    return error


def evaluate2DIntegral(values, x, y):
    dx, dy = np.shape(values)
    assert (dx == np.shape(x)[0])
    assert (dy == np.shape(y)[0])
    return simps([simps(value, x) for value in values], y)


def evaluate3DIntegral(values, x, y, z):
    dx, dy, dz = np.shape(values)
    assert (dx == np.shape(x)[0])
    assert (dy == np.shape(y)[0])
    assert (dz == np.shape(z)[0])
    return simps(simps([simps(value, x) for value in values], y), z)


def get_density(positions, nbins, bounds):
    if len(np.shape(positions)) == 1:
        # 1-D case (no channels)
        # print("% Computing 1-D histogram. %")
        density, beans = np.histogram(positions, nbins, bounds, density=True)
        grid_centers = (beans[:-1] + beans[1:]) / 2
        return density, grid_centers
    elif (len(np.shape(positions)) == 2) and (np.shape(positions)[1] == 2):
        # 2-D case
        # print("% Computing 2-D histogram. %")
        if len(np.shape(bounds)) == 1:
            assert (np.shape(bounds)[0] == 2)
            data_range = [[bounds[0], bounds[1]], [bounds[0], bounds[1]]]
            equal_aspect = True
        elif len(np.shape(bounds)) == 2:
            assert (np.shape(bounds)[0] == 2)
            assert (np.shape(bounds)[1] == 2)
            data_range = bounds
            equal_aspect = False
        else:
            raise ValueError("I don't know how to select bounds.")
        H, xedges, yedges = np.histogram2d(positions[:, 0],
                                           positions[:, 1],
                                           nbins,
                                           density=True,
                                           range=data_range)
        if equal_aspect: assert (np.all(xedges == yedges))
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        grid_centers = np.array([xcenters, ycenters])
        return H, grid_centers
    elif (len(np.shape(positions)) == 2) and (np.shape(positions)[1] == 3):
        # print("% Computing 3-D histogram. %")
        H, edges = np.histogramdd(positions,
                                  nbins,
                                  density=True,
                                  range=[[bounds[0], bounds[1]],
                                         [bounds[0], bounds[1]],
                                         [bounds[0], bounds[1]]])
        xedges = edges[0]
        yedges = edges[1]
        zedges = edges[2]
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        zcenters = (zedges[:-1] + zedges[1:]) / 2
        grid_centers = np.array([xcenters, ycenters, zcenters])
        return H, grid_centers
    else:
        raise ValueError("Not implemented!")


def evaluateL1HistErrorVector(data1, data2, nbins, bounds):
    density1, grid_centers1 = get_density(data1, nbins, bounds)
    # print(np.shape(data1))
    # print(np.min(data1))
    # print(np.max(data1))
    # print(np.isfinite(data1).all())
    # print(bounds)
    # print(np.shape(data2))
    # print(np.min(data2))
    # print(np.max(data2))
    # print(np.isfinite(data2).all())
    # print(bounds)
    density2, grid_centers2 = get_density(data2, nbins, bounds)
    # print(grid_centers1)
    # print(grid_centers2)
    # print(np.linalg.norm(grid_centers1-grid_centers2))
    # assert(np.all(grid_centers1==grid_centers2))
    assert (np.linalg.norm(grid_centers1 - grid_centers2) < 1e-5)
    error_vec = np.abs(density1 - density2)
    if len(np.shape(data1)) == 1:
        # 1-D case (no channels)
        error = simps(error_vec, grid_centers1)
    elif (len(np.shape(data1)) == 2) and (np.shape(data1)[1] == 2):
        # 2-D case
        error = evaluate2DIntegral(error_vec, grid_centers1[0],
                                   grid_centers1[1])
    elif (len(np.shape(data1)) == 2) and (np.shape(data1)[1] == 3):
        # 3-D case
        error = evaluate3DIntegral(error_vec, grid_centers1[0],
                                   grid_centers1[1], grid_centers1[2])
    return error, error_vec, density1, density2, grid_centers1
