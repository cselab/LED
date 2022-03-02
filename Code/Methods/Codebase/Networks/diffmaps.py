#!/usr/bin/env python3
# File              : diffusion_map.py
# Author            : George Arampatzis <garampat@ethz.ch>
# Date              : 29.06.2021
# Last Modified Date: 29.06.2021
# Last Modified By  : George Arampatzis <garampat@ethz.ch>
# Copyright 2021 ETH Zurich. All Rights Reserved.
#
# Class the implements diffusion maps, lift and restrict operators.
# The code has been trascripted from matlab. Original code is here:
# https://github.com/sandstede-lab/eq-free

import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.optimize import least_squares
from numpy.linalg import norm
from scipy.linalg import eig


class DiffusionMap:
    """
    Diffusion Maps with Restriction and Lift operators.
    """
    def __init__(self, data, d, weight=5, neighbors=10):
        """
          Constructor

          Arguments:
            data: array-like object of size (N,D), contains N data points of
                  dimension D.
            d: integer, dimension of the reduced space
            weight: integer, hyperparameter that multiplies the median of the data
        """
        """ Transposing the data """
        data = data.T

        self.D = data.shape[0]
        self.Ns = data.shape[1]

        self.data = data
        self.d = d
        self.neighbors = neighbors

        A = squareform(pdist(data.T))

        self.epsilon = weight * np.median(A.flatten())

        A = np.exp(-np.power(A, 2) / self.epsilon**2)

        A = A / A.sum(axis=1, keepdims=1)

        eigval, eigvec = eig(A, overwrite_a=True)

        eigval = np.real(eigval)
        eigvec = np.real(eigvec)

        index = eigval.argsort()[::-1]
        eigval = eigval[index]
        eigvec = eigvec[:, index]

        self.eigval = eigval[1:self.d + 1]
        self.eigval = self.eigval[..., np.newaxis]

        eigvec = eigvec[:, 1:self.d + 1]

        sg = np.diag(np.sign(eigvec[0, :]))
        self.eigvec = np.matmul(eigvec, sg)

    def testRestrict(self):
        """
        Computes and returns the error of the restriction operator.
        For each data point, it removes the data point and the corresponding
        eigenvector value from the database. It then restricts the data point 
        and compares the restriction with the removed eigenvector value.
        """
        restricted = np.zeros((self.Ns, self.d))
        restrictDiff = np.zeros((self.Ns, 1))
        percentError = np.zeros((self.Ns, 1))

        for i in range(self.Ns):
            x = self.data[:, i]
            x = x[..., np.newaxis]
            restricted[i, :] = np.squeeze(self.restrictVal(x, i))
            restrictDiff[i] = norm(self.eigvec[i, :] - restricted[i, :])
            percentError[i] = restrictDiff[i] / norm(self.eigvec[i, :])

        return (percentError, restricted, restrictDiff)

    def testLift(self, k):
        """
        Computes and returns the error of the lift operator.

        Arguments:
        k: integer, number of neighbors in the original space.
        """
        restricted = np.zeros((self.Ns, self.d))
        restrictDiff = np.zeros((self.Ns, 1))
        percentError = np.zeros((self.Ns, 1))

        for i in range(self.Ns):
            x = self.eigvec[i, :]
            x = x[np.newaxis, ...]
            lifted = self.lift(x, k)
            restricted[i, :] = np.squeeze(self.restrict(lifted))
            restrictDiff[i] = norm(self.eigvec[i, :] - restricted[i, :])
            percentError[i] = restrictDiff[i] / norm(self.eigvec[i, :])

        return (percentError, restricted, restrictDiff)

    def _restrict(self, newDataPoint, data, eigenVecs):
        """
        Restriction of a point from the original space to the lower dimensional
        space. The restriction is done through the Nystrom extension.

        Arguments:
        newDataPoint: array-like object of size (D,1), contains the new datapoint
        to be restricted
        data: array-like object of size (D,N), contains N datapoints of dimension D.
        eigenVecs: array-like object of size (N,d), corresponds to the truncated
        eigenvectors of the Markov transition matrix that is based on 'data'.

        Returns:
        array-like object of size (d,1), contains the restricted point
        """
        dist = cdist(newDataPoint.T, data.T).T
        w = np.exp(-np.power(dist, 2) / self.epsilon**2)
        w = w / w.sum(axis=0, keepdims=1)
        return np.matmul(eigenVecs.T, w) / self.eigval

    def restrictVal(self, newDataPoint, i):
        """
        Wrapper of the _restrict method. Computes the restriction of the 'newDataPoint'
        by removing the i-th data point from the original matrix and the corresponding
        values from the eigenvalues.

        Arguments:
        newDataPoint: array-like object of size (D,1), contains the new datapoint
        to be restricted
        i: integer, index of that data-point to be removed from the original dataset (self.data)

        Returns:
        array-like object of size (d,1), contains the restricted point
        """
        data_new = np.delete(self.data, (i), axis=1)
        eigvec_new = np.delete(self.eigvec, (i), axis=0)

        return self._restrict(newDataPoint, data_new, eigvec_new)

    def restrict(self, newDataPoint):
        """
        Wrapper of the _restrict method. Computes the restriction of the 'newDataPoint'
        using all the data from the original data-set (self.data).

        Arguments:
        newDataPoint: array-like object of size (D,1), contains the new datapoint
        to be restricted

        Returns:
        array-like object of size (d,1), contains the restricted point
        """

        return self._restrict(newDataPoint, self.data, self.eigvec)

    def lift(self, newDataPoint, k):
        """
        Lift a point from the low dimensional space (dimension d) to the original space 
        (dimension D) using the approach presented here: 
        http://www-users.math.umn.edu/~pcarter/publications/EquationFreeModeling.pdf
        "Enabling equation-free modeling via diffusion maps"

        Arguments:
        newDataPoint: array-like object of size (d,1), contains the new datapoint
        to be restricted
        k: integer, number of neighbors in the original space.

        Returns:
        array-like object of size (D,1), contains the lifted point
        """
        dist = np.squeeze(cdist(newDataPoint, self.eigvec))
        index = np.argsort(dist)
        index = index[:k]

        closestDataPoints = self.data[:, index]

        fun = lambda x: self.liftingEquations(x, closestDataPoints,
                                              newDataPoint)

        x0 = np.ones((k, )) / k

        bounds = [np.zeros((k, )), np.ones((k, ))]

        res = least_squares(fun, x0, bounds=bounds, verbose=0)

        coeffs = res['x']
        coeffs = coeffs[..., np.newaxis]

        return np.matmul(closestDataPoints, coeffs)

    def liftingEquations(self, coeffs, closestPoints, newVal):
        """
        The objective function that is used in the 'lift' method. For more information
        see the reference in the description of 'lift'.

        Arguments:
        coeffs: array-like of size (k,1)
        closestPoints: array-like of size (D,k)
        newVal: array-like of size (1,d)

        Returns:
        array-like of size (k+1,)
        """
        coeffs = coeffs[..., np.newaxis]
        liftGuess = np.matmul(closestPoints, coeffs)
        restrictGuess = np.squeeze(self.restrict(liftGuess))
        res = restrictGuess - np.squeeze(newVal)

        return np.append(res, np.sum(coeffs) - 1.)

    def transform(self, data):
        """
        Perform restriction on a number of samples.

        Arguments:
        data: array-like object of size (N,D), contains the datapoints to be restricted

        Returns:
        array-like object of size (N,d), contains the restricted points
        """
        return self.restrict(data.T).T

    def inverse_transform(self, data, neighbors=None):
        if neighbors is None:
            neighbors = self.neighbors

        data_rec = []
        for datap in data:
            temp = np.reshape(datap, (1, -1))
            temp = self.lift(temp, neighbors).T
            assert np.shape(temp)[0] == 1
            data_rec.append(temp[0])
        data_rec = np.array(data_rec)
        return data_rec
