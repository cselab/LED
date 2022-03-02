#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from io import StringIO
import sys, os
import io
import joblib
from joblib import Parallel, delayed

from . import utils_networks


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range]
                for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def smoothDimension(data_ic, window_size, polyorder):
    N = np.shape(data_ic)[0]
    temp = Parallel(n_jobs=10)(
        delayed(savitzky_golay)(data_ic[i], window_size, polyorder)
        for i in range(N))
    temp = np.array(temp)
    return temp


def smoothLatentSpace(model, training_data, window_size, polyorder):
    if window_size <= 1:
        return training_data

    if len(np.shape(training_data)) == 3:
        multiple_ics = True
    else:
        training_data = [training_data]
        multiple_ics = False

    # print(np.shape(training_data)) # (64, 120, 4)
    training_data = np.swapaxes(training_data, 1, 2)
    training_data = np.swapaxes(training_data, 0, 1)
    D, N, T = np.shape(training_data)

    training_data = Parallel(n_jobs=4)(
        delayed(smoothDimension)(training_data[j], window_size, polyorder)
        for j in range(D))

    training_data = np.array(training_data)

    training_data = np.swapaxes(training_data, 0, 1)
    training_data = np.swapaxes(training_data, 1, 2)
    # print(np.shape(training_data))

    if not multiple_ics: training_data = training_data[0]
    return training_data


def interpolateLatentSpace(model, data, interp_factor, dt):
    n_ics, T, D = np.shape(data)
    dt_fine = dt / interp_factor

    time = np.linspace(0, T - 1, num=T, endpoint=True) * dt
    time_fine = np.linspace(
        0, T - 1, num=(interp_factor * (T - 1) + 1), endpoint=True) * dt
    # print(time[1]-time[0])
    # print(time_fine[1]-time_fine[0])
    data_interp = [[
        interp1d(time, data[ic, :, d], kind='cubic')(time_fine)
        for d in range(D)
    ] for ic in range(n_ics)]
    data_interp = np.array(data_interp)
    data_interp = np.swapaxes(data_interp, 1, 2)
    data = [data_interp[i] for i in range(len(data_interp))]
    return data, time_fine, dt_fine


def plotTrainingExamplesAndGetError(model,
                                    time_fine,
                                    training_data,
                                    dt_sindy,
                                    plot=True):
    # print("# plotTrainingExample() #")
    # print(np.shape(training_data))

    if len(np.shape(training_data)) == 2:
        training_data = [training_data]

    if plot:
        traj_plot = np.min([np.shape(training_data)[0], 2])
        # Multiple trajectories
        for traj in range(traj_plot):
            x_test = training_data[traj]
            # for x_test in
            t_test = time_fine
            # Predict derivatives using the learned model
            x_dot_test_predicted = model.model_sindy.predict(x_test)
            # Compute derivatives with a finite difference method, for comparison
            x_dot_test_computed = model.model_sindy.differentiate(x_test,
                                                                  t=dt_sindy)

            fig, axs = plt.subplots(x_test.shape[1],
                                    1,
                                    sharex=True,
                                    figsize=(7, 9))
            if x_test.shape[1] == 1:
                for i in range(x_test.shape[1]):
                    axs.plot(t_test,
                             x_dot_test_computed[:, i],
                             'k',
                             label='numerical derivative')
                    axs.plot(t_test,
                             x_dot_test_predicted[:, i],
                             'r--',
                             label='model prediction')
                    axs.legend()
                    axs.set(xlabel='t', ylabel='$\dot z_{}$'.format(i))
            else:
                for i in range(x_test.shape[1]):
                    axs[i].plot(t_test,
                                x_dot_test_computed[:, i],
                                'k',
                                label='numerical derivative')
                    axs[i].plot(t_test,
                                x_dot_test_predicted[:, i],
                                'r--',
                                label='model prediction')
                    axs[i].legend()
                    axs[i].set(xlabel='t', ylabel='$\dot z_{}$'.format(i))
            # fig.show()
            # plt.show()
            fig.savefig(
                utils_networks.getFigureDir(model) +
                "/test_on_train_deriv_traj{:}.png".format(traj))

            # t_test = t_test[:400]
            # x_test = x_test[:400]
            # print(np.shape(x_test))
            # print(x_test[0])
            x_test_predicted = model.model_sindy.simulate(x_test[0], t_test)
            fig, axs = plt.subplots(x_test.shape[1],
                                    1,
                                    sharex=True,
                                    figsize=(7, 9))
            if x_test.shape[1] == 1:
                for i in range(x_test.shape[1]):
                    axs.plot(t_test, x_test[:, i], 'k', label='data')
                    axs.plot(t_test,
                             x_test_predicted[:, i],
                             'r--',
                             label='model prediction')
                    axs.set(xlabel='t', ylabel='$z_{}$'.format(i))
                    axs.legend()
            else:
                for i in range(x_test.shape[1]):
                    axs[i].plot(t_test, x_test[:, i], 'k', label='data')
                    axs[i].plot(t_test,
                                x_test_predicted[:, i],
                                'r--',
                                label='model prediction')
                    axs[i].set(xlabel='t', ylabel='$z_{}$'.format(i))
                    axs[i].legend()
            # fig.show()
            # plt.show()
            fig.savefig(
                utils_networks.getFigureDir(model) +
                "/test_on_train_iter_traj{:}.png".format(traj))

            # t_test = t_test[:400]
            # x_test = x_test[:400]

            # print(x_test[0])
            x_test_predicted_teacher = []
            prediction_length = len(x_test) - 10
            for t in range(prediction_length):
                # for t in range(20):
                temp_ = model.model_sindy.simulate(x_test[t], t_test[t:t + 2])
                # temp_ = model.model_sindy.simulate(x_test[t], t_test[t:t+8])
                x_test_predicted_teacher.append(temp_[-1])
            x_test_predicted_teacher = np.array(x_test_predicted_teacher)
            # print(np.shape(x_test_predicted_teacher))

            t_test = t_test[:prediction_length]
            x_test = x_test[:prediction_length]

            fig, axs = plt.subplots(x_test.shape[1],
                                    1,
                                    sharex=True,
                                    figsize=(7, 9))
            if x_test.shape[1] == 1:
                for i in range(x_test.shape[1]):
                    axs.plot(t_test, x_test[:, i], 'k', label='data')
                    axs.plot(t_test,
                             x_test_predicted_teacher[:, i],
                             'r--',
                             label='model prediction')
                    axs.set(xlabel='t', ylabel='$z_{}$'.format(i))
                    axs.legend()
            else:
                for i in range(x_test.shape[1]):
                    axs[i].plot(t_test, x_test[:, i], 'k', label='data')
                    axs[i].plot(t_test,
                                x_test_predicted_teacher[:, i],
                                'r--',
                                label='model prediction')
                    axs[i].set(xlabel='t', ylabel='$z_{}$'.format(i))
                    axs[i].legend()
            # fig.show()
            # plt.show()
            fig.savefig(
                utils_networks.getFigureDir(model) +
                "/test_on_train_teacher_traj{:}.png".format(traj))

    num_traj_evaluate = np.min([np.shape(training_data)[0], 10])
    print("Evaluating on {:} trajectories.".format(num_traj_evaluate))
    error = 0.0
    for traj in range(num_traj_evaluate):
        x_test = training_data[traj]
        t_test = time_fine
        x_test_predicted = model.model_sindy.simulate(x_test[0], t_test)
        error += np.mean(np.abs(x_test_predicted - x_test))
    error /= len(training_data)
    return error


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def writeLinesToLogfile(logfile, lines):
    with io.open(logfile, 'w') as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    return 0
