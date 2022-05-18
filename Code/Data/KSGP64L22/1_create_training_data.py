import numpy as np
from numpy import pi
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp2d
from Utils import KS

import h5py
import os

with open("./Simulation_Data/ks_sim.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    simdata = pickle.load(file)
    u = np.array(simdata["u"])
    L = np.array(simdata["L"])
    N = np.array(simdata["N"])
    dt = simdata["dt"]
    del simdata

dt_coarse = 0.25
subsample = int(dt_coarse/dt)

u           = u[::subsample]
dt_coarse          = subsample*dt
print(np.shape(u))


trajectory = u[:, np.newaxis]

data_max = np.max(trajectory, axis=(0,2))
data_min = np.min(trajectory, axis=(0,2))

print(np.shape(data_max))
print(np.shape(data_min))

data_dir_scaler =  "./Data"
os.makedirs(data_dir_scaler, exist_ok=True)
np.savetxt(data_dir_scaler + "/data_max.txt", data_max)
np.savetxt(data_dir_scaler + "/data_min.txt", data_min)

""" Save the timestep """
np.savetxt(data_dir_scaler + "/dt.txt", [dt_coarse])


timesteps_train = 15000
timesteps_val   = 15000
timesteps_test   = 15000

data_train = trajectory[:timesteps_train]
data_val = trajectory[timesteps_train:timesteps_train+timesteps_val]
data_test = trajectory[timesteps_train+timesteps_val:timesteps_train+timesteps_val+timesteps_test]





print("#"*20)
print(np.shape(data_train))
print(np.shape(data_val))
print(np.shape(data_test))
print("#"*20)

print("Test data start at time-step {:}".format(timesteps_train+timesteps_val))

num_ics_train   = 128
num_ics_val     = 128
num_ics_test    = 100

timestep_per_sequence_train     = 1001
timestep_per_sequence_val         = 1001
timestep_per_sequence_test        = 8001


num_datasets    = 3
data_dirs       = ["train", "val", "test"]
data_all            = [data_train, data_val, data_test]
num_ics_per_dataset_all = [num_ics_train, num_ics_val, num_ics_test]
timestep_per_sequence_all = [timestep_per_sequence_train, timestep_per_sequence_val, timestep_per_sequence_test]

# num_datasets    = 1
# data_dirs       = ["test"]
# data_all            = [data_test]
# num_ics_per_dataset_all = [num_ics_test]
# timestep_per_sequence_all = [timestep_per_sequence_test]



for dataset_num in range(num_datasets):

    data_dir_str    = data_dirs[dataset_num]
    data            = data_all[dataset_num]
    num_ics_per_dataset            = num_ics_per_dataset_all[dataset_num]
    timestep_per_sequence            = timestep_per_sequence_all[dataset_num]

    sequences       = []

    idxs = np.arange(np.shape(data)[0] - timestep_per_sequence)
    print("Number of possible ics {:}/{:}".format(num_ics_per_dataset, len(idxs)))




    data_dir = "./Data/{:}".format(data_dir_str)
    os.makedirs(data_dir, exist_ok=True)
    hf = h5py.File(data_dir + '/data.h5', 'w')

    for seq_num in range(num_ics_per_dataset):
        idx = np.random.choice(idxs)
        sequence    = data[idx:idx+timestep_per_sequence]

        idxs = set(idxs)
        idxs.remove(idx)
        idxs = list(idxs)

        print('batch_{:06d}'.format(seq_num))
        data_group =sequence
        data_group = np.array(data_group)
        print(np.shape(data_group))
        gg = hf.create_group('batch_{:06d}'.format(seq_num))
        gg.create_dataset('data', data=data_group)

    hf.close()






    """ Creating raw data sequence (needed for training SINDy) """
    data_dir_raw = "./Data/{:}_raw".format(data_dir_str)
    os.makedirs(data_dir_raw, exist_ok=True)
    hf = h5py.File(data_dir_raw + '/data.h5', 'w')
    data_group = data
    data_group = np.array(data_group)
    gg = hf.create_group('batch_{:010d}'.format(0))
    gg.create_dataset('data', data=data_group)
    hf.close()



    # channel=0
    # u_plot = sequence[:, channel]

    # N_plot = np.shape(u_plot)[0]
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # s, n = np.meshgrid(np.arange(N_plot)*dt_coarse, 2*pi*L/N*(np.array(range(N))+1))
    # cs = plt.contourf(s, n, u_plot.T, 50, cmap=plt.get_cmap("seismic"))
    # plt.colorbar()
    # plt.ylabel(r"$u$")
    # plt.xlabel(r"$t$")
    # for c in cs.collections: c.set_rasterized(True)
    # fig_dir = "./Figures/{:}".format(data_dir_str)
    # os.makedirs(fig_dir, exist_ok=True)
    # plt.savefig(fig_dir + "/Trajectory_plot_{:d}.png".format(N_plot), dpi=300)



