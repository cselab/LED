#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import pickle
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

import h5py

with open("./Simulation_Data/lattice_boltzmann_fhn.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    simdata = pickle.load(file)
    rho_act_all = np.array(simdata["rho_act_all"])
    rho_in_all = np.array(simdata["rho_in_all"])
    dt_coarse = simdata["dt_data"]
    del simdata

rho_act_all = np.expand_dims(rho_act_all, axis=2)
rho_in_all = np.expand_dims(rho_in_all, axis=2)
print(np.shape(rho_act_all))
print(np.shape(rho_in_all))

rho_all = np.concatenate((rho_act_all, rho_in_all), axis=2)

print(np.shape(rho_all))
sequences_raw = rho_all

data_dir_scaler = "./Data"
os.makedirs(data_dir_scaler, exist_ok=True)

data_max = np.max(sequences_raw, axis=(0,1,3))
data_min = np.min(sequences_raw, axis=(0,1,3))

print(np.shape(data_max))
print(np.shape(data_min))
np.savetxt(data_dir_scaler + "/data_max.txt", data_max)
np.savetxt(data_dir_scaler + "/data_min.txt", data_min)

""" Save the timestep """
np.savetxt(data_dir_scaler + "/dt.txt", [dt_coarse])


# Create training data
N_TRAIN = 451

ICS_TRAIN = [0, 1, 2]
N_ICS_TRAIN=len(ICS_TRAIN)
sequences_raw_train = sequences_raw[ICS_TRAIN, :N_TRAIN]


sequence_length = 121

data_dir_str            = "train"
batch_size  = 32






idxs_timestep = []
idxs_ic = []

for ic in range(N_ICS_TRAIN):
    seq_data = sequences_raw_train[ic]
    idxs = np.arange(0, np.shape(seq_data)[0]- sequence_length, 1)
    idxs = np.random.permutation(idxs)

    for idx_ in idxs:
        idxs_ic.append(ic)
        idxs_timestep.append(idx_)


max_batches = int(np.floor(len(idxs_ic)/batch_size))
print("Number of sequences = {:}/{:}".format(max_batches*batch_size, len(idxs_ic)))
num_sequences = max_batches*batch_size

# print(idxs_ic)
# print(idxs_timestep)
sequences   = []
for bn in range(num_sequences):
    idx_ic = idxs_ic[bn]
    idx_timestep = idxs_timestep[bn]

    sequence = sequences_raw_train[idx_ic, idx_timestep:idx_timestep+sequence_length]
    sequences.append(sequence)
    # print(np.shape(sequence))

sequences = np.array(sequences) 

print("sequences.shape")
print(np.shape(sequences))

data_dir = "./Data/{:}".format(data_dir_str)
os.makedirs(data_dir, exist_ok=True)

hf = h5py.File(data_dir + '/data.h5', 'w')
# Only a single sequence_example per dataset group
for seq_num_ in range(np.shape(sequences)[0]): # (960, 121, 202)
    # print('batch_{:010d}'.format(seq_num_))
    data_group = sequences[seq_num_]
    data_group = np.array(data_group)
    # print(np.shape(data_group))
    gg = hf.create_group('batch_{:010d}'.format(seq_num_))
    gg.create_dataset('data', data=data_group)
hf.close()

data_dir_raw = "./Data/{:}_raw".format(data_dir_str)
os.makedirs(data_dir_raw, exist_ok=True)
hf = h5py.File(data_dir_raw + '/data.h5', 'w')
for seq_num_ in range(np.shape(sequences_raw_train)[0]): # (960, 121, 202)
    data_group = sequences_raw_train[seq_num_]
    data_group = np.array(data_group)
    gg = hf.create_group('batch_{:010d}'.format(seq_num_))
    gg.create_dataset('data', data=data_group)
hf.close()






N_VAL=451
ICS_VAL = [3, 4]
N_ICS_VAL=len(ICS_VAL)
sequences_raw_val = sequences_raw[ICS_VAL, :N_VAL]

sequence_length = 121

data_dir_str            = "val"
batch_size  = 32

idxs_timestep = []
idxs_ic = []

for ic in range(N_ICS_VAL):
    seq_data = sequences_raw_val[ic]
    idxs = np.arange(0, np.shape(seq_data)[0]- sequence_length, 1)
    idxs = np.random.permutation(idxs)

    for idx_ in idxs:
        idxs_ic.append(ic)
        idxs_timestep.append(idx_)


max_batches = int(np.floor(len(idxs_ic)/batch_size))
print("Number of sequences = {:}/{:}".format(max_batches*batch_size, len(idxs_ic)))
num_sequences = max_batches*batch_size

# print(idxs_ic)
# print(idxs_timestep)
sequences   = []
for bn in range(num_sequences):
    idx_ic = idxs_ic[bn]
    idx_timestep = idxs_timestep[bn]

    sequence = sequences_raw_val[idx_ic, idx_timestep:idx_timestep+sequence_length]
    sequences.append(sequence)
    # print(np.shape(sequence))

sequences = np.array(sequences) 

print("sequences.shape")
print(np.shape(sequences))

data_dir = "./Data/{:}".format(data_dir_str)
os.makedirs(data_dir, exist_ok=True)

hf = h5py.File(data_dir + '/data.h5', 'w')
# Only a single sequence_example per dataset group
for seq_num_ in range(np.shape(sequences)[0]): # (960, 121, 202)
    # print('batch_{:010d}'.format(seq_num_))
    data_group = sequences[seq_num_]
    data_group = np.array(data_group)
    # print(np.shape(data_group))
    gg = hf.create_group('batch_{:010d}'.format(seq_num_))
    gg.create_dataset('data', data=data_group)
hf.close()


data_dir_raw = "./Data/{:}_raw".format(data_dir_str)
os.makedirs(data_dir_raw, exist_ok=True)
hf = h5py.File(data_dir_raw + '/data.h5', 'w')
for seq_num_ in range(np.shape(sequences_raw_val)[0]):
    data_group = sequences_raw_val[seq_num_]
    data_group = np.array(data_group)
    gg = hf.create_group('batch_{:010d}'.format(seq_num_))
    gg.create_dataset('data', data=data_group)
hf.close()




