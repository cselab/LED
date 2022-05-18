#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import pickle
import numpy as np
import os
import h5py

with open("./Simulation_Data/lattice_boltzmann_fhn_test.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    simdata = pickle.load(file)
    rho_act_all = np.array(simdata["rho_act_all"])
    rho_in_all = np.array(simdata["rho_in_all"])
    dt = simdata["dt"]
    del simdata

print(np.shape(rho_act_all))
print(np.shape(rho_in_all))

rho_act_all = np.expand_dims(rho_act_all, axis=2)
rho_in_all = np.expand_dims(rho_in_all, axis=2)
print(np.shape(rho_act_all))
print(np.shape(rho_in_all))

# rho_all = np.concatenate((rho_act_all[:,:,:,np.newaxis], rho_in_all[:,:,:,np.newaxis]), axis=3)
sequences_raw = np.concatenate((rho_act_all, rho_in_all), axis=2)

print(np.shape(sequences_raw))



# N_TEST = 1002
ICS_TEST = [5]
sequences_raw_test = sequences_raw[ICS_TEST]
sequences_raw_test = sequences_raw_test[0]

sequence_length = 9000

data_dir_str            = "test"
batch_size  = 1

idxs_timestep = []
idxs_ic = []

# N_ICS_TEST=len(ICS_TEST)
N_ICS_TEST=100

idxs = np.arange(0, np.shape(sequences_raw_test)[0]- sequence_length, 1)
idxs = np.random.permutation(idxs)


sequences   = []
for ic in range(N_ICS_TEST):

    if ic ==0:
        idx_start = 0
    else:
        idx_start = idxs[ic]

    print("idx_start = {:}".format(idx_start))

    sequence = sequences_raw_test[idx_start:idx_start+sequence_length]
    sequences.append(sequence)
    print("idx_start = {:}, np.shape(sequence)={:}".format(idx_start, np.shape(sequence)))

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








