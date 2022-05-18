#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
from numpy import loadtxt
from Utils.run_lb_fhn_ic import *

# u is the Activator
# v is the Inhibitor
import time

start = time.time()

file_names = ["y00", "y01", "y02", "y03", "y04", "y05"]
# file_names = ["y00"]

rho_act_all = []
rho_in_all = []
t_vec_all = []
mom_act_all = []
mom_in_all = []
energ_act_all = []
energ_in_all = []


for file_name in file_names:
    file_name_act = "./InitialConditions/" + file_name + "u.txt"
    rho_act_0 = loadtxt(file_name_act, delimiter="\n")

    file_name_in = "./InitialConditions/" + file_name + "v.txt"
    rho_in_0 = loadtxt(file_name_in, delimiter="\n")

    file_name_x = "./InitialConditions/y0x.txt"
    x = loadtxt(file_name_x, delimiter="\n")
    # print(x)
    # print(np.shape(x))
    # print(np.shape(rho_act_0))
    # print(np.shape(rho_in_0))
    tf = 10021

    rho_act, rho_in, t_vec, mom_act, mom_in, energ_act, energ_in, dt, N, L, dx, x, Dx, Dy, a0, a1, n1, omegas, tf, a0 = run_lb_fhn_ic(rho_act_0, rho_in_0, tf)

    # Removing initial transients of Dt=2
    DT_trans = 2
    n_trans = int(DT_trans/dt)
    rho_act = rho_act[n_trans:]
    rho_in = rho_in[n_trans:]
    SUBSAMPLING = 200
    # Removing periodic boundary conditions and subsampling
    rho_act = rho_act[::SUBSAMPLING]
    rho_in = rho_in[::SUBSAMPLING]
    print(np.shape(rho_act))
    print(np.shape(rho_in))


    rho_act_all.append(rho_act)
    rho_in_all.append(rho_in)

end = time.time()
total_time = end - start
total_time = total_time / float(len(file_names))

print("Time per IC run:")
print(total_time)

dt = dt * SUBSAMPLING
data = {
    "total_time":total_time,
    "rho_act_all":rho_act_all,
    "rho_in_all":rho_in_all,
    "t_vec_all":t_vec_all,
    "dt":dt,
}

with open("./Simulation_Data/lattice_boltzmann_fhn_test.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)



