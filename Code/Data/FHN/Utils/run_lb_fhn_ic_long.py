#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import sys
import scipy.io
from Utils.lattice_boltzmann_fitzhugh_nagumo import *

def run_lb_fhn_ic_long(rho_act_0, rho_in_0, tf, dt_save=0.005):
    ###########################################
    ## Simulation of the Lattice Boltzman Method
    ## for the FitzHugh-Nagumo
    ###########################################

    # N = 40
    N = 100
    L = 20
    x = np.linspace(0, L, N+1)
    dx = x[1]-x[0]
    # print(x)
    # print(np.shape(x))

    Dx = 1 # Dact (activator)
    Dy = 4 # Din  (inhibitor)

    a0 = -0.03
    a1 = 2.0

    dt = 0.005

    N_timesteps = int(np.ceil(tf/dt))
    print("Number of total timesteps: {:}".format(N_timesteps))

    save_every = int(dt_save/dt)
    print("Saving every: {:}".format(save_every))

    omegas = [2/(1+3*Dx*dt/(dx*dx)), 2/(1+3*Dy*dt/(dx*dx))]
    n1 = 1/3

    # tf = 1000
    # tf = 10021


    # Bifurcation parameter
    epsilon = 0.006

    t = 0
    timestep = 1

    # Storing the density
    rho_act = []
    rho_in  = []
    t_vec   = []

    # Activator
    f1_act  = 1/3*rho_act_0
    f0_act  = f1_act
    f_1_act = f1_act

    rho_act_t   = f0_act + f1_act + f_1_act
    # mom_act_t   = f1_act - f_1_act
    # energ_act_t = 0.5 * (f1_act + f_1_act)

    # Inhibitor
    f1_in   = 1/3*rho_in_0
    f0_in   = f1_in
    f_1_in  = f1_in

    rho_in_t   = f0_in + f1_in + f_1_in
    # mom_in_t   = f1_in - f_1_in
    # energ_in_t = 0.5 * (f1_in + f_1_in)

    while timestep < N_timesteps:
        # print("Time {:.3f}/{:.2f}. {:.2f}%".format(t, tf, t/tf*100.0))

        # Propagate the Lattice Boltzmann in time
        f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in = LBM( \
            f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in, \
            omegas, a1, a0, epsilon, n1, dt \
            )

        # Updating Activator
        rho_act_t   = f0_act + f1_act + f_1_act

        # Updating Inhibitor
        rho_in_t   = f0_in + f1_in + f_1_in

        # print(timestep)
        # print(save_every)
        if (timestep % save_every < 1e-5) or np.abs(timestep % save_every - save_every)<1e-5:
            print("Time {:.3f}/{:.2f}. {:.2f}%".format(t, tf, t/tf*100.0))
            # print("Saviung")
            # print(t)
            # print(dt_save)
            # print(np.abs(t % dt_save))
            rho_act.append(rho_act_t)
            rho_in.append(rho_in_t)
            t_vec.append(t)

        t +=dt
        timestep += 1

    rho_act = np.array(rho_act)
    rho_in = np.array(rho_in)
    t_vec = np.array(t_vec)
    return rho_act, rho_in, t_vec
