#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import sys
import scipy.io
from Utils.lattice_boltzmann_fitzhugh_nagumo import *

def run_lb_fhn_ic(rho_act_0, rho_in_0, tf):
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

    omegas = [2/(1+3*Dx*dt/(dx*dx)), 2/(1+3*Dy*dt/(dx*dx))]
    n1 = 1/3

    # tf = 1000
    # tf = 10021


    # Bifurcation parameter
    epsilon = 0.006

    t = 0
    it = 0

    N_T = int(np.ceil(tf/dt))

    # Storing the density
    rho_act = np.zeros((N_T, N+1))
    rho_in  = np.zeros((N_T, N+1))
    t_vec   = np.zeros((N_T))

    # Storing momentum terms
    mom_act = np.zeros((N_T, N+1))
    mom_in  = np.zeros((N_T, N+1))

    # Storing energy terms
    energ_act = np.zeros((N_T, N+1))
    energ_in  = np.zeros((N_T, N+1))

    # Initial density loaded from file
    # mat = scipy.io.loadmat('./mail2_Spiliotis_Kostas/y0initRelax40.mat')
    # # Initial density
    # rho0 = np.array(mat["y0"])
    # rho_act_0 = rho0[:N+1,0]
    # rho_in_0  = rho0[N+1:,0]

    # # Random initial conditions for inhibitor/activator
    # rho_in_0    = np.random.rand(N+1)
    # rho_act_0   = np.random.rand(N+1)



    # Activator
    f1_act  = 1/3*rho_act_0
    f0_act  = f1_act
    f_1_act = f1_act

    rho_act_t   = f0_act + f1_act + f_1_act
    mom_act_t   = f1_act - f_1_act
    energ_act_t = 0.5 * (f1_act + f_1_act)

    # Inhibitor
    f1_in   = 1/3*rho_in_0
    f0_in   = f1_in
    f_1_in  = f1_in

    rho_in_t   = f0_in + f1_in + f_1_in
    mom_in_t   = f1_in - f_1_in
    energ_in_t = 0.5 * (f1_in + f_1_in)

    while np.abs(t-tf)>1e-6:
        print("Time {:.3f}/{:.2f}. {:.2f}%".format(t, tf, t/tf*100.0))

        # Propagate the Lattice Boltzmann in time
        f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in = LBM( \
            f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in, \
            omegas, a1, a0, epsilon, n1, dt \
            )

        # Updating Activator
        rho_act_t   = f0_act + f1_act + f_1_act
        mom_act_t   = f1_act - f_1_act
        energ_act_t = 0.5 * (f1_act + f_1_act)

        # Updating Inhibitor
        rho_in_t   = f0_in + f1_in + f_1_in
        mom_in_t   = f1_in - f_1_in
        energ_in_t = 0.5 * (f1_in + f_1_in)

        rho_act[it]     = rho_act_t
        rho_in[it]      = rho_in_t
        t_vec[it]       = t
        mom_act[it]     = mom_act_t
        mom_in[it]      = mom_in_t
        energ_act[it]   = energ_act_t
        energ_in[it]    = energ_in_t

        it+=1
        t +=dt

    return rho_act, rho_in, t_vec, mom_act, mom_in, energ_act, energ_in, dt, N, L, dx, x, Dx, Dy, a0, a1, n1, omegas, tf, a0
