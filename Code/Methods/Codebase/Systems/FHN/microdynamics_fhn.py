#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np
""" lattice boltzmann method for microdynamics of FHN """
def LBM(f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in, \
        omegas, a1, a0, epsilon, n1, dt):

    #############################
    # Collision terms (omega)
    #############################

    # Activator
    rho_act_t = f0_act + f1_act + f_1_act
    ome_act = omegas[0]

    omega1_act = -ome_act * (f1_act - n1 * rho_act_t)
    omega_1_act = -ome_act * (f_1_act - n1 * rho_act_t)
    omega0_act = -ome_act * (f0_act - n1 * rho_act_t)

    # Inhibitor
    rho_in_t = f0_in + f1_in + f_1_in
    ome_in = omegas[1]

    omega1_in = -ome_in * (f1_in - n1 * rho_in_t)
    omega_1_in = -ome_in * (f_1_in - n1 * rho_in_t)
    omega0_in = -ome_in * (f0_in - n1 * rho_in_t)

    #############################
    # Reaction terms (R)
    #############################

    reac1_act = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)
    reac_1_act = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)
    reac0_act = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)

    reac1_in = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)
    reac_1_in = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)
    reac0_in = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)

    #############################
    # Updating Activator terms
    #############################

    f1_act_next = np.zeros_like(f1_act)
    f1_act_temp = f1_act + omega1_act + reac1_act
    f1_act_next[1:] = f1_act_temp[:-1]
    f1_act_next[0] = f1_act[0] + omega1_act[0] + reac1_act[0]

    f0_act_next = np.zeros_like(f0_act)
    f0_act_temp = f0_act + omega0_act + reac0_act
    f0_act_next = f0_act_temp

    f_1_act_next = np.zeros_like(f_1_act)
    f_1_act_temp = f_1_act + omega_1_act + reac_1_act
    f_1_act_next[:-1] = f_1_act_temp[1:]
    f_1_act_next[-1] = f1_act_next[-1]

    #############################
    # Updating Inhibitor terms
    #############################

    f1_in_next = np.zeros_like(f1_in)
    f1_in_temp = f1_in + omega1_in + reac1_in
    f1_in_next[1:] = f1_in_temp[:-1]
    f1_in_next[0] = f1_in[0] + omega1_in[0] + reac1_in[0]

    f0_in_next = np.zeros_like(f0_in)
    f0_in_temp = f0_in + omega0_in + reac0_in
    f0_in_next = f0_in_temp

    f_1_in_next = np.zeros_like(f_1_in)
    f_1_in_temp = f_1_in + omega_1_in + reac_1_in
    f_1_in_next[:-1] = f_1_in_temp[1:]
    f_1_in_next[-1] = f1_in_next[-1]

    return f1_act_next, f_1_act_next, f0_act_next, f1_in_next, f_1_in_next, f0_in_next


def evolveFitzHughNagumo(u0, total_time, dt_coarse):
    ###########################################
    ## Simulation of the Lattice Boltzman Method
    ## for the FitzHugh-Nagumo
    ###########################################
    N = 100
    L = 20
    """
    Input from the network, of the form (K, input_dim, D)
    """
    assert len(np.shape(u0)) == 3
    K, channels, D = np.shape(u0)
    assert K == 1
    assert channels == 2
    assert D == 101

    # rho_act_0 = u0[0, :N + 1]
    # rho_in_0 = u0[0, N + 1:]

    rho_act_0 = u0[0, 0]
    rho_in_0 = u0[0, 1]

    assert (np.shape(rho_act_0)[0] == N + 1)
    assert (np.shape(rho_in_0)[0] == N + 1)

    x = np.linspace(0, L, N + 1)
    dx = x[1] - x[0]

    Dx = 1  # Dact (activator)
    Dy = 4  # Din  (inhibitor)

    a0 = -0.03
    a1 = 2.0

    dt = 0.005

    omegas = [
        2 / (1 + 3 * Dx * dt / (dx * dx)), 2 / (1 + 3 * Dy * dt / (dx * dx))
    ]
    n1 = 1 / 3

    tf = total_time

    # Bifurcation parameter
    epsilon = 0.006

    t = 0
    it = 0

    N_T = int(np.ceil(tf / dt))

    # Storing the density
    rho_act = np.zeros((N_T, N + 1))
    rho_in = np.zeros((N_T, N + 1))
    t_vec = np.zeros((N_T))

    # Storing momentum terms
    mom_act = np.zeros((N_T, N + 1))
    mom_in = np.zeros((N_T, N + 1))

    # Storing energy terms
    energ_act = np.zeros((N_T, N + 1))
    energ_in = np.zeros((N_T, N + 1))

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
    f1_act = 1 / 3 * rho_act_0
    f0_act = f1_act
    f_1_act = f1_act

    rho_act_t = f0_act + f1_act + f_1_act
    mom_act_t = f1_act - f_1_act
    energ_act_t = 0.5 * (f1_act + f_1_act)

    # Inhibitor
    f1_in = 1 / 3 * rho_in_0
    f0_in = f1_in
    f_1_in = f1_in

    rho_in_t = f0_in + f1_in + f_1_in
    mom_in_t = f1_in - f_1_in
    energ_in_t = 0.5 * (f1_in + f_1_in)

    while np.abs(t - tf) > 1e-6:
        # print("Time {:.3f}/{:.2f}. {:.2f}%".format(t, tf, t/tf*100.0))

        # Propagate the Lattice Boltzmann in time
        f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in = LBM( \
            f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in, \
            omegas, a1, a0, epsilon, n1, dt \
            )

        # Updating Activator
        rho_act_t = f0_act + f1_act + f_1_act
        mom_act_t = f1_act - f_1_act
        energ_act_t = 0.5 * (f1_act + f_1_act)

        # Updating Inhibitor
        rho_in_t = f0_in + f1_in + f_1_in
        mom_in_t = f1_in - f_1_in
        energ_in_t = 0.5 * (f1_in + f_1_in)

        rho_act[it] = rho_act_t
        rho_in[it] = rho_in_t
        t_vec[it] = t
        mom_act[it] = mom_act_t
        mom_in[it] = mom_in_t
        energ_act[it] = energ_act_t
        energ_in[it] = energ_in_t

        it += 1
        t += dt

    rho_act = np.transpose(rho_act[np.newaxis], (1, 0, 2))
    rho_in = np.transpose(rho_in[np.newaxis], (1, 0, 2))
    u = np.concatenate((rho_act, rho_in), axis=1)
    u = np.concatenate((u0, u), axis=0)
    """ Subsampling """
    subsample = int(dt_coarse / dt)
    u = u[::subsample]
    """ Removing the first time-step """
    u = u[1:]

    return u
