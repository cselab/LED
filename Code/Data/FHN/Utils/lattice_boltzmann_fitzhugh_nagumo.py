#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np

def LBM(f1_act, f_1_act, f0_act, f1_in, f_1_in, f0_in, \
        omegas, a1, a0, epsilon, n1, dt):

    #############################
    # Collision terms (omega)
    #############################

    # Activator
    rho_act_t = f0_act + f1_act + f_1_act
    ome_act = omegas[0]

    omega1_act  = -ome_act * (f1_act  - n1 * rho_act_t)
    omega_1_act = -ome_act * (f_1_act - n1 * rho_act_t)
    omega0_act  = -ome_act * (f0_act  - n1 * rho_act_t)

    # Inhibitor
    rho_in_t = f0_in + f1_in + f_1_in
    ome_in = omegas[1]

    omega1_in  = -ome_in * (f1_in  - n1 * rho_in_t)
    omega_1_in = -ome_in * (f_1_in - n1 * rho_in_t)
    omega0_in  = -ome_in * (f0_in  - n1 * rho_in_t)

    #############################
    # Reaction terms (R)
    #############################

    reac1_act  = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)
    reac_1_act = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)
    reac0_act  = n1 * dt * (rho_act_t - np.power(rho_act_t, 3.0) - rho_in_t)

    reac1_in   = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)
    reac_1_in  = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)
    reac0_in   = n1 * dt * epsilon * (rho_act_t - a1 * rho_in_t - a0)

    #############################
    # Updating Activator terms
    #############################

    f1_act_next     = np.zeros_like(f1_act)
    f1_act_temp     = f1_act + omega1_act + reac1_act
    f1_act_next[1:] = f1_act_temp[:-1]
    f1_act_next[0]  = f1_act[0] + omega1_act[0] + reac1_act[0]

    f0_act_next     = np.zeros_like(f0_act)
    f0_act_temp     = f0_act + omega0_act + reac0_act
    f0_act_next     = f0_act_temp

    f_1_act_next    = np.zeros_like(f_1_act)
    f_1_act_temp    = f_1_act + omega_1_act + reac_1_act
    f_1_act_next[:-1]= f_1_act_temp[1:]
    f_1_act_next[-1] = f1_act_next[-1]

    #############################
    # Updating Inhibitor terms
    #############################

    f1_in_next     = np.zeros_like(f1_in)
    f1_in_temp     = f1_in + omega1_in + reac1_in
    f1_in_next[1:] = f1_in_temp[:-1]
    f1_in_next[0]  = f1_in[0] + omega1_in[0] + reac1_in[0]

    f0_in_next     = np.zeros_like(f0_in)
    f0_in_temp     = f0_in + omega0_in + reac0_in
    f0_in_next     = f0_in_temp

    f_1_in_next    = np.zeros_like(f_1_in)
    f_1_in_temp    = f_1_in + omega_1_in + reac_1_in
    f_1_in_next[:-1]= f_1_in_temp[1:]
    f_1_in_next[-1] = f1_in_next[-1]

    return f1_act_next, f_1_act_next, f0_act_next, f1_in_next, f_1_in_next, f0_in_next






