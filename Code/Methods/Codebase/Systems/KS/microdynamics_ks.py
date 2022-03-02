#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
from .ks_solver.KS import *

import time


def evolveKSGP64L22(u0, total_time, dt_model):
    """
    Input from the network, of the form (K, input_dim, D)
    """
    assert len(np.shape(u0)) == 3
    K, channels, D = np.shape(u0)
    assert K == 1
    assert channels == 1
    assert D == 64

    u0 = np.reshape(u0, (-1))
    #------------------------------------------------------------------------------
    # define data and initialize simulation
    L = 22 / (2 * pi)
    N = 64
    dt = 0.00025
    # time0 = time.time()
    dns = KS(L=L, N=N, dt=dt, tend=total_time, u0=u0)
    # time1 = time.time()
    # print("Initialization time : {:}".format(time1-time0))
    #------------------------------------------------------------------------------
    # time0 = time.time()
    # simulate initial transient
    dns.simulate()
    # time1 = time.time()
    # print("Simulation time : {:}".format(time1-time0))
    # convert to physical space
    dns.fou2real()
    # print(np.shape(u0))
    # print(np.shape(dns.uu[0]))
    # print(np.linalg.norm(u0-dns.uu[0]))
    # print(ark)
    # print(dt_model)
    subsample = int(dt_model / dt)
    # u = dns.uu[1:]
    u = dns.uu
    # Subsampling
    u = u[::subsample]
    # Removing the first time-step
    u = u[1:]
    # print(np.shape(u))
    # print(ark)
    u = np.reshape(u, (-1, 1, N))
    return u
