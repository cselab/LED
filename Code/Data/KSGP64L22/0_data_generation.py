import numpy as np
from numpy import pi
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp2d
from Utils import KS

import h5py
import os

#------------------------------------------------------------------------------
# define data and initialize simulation
L    = 22/(2*pi)
N    = 64
# dt   = 0.25
# dt   = 0.025
dt   = 0.00025

ttransient = 1000
tend = 12000 + ttransient
# ttransient = 0
# tend = 120 + ttransient
dns  = KS.KS(L=L, N=N, dt=dt, tend=tend)

#------------------------------------------------------------------------------
# simulate initial transient
dns.simulate()
# convert to physical space
dns.fou2real()
# get field
u = dns.uu

""" Get rid of initial transients """
ntransientsteps = int(ttransient/dt)
u           = u[ntransientsteps:]


data = {
    "u":u,
    "L":L,
    "N":N,
    "dt":dt,
}
os.makedirs("./Simulation_Data/", exist_ok=True)

with open("./Simulation_Data/ks_sim.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)



