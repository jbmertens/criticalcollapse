#!/usr/bin/env python
# coding: utf-8

import sys, os, re
from importlib import reload 
import subprocess
import numpy as np
import scipy.optimize as opt

import ctypes
c_lib = ctypes.CDLL('cpp/ms.so')
c_real_t = ctypes.c_double
c_bool_t = ctypes.c_bool
c_real_ptr_t = ctypes.POINTER(c_real_t)

c_lib.P_of_rho.restype = c_real_t
c_lib.P_of_rho.argtypes = [c_real_t]
c_lib.dPdrho.restype = c_real_t
c_lib.dPdrho.argtypes = [c_real_t]

c_lib.rho_background.restype = c_real_t
c_lib.rho_background.argtypes = [c_real_t]
c_lib.P_background.restype = c_real_t
c_lib.P_background.argtypes = [c_real_t]

c_lib.G.restype = c_real_t
c_lib.G.argtypes = [c_real_t]


c_lib.ics.argtypes = [ c_real_ptr_t, c_real_ptr_t, c_real_ptr_t, c_real_ptr_t,
    c_real_ptr_t, c_real_t, c_real_t, ctypes.c_int, c_real_t, c_bool_t ]
c_lib.ics.restype = None

c_lib.run_sim.argtypes = [ c_real_ptr_t, c_real_ptr_t, c_real_ptr_t, c_real_ptr_t, c_real_ptr_t,
                          ctypes.c_int, ctypes.c_int,
                          c_bool_t, c_real_t, c_bool_t, c_bool_t,
                          ctypes.c_int, c_real_t, c_real_t, c_real_t]
c_lib.run_sim.restype = ctypes.c_int

c_lib.regrid.argtypes = [ c_real_ptr_t, c_real_t, c_real_t ]
c_lib.regrid.restype = None
c_lib.agg_pop.argtypes = [ c_real_ptr_t, c_real_t ]
c_lib.agg_pop.restype = None


def find_mass(
    l_simstart=0,
    l_simeq=0,
    amp = 0.7162501041727418,
    steps=2000000,
    N=6400,
    Ld=32.0,
    USE_FIXW=False,
    q_mult=0.1,
    TOL=1.0e-8,
) :
    deltaH = c_real_t(-2)
    max_rho0 = c_real_t(0)
    bh_mass = c_real_t(0)
    agg = (c_real_t*(N*13))()
    l = c_real_t(l_simstart)

    c_lib.ics(agg, ctypes.byref(l), ctypes.byref(deltaH), ctypes.byref(max_rho0), ctypes.byref(bh_mass),
              amp*c_lib.G(l_simstart)/c_lib.G(l_simeq), np.exp(l_simeq), N, Ld, USE_FIXW)

    result = c_lib.run_sim(agg, ctypes.byref(l), ctypes.byref(deltaH), ctypes.byref(max_rho0), ctypes.byref(bh_mass),
                        steps, -1, False, q_mult, True, True, -400, 1.0, 0.001, TOL)
    print(result, l, c_lib.G(l_simstart)/c_lib.G(l_simeq), deltaH, max_rho0, bh_mass)

    fields = np.reshape(np.copy(agg), (13, N))

    return ( result, l, c_lib.G(l_simstart)/c_lib.G(l_simeq), deltaH, max_rho0, bh_mass )

runs = []
for i in np.arange(10) :
    amp = 0.7162501041727418 + i*0.0001
    runs.append(find_mass( l_simstart=0, l_simeq=0, amp=amp ))
