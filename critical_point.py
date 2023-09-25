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


def min_gammab2(amp, l_simstart, l_simeq, USE_FIXW=False, Ld=30, N=3200) :
    max_rho0 = c_real_t(0)
    deltaH = c_real_t(-2)
    bh_mass = c_real_t(0)
    agg = (c_real_t*(N*13))()
    l = c_real_t(l_simstart)
    c_lib.ics(agg, ctypes.byref(l), ctypes.byref(deltaH), ctypes.byref(max_rho0), ctypes.byref(bh_mass),
              amp*c_lib.G(l_simstart)/c_lib.G(l_simeq), # Amplitude scaled down by the relative growth factor
              np.exp(l_simeq), # Abar scale set by the equality factor
              N, Ld, USE_FIXW)
    agg = np.reshape(np.copy(agg), (13, N))
    gammab2 = agg[6];
    return np.min(gammab2)

def max_phys_amp(l_simstart, l_simeq, USE_FIXW=False, Ld=30, N=3200) :
    return opt.root( min_gammab2, 0.5, (l_simstart, l_simeq, USE_FIXW, Ld, N) ).x[0]


def find_crit(iters=12,
    l_simstart=0,
    l_simeq=0,
    lower_amp=-1,
    upper_amp=-1,
    steps=2000000,
    N=800,
    Ld=32.0,
    USE_FIXW=False,
    q_mult=0.25,
    TOL=1.0e-7,
    failstop=False
):
    """
    Binary search between lower and upper to find a critical amplitude
    (Note that this is NOT the critical density)
    return (critical, upper value)
    """
    if lower_amp < 0 :
        lower_amp = 0.2*max_phys_amp(l_simstart, l_simeq, USE_FIXW, Ld, N)
    if upper_amp < 0 :
        upper_amp = 0.9*max_phys_amp(l_simstart, l_simeq, USE_FIXW, Ld, N)
    upper_fields = -1
    lower_fields = -1
    bisection_factor = 1/2
    for i in range(iters):
        try :
            amp_diff = upper_amp - lower_amp
            middle_amp = upper_amp - amp_diff*bisection_factor
            print('Iteration No', str(i), '-- Checking to see if a BH forms at amplitude', str(middle_amp),
                ' ( bracket is ', lower_amp, '/', upper_amp, ')')

            deltaH = c_real_t(-2)
            max_rho0 = c_real_t(0)
            bh_mass = c_real_t(0)
            agg = (c_real_t*(N*13))()
            l = c_real_t(l_simstart)
           
            c_lib.ics(agg, ctypes.byref(l), ctypes.byref(deltaH), ctypes.byref(max_rho0), ctypes.byref(bh_mass),
                      middle_amp*c_lib.G(l_simstart)/c_lib.G(l_simeq), np.exp(l_simeq), N, Ld, USE_FIXW)

            result = c_lib.run_sim(agg, ctypes.byref(l), ctypes.byref(deltaH), ctypes.byref(max_rho0), ctypes.byref(bh_mass),
                                steps, -1, True, q_mult, True, True, -400, 1.0, 0.001, TOL)
            print(result, l, c_lib.G(l_simstart)/c_lib.G(l_simeq), deltaH, max_rho0)
            
            fields = np.reshape(np.copy(agg), (13, N))

            if result == 0 : # Ran out of steps
                print("Ran out of steps. Breaking.")
                break
            elif result <= 1 and failstop : # Some sort of error during run
                print("Stopping on failure.")
                break
            elif result <= 1 : # Some sort of error during run
                print("Encountered error during run. Trying again with a different amplitude.")
                # change bisection factor (don't divide just in half; walk in by 1/4, 1/8, ... from the ends.)
                if bisection_factor >= 1/2 :
                    bisection_factor = (1-bisection_factor)/2
                else :
                    bisection_factor = 1-bisection_factor
            else :
                bisection_factor = 1/2
                if(result == 3):
                    upper_amp = middle_amp
                    upper_fields = fields
                elif(result == 2):
                    lower_amp = middle_amp
                    lower_fields = fields

        except Exception as e :
            print("Run failed! Stopping search. Reason below.")
            print(e)
            break
    
    print("Critical amplitude appears to be between", lower_amp, "and", upper_amp)

    return (lower_amp, upper_amp)


l_simstart_str = sys.argv[1]
l_simstart = float(sys.argv[1])
l_simeq_str = sys.argv[2]
l_simeq = float(sys.argv[2])
N = 400
N_max = 6400

lower_amp = -1.0
upper_amp = -1.0

prev_run_file = "output/run1_"+l_simeq_str+"_"+l_simstart_str+".txt"
if os.path.exists(prev_run_file) :
    print("Old file found,", prev_run_file)
    with open(prev_run_file, 'r') as fp :
        lines = fp.readlines()
        for row in lines:
            if row.find('Former bounds are') != -1 :
                print(row)
            if row.find('Final bounds are') != -1 :
                matches = re.findall(r"[0-9\.]+", row)
                lower_amp = float(matches[0])
                upper_amp = float(matches[1])
                print("Former bounds are", lower_amp, upper_amp)
                # broaden bounds a bit...
                delta_amp = upper_amp - lower_amp
                upper_amp = upper_amp + 3*delta_amp
                lower_amp = lower_amp - 2*delta_amp
            if row.find('gridpoints, amplitude') != -1 :
                old_N = re.findall(r"[0-9]+", row)
                new_N = int(old_N[0])*2
                if new_N > N_max :
                    new_N = N_max
                if N != new_N :
                    print("Old N was", N, "-- using new N", new_N)
                    N = new_N
else :
    print("Old file NOT found,", prev_run_file)


print("Using bounds (", lower_amp, upper_amp, ")")

while True :
    print("Searching for CP with", N, lower_amp, upper_amp)
    if N == N_max :
        lower_amp, upper_amp = find_crit(iters=4, l_simstart=l_simstart, l_simeq=l_simeq,
                  lower_amp=lower_amp, upper_amp=upper_amp,
                  N=N, USE_FIXW=False, q_mult=0.2, TOL=1e-8, failstop=False)
        lower_amp, upper_amp = find_crit(iters=4, l_simstart=l_simstart, l_simeq=l_simeq,
                  lower_amp=lower_amp, upper_amp=upper_amp,
                  N=N, USE_FIXW=False, q_mult=0.15, TOL=8e-9, failstop=False)
        lower_amp, upper_amp = find_crit(iters=4, l_simstart=l_simstart, l_simeq=l_simeq,
                  lower_amp=lower_amp, upper_amp=upper_amp,
                  N=N, USE_FIXW=False, q_mult=0.1, TOL=5e-9, failstop=False)
        break
    else :
        # Narrow in bounds
        if N >= N_max//2 :
            # Getting close...
            lower_amp, upper_amp = find_crit(iters=6, l_simstart=l_simstart, l_simeq=l_simeq,
                      lower_amp=lower_amp, upper_amp=upper_amp,
                      N=N, USE_FIXW=False, q_mult=0.2, TOL=3e-8, failstop=False)
        else :
            lower_amp, upper_amp = find_crit(iters=4, l_simstart=l_simstart, l_simeq=l_simeq,
                      lower_amp=lower_amp, upper_amp=upper_amp,
                      N=N, USE_FIXW=False, q_mult=0.25, TOL=3e-8, failstop=False)
        # Broaden a bit, since the resolution can change when it is increased
        delta_amp = upper_amp - lower_amp
        upper_amp = upper_amp + 4*delta_amp
        lower_amp = lower_amp - 2*delta_amp
        N = N*2
        if N > N_max :
            N = N_max

print("Final bounds are -- ", lower_amp, upper_amp, l_simstart, l_simeq)