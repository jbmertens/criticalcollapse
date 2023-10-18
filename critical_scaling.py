#!/usr/bin/env python
# coding: utf-8

import sys, os, glob, re
import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.optimize as opt

def fitfn(x, p, x0, A) :
  return p*np.log( np.abs(x - x0) ) + A

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
    amp=0.7162501041727418,
    steps=2000000,
    N=65536,
    Ld=21.0,
    USE_FIXW=False,
    q_mult=0.08,
    TOL=4e-9,
    horizon_stop=False
) :
    deltaH = c_real_t(-2)
    max_rho0 = c_real_t(0)
    bh_mass = c_real_t(0)
    agg = (c_real_t*(N*13))()
    l = c_real_t(l_simstart)

    c_lib.ics(agg, ctypes.byref(l), ctypes.byref(deltaH), ctypes.byref(max_rho0), ctypes.byref(bh_mass),
              amp*c_lib.G(l_simstart)/c_lib.G(l_simeq), np.exp(l_simeq), N, Ld, USE_FIXW)

    result = c_lib.run_sim(agg, ctypes.byref(l), ctypes.byref(deltaH), ctypes.byref(max_rho0), ctypes.byref(bh_mass),
                        steps, -1, horizon_stop, q_mult, True, True, -400, 1.0, 0.001, TOL)
    print(result, l, c_lib.G(l_simstart)/c_lib.G(l_simeq), deltaH, max_rho0, bh_mass)

    fields = np.reshape(np.copy(agg), (13, N))

    return ( result, l, c_lib.G(l_simstart)/c_lib.G(l_simeq), deltaH, max_rho0, bh_mass )


l_simstart_str = sys.argv[1]
l_simstart = float(sys.argv[1])
l_simeq_str = sys.argv[2]
l_simeq = float(sys.argv[2])
fixw = False
files = glob.glob('output/run_'+l_simstart_str+'_'+l_simeq_str+'.txt')
if sys.argv[3] == "fixw" :
    fixw = True
    files = glob.glob('output/fixw_'+l_simstart_str+'_'+l_simeq_str+'.txt')
print(files)

if len(files) != 1 :
    print("Error determining files.")
else :
    file = files[0]
    with open(file, 'r') as f:
        lines = f.readlines()

        # Get amplitude bounds
        lower_amp = 0.0
        upper_amp = 0.0
        for row in lines:
            if row.find('Final bounds are') != -1 :
                print("Matched line:", row)
                matches = re.findall(r"[0-9\.]+", row)
                lower_amp = float(matches[0])
                upper_amp = float(matches[1])
                break

        damp = upper_amp - lower_amp
        damps = np.logspace( np.log10(damp/10), np.log10(0.02), 30 )
        amps = np.concatenate(( lower_amp + damps, upper_amp - damp/20 - damps))
        print("Bounds identified were (", lower_amp, ",", upper_amp, "). Running with amplitudes", amps)

        runs = []
        for amp in np.flip(amps) :
            print("Getting mass with l_simstart =", l_simstart, "l_simeq =", l_simeq, "amp =", amp)
            runs.append(find_mass( l_simstart=l_simstart, l_simeq=l_simeq, amp=amp, USE_FIXW=fixw, horizon_stop=True ))

        print("Final data:", np.array(runs))

        gmask = runs[:,0]==3
        pars = opt.curve_fit( fitfn, runs[gmask,3], np.log(runs[gmask,5]), [0.35, 0.43, 2.0], bounds=( 0.0, [1.0, 1.0, 30.0] ) )[0]
        print("Fit parameters:", pars)
        # plt.scatter( np.log(fwd[gmask,3]-pars[1]), np.log(fwd[gmask,5]) );
        # plt.plot( np.log(fwd[gmask,3]-pars[1]), fitfn(fwd[gmask,3], pars[0], pars[1], pars[2]) );
