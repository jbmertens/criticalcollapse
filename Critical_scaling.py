#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
from importlib import reload 
import scipy.interpolate as interp
import scipy.constants as const

import cython
get_ipython().run_line_magic('load_ext', 'Cython')

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)


# importing MS and HM modules. Reload modules if needed
try :
    reload(ms_hm)
    reload(ms_hm.QCD_EOS)
    reload(ms_hm.MS)
    reload(ms_hm.HM)
    reload(ms_hm.timer)
except :
    print("Did not reload modules, they may not have been imported yet.")

import ms_hm
from ms_hm.QCD_EOS import *
from ms_hm.MS import *
from ms_hm.HM import *
from ms_hm.timer import *


# ## Various functions that use the MS and HM classes
# 
# Functions below compute various things using the Misner-Sharp or Hernandez-Misher classes.


get_ipython().run_cell_magic('cython', '', '\nimport cython\ncimport numpy as np\n\ncimport cython\nctypedef np.double_t DTYPE_t\n\nfrom libc.math cimport exp\nfrom libc.math cimport sqrt\n\nfrom cython.parallel import prange\n\n@cython.boundscheck(False)  # Deactivate bounds checking\ncpdef zero_crossing(np.ndarray x_in, double [::1] y):\n    """\n    Given an array of inputs y, find x where the function y(x) = 0\n    """\n    cdef int size = x_in.shape[0]\n    cdef double [:] x = x_in\n    cdef double w, zero=-1\n    cdef int i\n    \n    for i in range(size-1): # loop through all y values\n        if(y[i] * y[i+1] < 0): # if subsequent elements have opposite signs, a zero-crossing was found.\n            # linearly extrapolate zero-crossing\n            w = abs(y[i] / (y[i+1] - y[i])) \n            zero = x[i] * (1 - w) + x[i+1] * w\n            break\n            \n    return zero')


def mix_grid(left, right, n):
    """
    Function to generate coordinate spacings with a mix of uniform and logarithmic spacings,
    with uniform spacing at small values (0 to "left") then logarithmically spaced ("left" to "right").
    
    Returns an array of coordinate positions.
    """
    
    # Generate logarithmically spaced coordinates between "left" and "right"
    A = np.exp(np.linspace(left, right, n))
    dA = A[1] - A[0]
    
    # Generate uniformly spaced coordinates
    A = np.concatenate( (np.linspace(0, A[0], int(np.ceil(A[0] / dA)), endpoint=False), A))
    
    return A

def uni_grid(right, n):
    """
    Function to generate a uniformly-spaced array of coordinate positions
    
    Returns the array of uniformly spaced coordinates.
    """
    A = np.linspace(0, np.exp(right), n)
    return A

def exp_grid(left, right, n):
    """
    Function to generate an exponentially-spaced array of coordinate positions
    """
    # Generate logarithmically spaced coordinates between "left" and "right"
    A = np.logspace(left, right, n)
    A = np.concatenate( ([0], A))

    return A


# Check if a BH forms
# The MS run should proceed until MS until it breaks. If 2m / R > 1, return true
def BH_form(Abar, rho0, amp, default_steps=1500000, sm_sigma=0.0, fixw=False):
    
    ms = MS(Abar, rho0, amp, trace_ray=False, BH_threshold=-1,
        sm_sigma=sm_sigma, fixw=fixw)
    
    run_result = ms.adap_run_steps(default_steps)
    if run_result == -1 :
        return (True, run_result, ms.delta, ms)
    
    return (False, run_result, ms.delta, ms)


def find_crit(Abar, rho0, lower_amp, upper_amp,
    sm_sigma=0.0, fixw=False):
    """
    Binary search between lower and upper to find a critical amplitude
    (Note that this is NOT the critical density)
    return (critical, upper value)
    """
    upper_ms = -1
    lower_ms = -1
    bisection_factor = 1/2
    for i in range(20):
        try :
            amp_diff = upper_amp - lower_amp
            middle_amp = upper_amp - amp_diff*bisection_factor
            print('Iteration No', str(i), '-- Checking to see if a BH forms at amplitude', str(middle_amp),
                ' (bracket is ', lower_amp, '/', upper_amp, ')')

            forms, result, delta, ms = BH_form(Abar, rho0, middle_amp, sm_sigma=sm_sigma, fixw=fixw)

            if result > 0 : # Some sort of error during run
                print("Encountered error during run. Trying again with a different amplitude.")
                # change bisection factor (don't divide just in half; walk in by 1/4, 1/8, ... from the ends.)
                if bisection_factor >= 1/2 :
                    bisection_factor = (1-bisection_factor)/2
                else :
                    bisection_factor = 1-bisection_factor
            else :
                bisection_factor = 1/2
                if(forms == True):
                    upper_amp = middle_amp
                    upper_ms = ms
                else:
                    lower_amp = middle_amp
                    lower_ms = ms

        except Exception as e :
            print("Run failed! Stopping search. Reason below.")
            print(e)
            break
    
    print("Critical amplitude appears to be between", lower_amp, "and", upper_amp)
    try :
        upper_ms.plot_fields(True)
        lower_ms.plot_fields(True)
    except :
        pass

    return (lower_amp, upper_amp)


def find_mass(Abar, rho0, amp, mOverR_thresh=0.98,
        is_searching_for_crit=False, ms_steps=1500000,
        MS_sm_sigma=0.0,
        hm_steps=1500000, HM_sm_sigma=15.0,
        HM_Abar=None, # can specify a different Abar for HM run
        HM_cflfac=0.1
    ):
    """
    Find mass of BHs for certain amplitude
    set is_searching_for_crit=True when searching for the critical point
    """
    print('Finding mass with amp ' + str(amp))
            
    # Perform an MS run without raytracing to get the overdensity delta
    ms = MS(Abar, rho0, amp, trace_ray=False, BH_threshold=-1e1, sm_sigma=MS_sm_sigma)
    ms.adap_run_steps(ms_steps)
    delta = ms.delta
    
    # Perform an MS run with raytracing to get ICs for an HM run
    ms = MS(Abar, rho0, amp, trace_ray=True, BH_threshold=-1e1, sm_sigma=MS_sm_sigma)
    flag = ms.adap_run_steps(ms_steps)
    if(flag != 0):
        raise ValueError('Not finishing ray-tracing with the amplitude ' + str(amp))
        
    # Perform an HM run
    hm = HM(ms, mOverR=mOverR_thresh, sm_sigma=HM_sm_sigma, Abar=HM_Abar, cflfac=HM_cflfac)
    bh_formed = hm.run_steps(hm_steps) == 1
    if(not bh_formed and is_searching_for_crit==False):
        print('Unable to reach the target 2m/R with the amplitude ' + str(amp))
    
    return (delta, hm.BH_mass2(), ms, hm)



# simulation resolution parameter
# (Not exactly the number of gridpoints for a mixed grid)
n_mix = 3200
n_uni = 3200

# Generate an array of coordinate positions for the simulation to run at
lower = np.log(5.65) # The coordinates will be linearly spaced from 0 to e^lower
upper = np.log(21.0) # The coordinates will be log spaced from e^lower to e^upper
Abar_mix = mix_grid(lower, upper, n_mix)
Abar_uni = uni_grid(upper, n_uni)

plt.plot(Abar_mix, 'k.')
plt.plot(Abar_uni, 'b.')
print("The grid of Abar values is linearly spaced from Abar = 0 to", np.exp(lower),
      "then log spaced until Abar =", np.exp(upper))






# rho0 = float(sys.argv[1]) # initial density value in MeV^4
# fixw = int(sys.argv[2]) # 0 for not fixed, 1 for fixed initially, 2 for fixed at turnaround
# if(fixw == 0) :
#     fixw = False
#     use_turnaround = False
# elif(fixw == 1) :
#     fixw = True
#     use_turnaround = False
# elif(fixw == 2) :
#     fixw = True
#     use_turnaround = True
# else :
#     print("Error running simulation, bad fixw specified.")

# print("Running simulation with rho0 =", rho0, ", fixw =", fixw, ", and use_turnaround = ", use_turnaround)
# find_crit(Abar, rho0, 0.15, 0.3, fixw=fixw,sm_sigma=2.0)



# find_crit(Abar_mix, 1.0e0, 0.15, 0.3, fixw=False, sm_sigma=0.0)
delta, mass, ms, hm = find_mass(Abar_mix, 1.0e0, 0.2727, mOverR_thresh=0.994,
          MS_sm_sigma=0.0, hm_steps=79000, HM_sm_sigma=30.0, HM_Abar=Abar_mix, HM_cflfac=0.1)


delta, mass, ms, hm = find_mass(Abar_mix, 1.0e0, 0.2727, mOverR_thresh=0.994,
          MS_sm_sigma=0.0, hm_steps=150000, HM_sm_sigma=50.0, HM_Abar=Abar_mix, HM_cflfac=0.1)


# ms = MS(Abar_mix, 1.0e0, 0.28, trace_ray=True, BH_threshold=-1e1, sm_sigma=0.0)
# ms.adap_run_steps(500)
# ms.timer.results()

# hm = HM(ms, mOverR=0.99, sm_sigma=15.0)
# hm.adap_run_steps(100)
# hm.timer.results()
