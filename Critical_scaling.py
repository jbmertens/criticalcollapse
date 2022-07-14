#!/usr/bin/env python
# coding: utf-8

# # Critical Collapse Study
# 
# Below is code to help study the formation of black holes. The equations integrated are described in the work by Bloomfield et. al.
# 
# arXiv paper on : [1504.02071](https://arxiv.org/pdf/1504.02071.pdf) ([ar5iv](https://ar5iv.org/abs/1504.02071))
# 
# For collapse with an approximate QCD equation of state, a previous work is
# 
# arXiv: [1801.06138](https://arxiv.org/pdf/1801.06138.pdf) ([ar5iv](https://ar5iv.org/abs/1801.06138))
# 
# ## Import Modules


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
    reload(ms_hm.MS)
    reload(ms_hm.HM)
    reload(ms_hm.QCD_EOS)
except :
    pass

import ms_hm
from ms_hm.QCD_EOS import *
from ms_hm.MS import *
from ms_hm.HM import *


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
        return (True, ms.delta, ms)
    
    return (False, ms.delta, ms)


def find_crit(Abar, rho0, lower_amp, upper_amp,
    sm_sigma=0.0, fixw=False):
    """
    Binary search between lower and upper to find a critical amplitude
    (Note that this is NOT the critical density)
    return (critical, upper value)
    """
    upper_ms = -1
    lower_ms = -1
    for i in range(20):
        middle_amp = (lower_amp + upper_amp) / 2
        print('Iteration No', str(i), '-- Checking to see if a BH forms at amplitude', str(middle_amp))

        try :
            forms, delta, ms = BH_form(Abar, rho0, middle_amp, sm_sigma=sm_sigma, fixw=fixw)
        except Exception as e :
            print("Run failed with amplitude", middle_amp, "! Stopping search. Reason below.")
            print(e)
            break
        
        if(forms == True):
            upper_amp = middle_amp
            upper_ms = ms
        else:
            lower_amp = middle_amp
            lower_ms = ms
    
    print("Critical amplitude appears to be between", lower_amp, "and", upper_amp)
    try :
        upper_ms.plot_fields(True)
        lower_ms.plot_fields(True)
    except :
        pass

    return (lower_amp, upper_amp)


def find_mass(Abar, rho0, amp, is_searching_for_crit=False, default_steps=1500000):
    """
    Find mass of BHs for certain amplitude
    set is_searching_for_crit=True when searching for the critical point
    """
    print('Finding mass with amp ' + str(amp))
            
    # Perform a MS run without raytracing
    ms = MS(Abar, rho0, amp, trace_ray=False, BH_threshold=-1e1)
    ms.run_steps(default_steps)
    delta = ms.delta
    
    # Perform a run *with* raytracing to get ICs for an HM run
    ms = MS(Abar, rho0, amp, trace_ray=True, BH_threshold=-1e1)
    flag = ms.run_steps(default_steps)
    if(flag != 0):
        raise ValueError('Not finishing ray-tracing with the amplitude ' + str(amp))
        
    # Perform an HM run
    hm = HM(ms, mOverR=0.99, sm_sigma=50)
    bh_formed = hm.adap_run_steps(550000) == 1
    if(not bh_formed and is_searching_for_crit==False):
        raise ValueError('Cannot get the target 2m/R with the amplitude ' + str(amp))
    
    print(ms.delta, hm.BH_mass2())
    return (ms.delta, hm.BH_mass2())


n = 400
Abar = mix_grid(np.log10(1.5), np.log10(40), n)
rho0 = float(sys.argv[1]) # initial density value in MeV^4
find_crit(Abar, rho0, 0.15, 0.3, fixw=False)

