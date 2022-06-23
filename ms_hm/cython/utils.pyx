import numpy as np
import cython
cimport numpy as np

cimport cython
ctypedef np.double_t DTYPE_t

from libc.math cimport exp
from libc.math cimport sqrt

from cython.parallel import prange

# first derivative
@cython.boundscheck(False)  # Deactivate bounds checking
cpdef dfdA(np.ndarray arr, double [::1] A, double bd = 0, int exec_pos = -1):
    cdef int size = arr.shape[0]
    cdef double [:] f = arr
    cdef double [:] res =  np.zeros(size, dtype=np.double)

    cdef int i
    for i in prange(exec_pos+2, size - 1,nogil=True):
        res[i] = (f[i+1] - f[i-1]) / (A[i+1] - A[i-1])
    if(exec_pos > -1):
        res[exec_pos+1] = (f[exec_pos+2] - f[exec_pos+1]) / (A[exec_pos+2]-A[exec_pos+1])
    elif(bd < 1e99): # if bd value is valid, set 0th component to bd
        res[0] = bd
    else: # invalid bd value, so use one-side derivative
        res[0] = (f[1] - f[0]) / (A[1] - A[0])
        #res[0] = (-11 * f[0] / 6 + 3 * f[1] - 3 * f[2] / 2 + f[3] / 3) / (A[1] - A[0])
    res[size - 1] = (f[-1] - f[-2]) / (A[size - 1] - A[size-2])
    return np.asarray(res)

# first derivative
@cython.boundscheck(False)  # Deactivate bounds checking
cpdef stg_dfdA(np.ndarray arr, double [::1] A):
    cdef int size = arr.shape[0]
    cdef double [:] f = arr
    cdef double [:] res =  np.zeros(size-1, dtype=np.double)

    cdef int i
    for i in prange(0, size - 1,nogil=True):
        res[i] = (f[i+1] - f[i]) / (A[i+1] - A[i])

    return np.asarray(res)

# first derivative
@cython.boundscheck(False)  # Deactivate bounds checking
cpdef WENO_dfdA(np.ndarray arr,  double [::1] A, double bd=0):
    cdef int size = arr.shape[0]
    cdef double [:] f = arr
    cdef double [:] st_f =  np.zeros(size, dtype=np.double) # i+1/2 field
    cdef double [:] res =  np.zeros(size, dtype=np.double)

    cdef int i
    cdef double b1, b2, b3, w1, w2, w3, ws, st_f1, st_f2, st_f3
    for i in prange(2, size - 2, nogil=True):
        b1 = (4 * f[i-2]**2 - 19 * f[i-2] * f[i-1] + 25 * f[i-1]**2 + 11 * f[i-2] * f[i] - 31 * f[i-1] * f[i] + 10 * f[i]**2) / 3
        b2 = (4 * f[i-1]**2 - 13 * f[i-1] * f[i] + 13 * f[i]**2 + 5 * f[i-1] * f[i+1] - 13 * f[i] * f[i+1] + 4 * f[i+1]**2) / 3
        b3 = (10 * f[i]**2 - 31 * f[i] * f[i+1] + 25 * f[i+1]**2 + 11 *f[i] * f[i+2] - 19 * f[i+1] * f[i+2] + 4 * f[i+2]**2) / 3
        w1 = 1/10/(1e-6 + b1)**2
        w2 = 6/10/(1e-6 + b2)**2
        w3 = 3/10/(1e-6 + b3)**2
        ws = w1 + w2 + w3
        w1 = w1 / ws
        w2 = w2 / ws
        w3 = w3 / ws
        st_f1 = 3 * f[i-2] / 8 - 5 * f[i-1] / 4 + 15 * f[i] / 8
        st_f2 = -f[i-1] / 8 + 3 * f[i] / 4 + 3 * f[i+1] / 8
        st_f3 = 3 * f[i] / 8 + 3 * f[i+1] /4 - f[i+2] / 8
        st_f[i] = st_f1 * w1 + st_f2 * w2 + st_f3 * w3


    for i in prange(3, size - 2, nogil=True):
        res[i] = (st_f[i] - st_f[i-1]) / ((A[i+1] - A[i-1]) / 2)

    res[2] = (f[3] - f[1]) / (A[3] - A[1])
    res[1] = (f[2] - f[0]) / (A[2] - A[0])
    res[0] = (f[1] - f[0]) / (A[1] - A[0])

    res[size - 1] = (f[size-1] - f[size-2]) / (A[size-1] - A[size-2])
    res[size-2] = (f[size-1] - f[size-3]) / (A[size-1] - A[size-3])

    return np.asarray(res)


@cython.boundscheck(False)  # Deactivate bounds checking
cpdef WENO_to_stg(np.ndarray arr):
    cdef int size = arr.shape[0]
    cdef double [:] f = arr
    cdef double [:] st_f =  np.zeros(size-1, dtype=np.double) # i+1/2 field

    cdef int i
    cdef double b1, b2, b3, w1, w2, w3, ws, st_f1, st_f2, st_f3
    for i in prange(2, size - 2,nogil=True):
        b1 = (4 * f[i-2]**2 - 19 * f[i-2] * f[i-1] + 25 * f[i-1]**2 + 11 * f[i-2] * f[i] - 31 * f[i-1] * f[i] + 10 * f[i]**2) / 3
        b2 = (4 * f[i-1]**2 - 13 * f[i-1] * f[i] + 13 * f[i]**2 + 5 * f[i-1] * f[i+1] - 13 * f[i] * f[i+1] + 4 * f[i+1]**2) / 3
        b3 = (10 * f[i]**2 - 31 * f[i] * f[i+1] + 25 * f[i+1]**2 + 11 *f[i] * f[i+2] - 19 * f[i+1] * f[i+2] + 4 * f[i+2]**2) / 3
        w1 = 1/10/(1e-6 + b1)**2
        w2 = 6/10/(1e-6 + b2)**2
        w3 = 3/10/(1e-6 + b3)**2
        ws = w1 + w2 + w3
        w1 = w1 / ws
        w2 = w2 / ws
        w3 = w3 / ws
        st_f1 = 3 * f[i-2] / 8 - 5 * f[i-1] / 4 + 15 * f[i] / 8
        st_f2 = -f[i-1] / 8 + 3 * f[i] / 4 + 3 * f[i+1] / 8
        st_f3 = 3 * f[i] / 8 + 3 * f[i+1] /4 - f[i+2] / 8
        st_f[i] = st_f1 * w1 + st_f2 * w2 + st_f3 * w3

    st_f[1] = (f[2] + f[1]) / 2
    st_f[0] = (f[1] + f[0]) / 2

    st_f[size-2] = (f[size-1] + f[size-2]) / 2

    return np.asarray(st_f)

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef inv_derv_phi(np.ndarray rho_in, double [:] P, double off_set=0, int exec_pos = -1):
    cdef int size = rho_in.shape[0]
    cdef double [:] rho = rho_in
    cdef double [:] res =  np.zeros(size, dtype=np.double)
    cdef int i
    res[size - 1] = off_set
    res[size - 2] =  (P[size-1] - P[size-2]) / (P[size-2] + rho[size-2]) + res[size-1]
    for i in range(size-3, -1, -1):
        res[i] = (P[i+2] - P[i]) / (P[i+1] + rho[i+1]) + res[i+2]
    return np.asarray(res)

@cython.boundscheck(False)  # Deactivate bounds checking
# for exp(\psi) in Eq. 165e
cpdef derv_psi( np.ndarray xi, np.ndarray c, double off_set=0):
    cdef int size = xi.shape[0]
    cdef double [:] res =  np.zeros(size, dtype=np.double)
    cdef int i
    res[0] = off_set
    res[1] = - (xi[2] - xi[1]) * c[0] + res[0]

    for i in range(2, size):
        res[i] = - (xi[i] - xi[i-2]) * c[i - 1] + res[i-2]
    return np.asarray(res)

@cython.boundscheck(False)  # Deactivate bounds checking
# for exp(\psi) in Eq. 165e
cpdef inv_derv_psi( np.ndarray xi_in, double [::1] c, double off_set=0):
    cdef int size = xi_in.shape[0]
    cdef double [:] xi = xi_in
    cdef double [:] res =  np.zeros(size, dtype=np.double)
    cdef int i
    res[size - 1] = off_set
    res[size - 2] =  (xi[size-1] - xi[size-2]) * c[size-2] + res[size-1]
    #for i in range(size-3, -1, -1):
    #    res[i] = (xi[i+1] - xi[i]) * c[i] + res[i+1]
    for i in range(size-3, -1, -1):
        res[i] = (xi[i+2] - xi[i]) * c[i+1] + res[i+2]
    return np.asarray(res)

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef find_exec_pos(np.ndarray arr):
    cdef int size = arr.shape[0]
    cdef double [:] f = arr

    cdef int i
    for i in range(size - 1, 1,-1):
        if(arr[i] < 1 and arr[i-1] > 1): # found a horizon
            return i-1
    return -1

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef ms_rho_term(np.ndarray R_in, double [:] m, double [:] A):
    cdef int size = R_in.shape[0]
    cdef double [:] res =  np.zeros(size, dtype=np.double)
    cdef double [:] R = R_in
    cdef int i
    # expression on interior of grid
    for i in prange(1, size - 1,nogil=True):
        res[i] = A[i] * R[i] * (m[i+1] - m[i-1]) \
                / (3 * (A[i+1] * R[i+1] - A[i-1] * R[i-1]))
    # expression at origin
    i = 0
    res[i] = 0
    
    # expression at outer boundary
    i = size - 1
    res[i] = A[i] * R[i] * (m[i] - m[i-1]) \
                / (3 * (A[i] * R[i] - A[i-1] * R[i-1]))
    return np.asarray(res)

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef ms_rho_term_stg(np.ndarray R_in, double [:] m, double [:] A,
                     double [:] R_stg, double [:] A_stg):
    cdef int size = R_in.shape[0]
    cdef double [:] res =  np.zeros(size-1, dtype=np.double)
    cdef double [:] R = R_in
    cdef int i
    for i in prange(0, size - 1,nogil=True):
        res[i] = A_stg[i] * R_stg[i] * (m[i+1] - m[i]) \
                / (3 * (A[i+1] * R[i+1] - A[i] * R[i]))
    return np.asarray(res)

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef hm_rho_term(np.ndarray R_in, double [:] m, double [:] A, double [:] xi,
               double a):
    cdef int size = R_in.shape[0]
    cdef double [:] res =  np.zeros(size, dtype=np.double)
    cdef double [:] R = R_in
    cdef int i
    for i in prange(1, size - 1,nogil=True):
        res[i] = (m[i+1] - m[i-1] - 2 * m[i] * (xi[i+1] - xi[i-1])) \
                / (a * A[i] * R[i] * (xi[i+1] - xi[i-1]) + (A[i+1] * R[i+1] - A[i-1] * R[i-1]))

    i = 0
    res[0] = (m[i+1] - m[i] - 2 * m[i] * (xi[i+1] - xi[i])) \
                / (a * A[i] * R[i] * (xi[i+1] - xi[i]) + (A[i+1] * R[i+1] - A[i] * R[i]))
    i = size - 1
    res[size-1] = (m[size-1] - m[i-1] - 2 * m[i] * (xi[size-1] - xi[i-1])) \
                / (a * A[i] * R[i] * (xi[size-1] - xi[i-1]) + (A[size-1] * R[size-1] - A[i-1] * R[i-1]))
    return np.asarray(res)

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef rho_term_stg(np.ndarray R_in, double [:] m, double [:] A, double [:] xi,
               double [:] R_stg, double [:] m_stg, double [:] A_stg, double a):
    cdef int size = R_in.shape[0]
    cdef double [:] res =  np.zeros(size-1, dtype=np.double)
    cdef double [:] R = R_in
    cdef int i
    for i in prange(0, size - 1,nogil=True):
        res[i] = (m[i+1] - m[i] - 2 * m_stg[i] * (xi[i+1] - xi[i])) \
                / (a * A_stg[i] * R_stg[i] * (xi[i+1] - xi[i]) + (A[i+1] * R[i+1] - A[i] * R[i]))

    return np.asarray(res)

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef ephi_term(np.ndarray R_in, double [:] U, double [:] A, double [:] xi, double [:] g,
               double a):
    cdef int size = R_in.shape[0]
    cdef double [:] res =  np.zeros(size, dtype=np.double)
    cdef double [:] R = R_in
    cdef int i
    for i in prange(1, size - 1,nogil=True):
        res[i] = (a * (xi[i+1] - xi[i-1]) * A[i] * R[i] + (A[i+1] * R[i+1] - A[i-1] * R[i-1])) /\
                (a * (xi[i+1] - xi[i-1]) * (g[i] + A[i] * R[i] * U[i]))

    i = 0
    res[0] = (a * (xi[i+1] - xi[i]) * A[i] * R[i] + (A[i+1] * R[i+1] - A[i] * R[i])) /\
                (a * (xi[i+1] - xi[i]) * (g[i] + A[i] * R[i] * U[i]))
    i = size - 1
    res[size-1] = (a * (xi[size-1] - xi[i-1]) * A[i] * R[i] + (A[size-1] * R[size-1] - A[i-1] * R[i-1])) /\
                (a * (xi[size-1] - xi[i-1]) * (g[i] + A[i] * R[i] * U[i]))
    return np.asarray(res)

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef rho_prime(np.ndarray R_in, double [:] m, double [:] A, double [:] xi,
               double [:] Rp, double [:] mp, double [:] xip,
               double [:] Rpp, double [:] mpp, double [:] xipp, double a):
    cdef int size = R_in.shape[0]
    cdef double [:] res =  np.zeros(size, dtype=np.double)
    cdef double [:] R = R_in
    cdef int i
    for i in prange(0, size-1 ,nogil=True):
        res[i] = ((R[i] + A[i] * Rp[i] + A[i] * a * R[i] * xip[i]  )
                  * (-2 * mp[i] * xip[i] + mpp[i] - 2 * m[i] * xipp[i])
                 - (mp[i] - 2 * m[i] * xip[i] ) * (Rp[i] * (2 + A[i] * a * xip[i])
                        + A[i] * Rpp[i] + a * R[i] * (xip[i] + A[i] * xipp[i])) ) \
                / (R[i] + A[i] * Rp[i] + A[i] * a * R[i] * xip[i])**2
    return np.asarray(res)


@cython.boundscheck(False)  # Deactivate bounds checking
cpdef zero_crossing(np.ndarray x_in, double [::1] y):
    cdef int size = x_in.shape[0]
    cdef double [:] x = x_in
    cdef double w, res=-1
    cdef int i
    for i in range(size-1):
        if(y[i] * y[i+1] < 0):
            w = abs(y[i] / (y[i+1] - y[i]))
            res = x[i] * (1 - w) + x[i+1] * w
            break
    return res
