import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy import stats

from ms_hm.utils import *
from ms_hm.timer import *

from ipywidgets import HTML
from IPython.display import display
import time

class HM:
    """
    Class to integrate the Hernandez-Misner equations to study
    critical collapse of a fluid to a black hole.

    Equations integrated are 166a - 166d in
    https://arxiv.org/pdf/1504.02071.pdf .
    These have been slightly modified to allow for a general
    (local, temperature-dependent) pressure term.
    """

    def __init__(self, MS, mOverR=0.999, sm_sigma=5,
        Abar=None, cflfac=0.2):

        self.timer = timer()
        
        # Try to work in case raytracing didn't finish
        R = MS.R_hm
        self.N = np.max(np.where(R>0))

        self.R = MS.R_hm[:self.N]
        self.m = MS.m_hm[:self.N]
        self.U = MS.U_hm[:self.N]
        self.xi = MS.xi_hm[:self.N]
        self.rho_p = MS.rho_hm[:self.N]
        self.Abar = MS.Abar[:self.N]

        if Abar is not None :
            print("Computing field values at specified Abar")
            Abar_max = np.searchsorted(Abar, self.Abar[-1], "left")

            self.R = np.interp(Abar[:Abar_max], self.Abar, self.R)
            self.m = np.interp(Abar[:Abar_max], self.Abar, self.m)
            self.U = np.interp(Abar[:Abar_max], self.Abar, self.U)
            self.xi = np.interp(Abar[:Abar_max], self.Abar, self.xi)
            self.rho_p = np.interp(Abar[:Abar_max], self.Abar, self.rho_p)
            self.Abar = Abar[:Abar_max]

        self.Abar_stg = self.to_stg(self.Abar)
        
        self.sm_sigma = sm_sigma

        self.w0 = MS.w0
        self.alpha = MS.alpha

        self.qcd = MS.qcd

        self.t0 = MS.t0
        self.t = self.t0
        self.u = 0
        self.RH = MS.RH

        self.qcd = MS.qcd
        self.w0 = MS.w0
        self.alpha = MS.alpha

        # self.kappa = 2
        self.Q = np.zeros(self.N)
        self.Q_du = np.zeros(self.N)
        self.Q_old = np.zeros(self.N)
        self.Qprime = np.zeros(self.N)

        self.tmp = 0
        
        self.deltau_i = self.cfl_deltau(self.R, self.m, self.U, self.xi) * cflfac
        self.deltau_adap = self.deltau_i

        self.mOverR = mOverR

        self.step = 0

        self.start_time = time.process_time()
        self.display = HTML(value="Running HM simulation.")
        display(self.display)

        self.mass_data = np.array([])
        self.max2moR_data = np.array([])

        return

    # convert to half grid
    def to_stg(self,arr):
        return (arr[0:-1] + arr[1:]) / 2

    def to_cubic_stg(self,arr):
        a1 = arr[0:-1] ** 3
        a2 = arr[1:] ** 3
        return (a1 + a2) / (np.abs(a1 + a2)) * ( np.abs(a1 + a2)/ 2) **(1/3)

    def gamma(self, R, m, U, xi):
        return np.sqrt(np.exp(2 * (1 - self.alpha) * xi)
                       + (self.Abar * R)**2 * (U**2 - m))
    
    def P(self, rho) :
        """
        Compute (tilded) pressure as a function of (tilded) density.
        """
        self.timer.start("P")
        H = np.exp(-self.xi) / self.RH
        rhob = 3 / (8*np.pi) * H**2
        realRho = rho * rhob
        realP = self.qcd.P(realRho)
        P = realP/rhob
        self.timer.stop("P")
        return P
    
    # return the L2 error in rho
    def rho_err(self, R, m, U, xi, g, xiprime, Rprime, mprime, P, rho_p):
        temp = rho_p * g - P * self.Abar * R * U  - (g + self.Abar * R * U)  \
            * (m + self.Abar * R * hm_rho_term(R, m, self.Abar, xi, self.alpha) / 3)
        # temp = rho_p - (g + self.Abar * R * U) / (g - P/rho_p * self.Abar * R * U ) \
        #    * (m + self.Abar * R * hm_rho_term(R, m, self.Abar, xi, self.alpha) / 3)
        return temp
    
    def rho_err_prime(self, R, m, U, xi, g, xiprime, Rprime, mprime, dPdrho, rho_p):
        temp =  g - dPdrho * self.Abar * R * U \
        #- (g + self.Abar * R * U)  \
        #   * (m + self.Abar * R * hm_rho_term(R, m, self.Abar, xi, self.alpha) / 3)
        #print((m + self.Abar * R * hm_rho_term(R, m, self.Abar, xi, self.alpha) / 3))
        #print(np.abs(temp).min())
        
        #print( [(rho_p * g)[0] , (dPdrho * self.Abar * R * U)[0], (g + self.Abar * R * U)[0], (m + self.Abar * R * hm_rho_term(R, m, self.Abar, xi, self.alpha) / 3)[0]  ] )
        self.tmp = temp
        return temp
    
    def rho(self, R, m, U, xi, g, xiprime, Rprime, mprime):
        self.timer.start("rho")

        H = np.exp(-self.xi) / self.RH
        rhob = 3 / (8*np.pi) * H**2
        
        err = np.ones_like(R)

        while( np.linalg.norm(err) > 1e-5 ):
            # print(np.linalg.norm(err))

            # Iterative method (slower)
            # P = self.qcd.P(self.rho_p * rhob) / rhob
            # rho_std = (g + self.Abar * R * U) / (g - P/self.rho_p * self.Abar * R * U ) \
            #    * (m + self.Abar * R * hm_rho_term(R, m, self.Abar, xi, self.alpha) / 3)
            # err = rho_std - self.rho_p
            # self.rho_p = rho_std

            # Newton's method
            dPdrho = self.qcd.dPdrho(self.rho_p * rhob)
            P = self.qcd.P(self.rho_p * rhob) / rhob
            err = self.rho_err(R, m, U, xi, g, xiprime, Rprime, mprime, P, self.rho_p)
            err_prime = self.rho_err_prime(R, m, U, xi, g, xiprime, Rprime, mprime, dPdrho, self.rho_p)
            self.rho_p = self.rho_p - err / err_prime
        
        self.rho_p = gaussian_filter1d(self.rho_p, sigma=self.sm_sigma, mode='nearest')

        if(len(self.rho_p[self.rho_p<=0]) > 0) :
            self.rho_p[self.rho_p<=0] = 1.0e-10
            # plt.plot(self.rho_p)
            # raise ValueError('Rho is negative.')

        self.timer.stop("rho")
        return self.rho_p

    def rho_stg(self, R, m, U, xi, g, xiprime, Rprime, mprime):
        R_stg = self.to_stg(R)
        m_stg = self.to_stg(m)
        U_stg = self.to_stg(U)
        g_stg = self.to_stg(g)
        #R_stg = WENO_to_stg(R)
        #m_stg = WENO_to_stg(m)
        #U_stg = WENO_to_stg(U)
        #g_stg = WENO_to_stg(g)
        A_stg = self.Abar_stg

        H = np.exp(-self.xi) / self.RH
        rhob = 3 / (8*np.pi) * H**2
        P = self.qcd.P(rho_p * rhob) / rhob
        
        err = rho_err(R, m, U, xi, g, xiprime, Rprime, mprime, P, rho_p)
        
        temp = (g_stg + A_stg * R_stg * U_stg) / (g_stg - (self.w + Q_stg ) * A_stg * R_stg * U_stg ) \
            * (m_stg + A_stg * R_stg * rho_term_stg(R, m, self.Abar, xi, R_stg, m_stg, A_stg, self.alpha) / 3)
        #temp = scipy.signal.savgol_filter(temp, 31, 3, mode='interp')
        return temp

    def ephi(self, R, U, g, xi, xiprime, Rprime):
        return ephi_term(R, U, self.Abar, xi, g, self.alpha)

    def elambda(self, ephi, exi, xiprime):
        return self.alpha * ephi * exi * xiprime

    def epsi(self, R, U, g, xi, rho, ephi, Q):
        c = self.alpha - 1 + ephi * self.Abar * R * rho * (1 + Q) /\
            ((g + self.Abar * R * U) * (1+self.w0))
        offset = np.log(1/ephi[-1] * (g[-1] + self.Abar[-1] * R[-1] * U[-1]))
        temp = inv_derv_psi(xi, c, offset)
        return  (g + self.Abar * R * U) / np.exp(temp)

    def drho(self, R, m, U, g, xi, Rp, mp, xip):
        rho_stg = self.rho_stg(R, m, U, xi, g, xip, Rp, mp)
        return np.concatenate( ([0], stg_dfdA(rho_stg, self.to_stg(self.Abar)) ,[0]) )

    def set_Q_old(self,  R, m, U, xi):
        xiprime = WENO_dfdA(xi, self.Abar, 1e100)
        Rprime = WENO_dfdA(R, self.Abar, 1e100)
        mprime = WENO_dfdA(m, self.Abar, 1e100)
        Uprime = WENO_dfdA(U, self.Abar, 1e100)

        #xiprime = WENO_nuni_dfdA(xi, self.Vbar, self.Abar, 1e100)
        #Rprime = WENO_nuni_dfdA(R, self.Vbar, self.Abar, 1e100)
        #mprime = WENO_nuni_dfdA(m, self.Vbar, self.Abar, 1e100)
        #Uprime = WENO_nuni_dfdA(U, self.Vbar, self.Abar, 1e100)

        g = self.gamma(R, m, U, xi)
        r = self.rho(R, m, U, xi, g, xiprime, Rprime, mprime)
        p = self.P(r)
        self.Q_old = p / r
        
    def k_coeffs(self, R, m, U, xi) :
        self.timer.start("k_coeffs")

        xiprime = WENO_dfdA(xi, self.Abar, 1e100)
        Rprime = WENO_dfdA(R, self.Abar, 1e100)
        mprime = WENO_dfdA(m, self.Abar, 1e100)
        Uprime = WENO_dfdA(U, self.Abar, 1e100)

        #xiprime = WENO_nuni_dfdA(xi, self.Vbar, self.Abar, 1e100)
        #Rprime = WENO_nuni_dfdA(R, self.Vbar, self.Abar, 1e100)
        #mprime = WENO_nuni_dfdA(m, self.Vbar, self.Abar, 1e100)
        #Uprime = WENO_nuni_dfdA(U, self.Vbar, self.Abar, 1e100)

        g = self.gamma(R, m, U, xi)
        r = self.rho(R, m, U, xi, g, xiprime, Rprime, mprime)
        p = self.P(r)
        Q = p / r
        Q_du = (Q - self.Q_old) / self.deltau
        Qprime = dfdA(Q, self.Abar, 1e100)
        exi = np.exp(xi)
        ephi = self.ephi(R, U, g, xi, xiprime, Rprime)
        elambda = self.elambda(ephi, exi, xiprime)
        epsi = self.epsi(R, U, g, xi, r, ephi, p / r)

        # drho = self.drho(R, m, U, g, xi, Rprime, mprime, xiprime)
        drho  = dfdA(r, self.Abar, 1e100)
        drho[0] = 3 * elambda[0] / exi[0] * ((1 + self.w0) * r[0] / ephi[0] - (r[0] + p[0]) * U[0])
        drho[-1] = 0

        kxi = epsi / ephi / np.exp(xi) / self.alpha

        kR = epsi / exi * R * (U - 1/ephi)

        km = 3 * epsi / exi * (1/ephi * m * (1+self.w0) - U * (p +m))

        kU = - epsi / exi / (1 - Q) * (
            (m + 3 * p) / 2 + U**2 - U / self.alpha / ephi
            + (Q) * exi / elambda * Uprime + g * (Q) / ( np.concatenate( ([1], self.Abar[1:]) ) \
                                                                                * R * (1 + Q)) * (
                3 * (1 + Q) * U + exi * (Qprime / elambda - Q_du / epsi) / (Q)
            - 3 * (1 + self.w0) * (1 / ephi) + exi / elambda * drho / r)
        )

        # boundary conditions
        kxi[0] = epsi[0] / elambda[0] * (xi[1] - xi[0]) / ( (self.Abar[1] - self.Abar[0]) )
        kR[0] = epsi[0] / elambda[0] * (R[1] - R[0]) / ( (self.Abar[1] - self.Abar[0]) )
        km[0] = epsi[0] / elambda[0] * (m[1] - m[0]) / ( (self.Abar[1] - self.Abar[0]) )
        kU[0] = epsi[0] / elambda[0] * (U[1] - U[0]) / ( (self.Abar[1] - self.Abar[0]) )

        self.timer.stop("k_coeffs")
        return kxi, kR, km, kU

    def check_progress(self, n_steps) :
        
        mass, max2moR = self.BH_mass2()
        self.mass_data = np.append(self.mass_data, mass)
        self.max2moR_data = np.append(self.max2moR_data, max2moR)

        if(max2moR > self.mOverR):
            print('2m/R is larger than ' + str(self.mOverR))
            return 1

        if(self.step%20==0) :
            self.display.value = "Running HM sim, step "+str(self.step)+" of max "+str(n_steps)\
                +". Current u is "+str(self.u)+", max 2m/R is currently "+str(max2moR)+".<br />"\
                +"Time Elapsed is: "+str(time.process_time() - self.start_time)+" s"

        if(self.step%1000==0) :
            print("u:", self.u, "time:", time.process_time() - self.start_time,
                "step:", self.step, "max2moR:", max2moR, "mass:", mass)

        return 0

    def extrap_mass(self, start=-25, len=15, incr=500) :
        mextraps = []

        for s in range(start, -len-1) :
            e=s+len
            x = self.max2moR_data[s*incr:e*incr:incr]
            y = self.mass_data[s*incr:e*incr:incr]
            fit = stats.linregress(x, y)
            mextrap = fit.slope*1 + fit.intercept
            mextraps.append(mextrap)

        self.mextraps = mextraps
        return mextraps[-1], np.mean(mextraps), np.std(mextraps)

    def adap_run_steps(self, n_steps, adj_intv=-1, tol=1e-7) :

        deltau = self.deltau_adap
        self.deltau = deltau
        self.set_Q_old(self.R, self.m, self.U, self.xi)

        kxi1, kR1, km1, kU1 = self.k_coeffs(self.R, self.m, self.U,  self.xi)

        while(self.step < n_steps) :
            self.timer.start("adap_run_steps")

            # if(self.step % 200 == 0) : 
                #plt.plot(np.sqrt(np.exp(2 * (1 - self.alpha) * self.xi)))
                # plt.semilogy(self.R**2 * self.m * self.Abar**2 * np.exp(2 * (self.alpha-1) * self.xi))
            if (deltau < 1e-10):
                print("Warning, the time step is too small!")
                break
            if self.check_progress(n_steps) > 0 :
                break

            self.deltau = deltau
            kxi2, kR2, km2, kU2 = self.k_coeffs(self.R + deltau/2*kR1, self.m + deltau/2*km1,
                                                 self.U + deltau/2*kU1, self.xi + deltau/2*kxi1)
            kxi3, kR3, km3, kU3 = self.k_coeffs(self.R + 3*deltau/4*kR2, self.m + 3*deltau/4*km2,
                                                 self.U + 3*deltau/4*kU2, self.xi + 3*deltau/4*kxi2)



            xi_new = self.xi + deltau/9*(2*kxi1 + 3*kxi2 + 4*kxi3 )
            R_new = self.R + deltau/9*(2*kR1 + 3*kR2 + 4*kR3 )
            m_new = self.m + deltau/9*(2*km1 + 3*km2 + 4*km3 )
            U_new = self.U + deltau/9*(2*kU1 + 3*kU2 + 4*kU3 )

            kxi4, kR4, km4, kU4 = self.k_coeffs(R_new , m_new , U_new, xi_new)

            E_xi = np.max( np.abs(deltau * (-5*kxi1/72 + kxi2/12 + kxi3/9 - kxi4/8)))
            E_R = np.max( np.abs((deltau * (-5*kR1/72 + kR2/12 + kR3/9 - kR4/8))))
            E_m = np.max( np.abs(deltau * (-5*km1/72 + km2/12 + km3/9 - km4/8)))
            E_U = np.max( np.abs(deltau * (-5*kU1/72 + kU2/12 + kU3/9 - kU4/8)))

            max_err_xi = np.max(np.abs(self.xi)) * tol
            max_err_R = np.max(np.abs(self.R)) * tol
            max_err_m = np.max(np.abs(self.m)) * tol
            max_err_U = np.max(np.abs(self.U)) * tol

            if(E_xi < max_err_xi and E_R < max_err_R and E_m < max_err_m and E_U < max_err_U):
                self.xi = xi_new
                self.R = R_new
                self.m = m_new
                self.U = U_new

                kxi1 = np.copy(kxi4)
                kR1 = np.copy(kR4)
                km1 = np.copy(km4)
                kU1 = np.copy(kU4)

                self.step += 1
                self.set_Q_old(self.R, self.m, self.U, self.xi)
                self.u += deltau

            # Adjust step size.
            self.q = 0.8*np.min((max_err_xi/E_xi, max_err_R/E_R, max_err_m/E_m, max_err_U/E_U) )**(1/3)   # conservative optimal step factor
            self.q = min(self.q,10)               # limit stepsize growth
            deltau *= self.q
            self.deltau_adap = deltau

            self.timer.stop("adap_run_steps")


    def run_steps(self, n_steps, adj_intv=-1) :

        deltau = self.deltau_i
        self.deltau = deltau
        self.set_Q_old(self.R, self.m, self.U, self.xi)

        while(self.step < n_steps) :
            self.timer.start("run_steps")

            if self.check_progress(n_steps) > 0 :
                break

            if(adj_intv > 0 and self.step % adj_intv == 0):
                deltau = self.cfl_deltau(self.R, self.m, self.U, self.xi) * 0.05

            self.deltau = deltau
            #der_U = dfdA(np.exp(self.xi * (self.alpha-1)) * self.Abar * self.R * self.U , self.Abar, 1e100)

            #self.Q = self.kappa * (self.Abar[1])**2 * der_U**2
            #Q = self.w*np.ones_like(p)
            #self.Qprime = dfdA(self.Q , self.Abar, 1e100)
            #self.Q_du = (Q - self.Q_old) / deltau

            #self.Q[der_U > 0] = 0
            #self.Qprime[der_U>0] = 0
            #self.Q_du[der_U>0] = 0

            kxi1, kR1, km1, kU1 = self.k_coeffs(self.R, self.m, self.U,  self.xi)
            kxi2, kR2, km2, kU2 = self.k_coeffs(self.R + deltau/2*kR1, self.m + deltau/2*km1,
                                                 self.U + deltau/2*kU1, self.xi + deltau/2*kxi1)
            kxi3, kR3, km3, kU3 = self.k_coeffs(self.R + deltau/2*kR2, self.m + deltau/2*km2,
                                                 self.U + deltau/2*kU2, self.xi + deltau/2*kxi2)
            kxi4, kR4, km4, kU4 = self.k_coeffs(self.R + deltau*kR3, self.m + deltau*km3,
                                                 self.U + deltau*kU3,  self.xi + deltau*kxi3)

            # print(deltau, ((kU1 + 2*kU2 + 2*kU3 + kU4)), self.U)

            self.xi = self.xi + (deltau/6*(kxi1 + 2*kxi2 + 2*kxi3 + kxi4))
            self.R = self.R + (deltau/6*(kR1 + 2*kR2 + 2*kR3 + kR4))
            self.m = self.m + (deltau/6*(km1 + 2*km2 + 2*km3 + km4))
            self.U = self.U + (deltau/6*(kU1 + 2*kU2 + 2*kU3 + kU4))

            self.set_Q_old(self.R, self.m, self.U, self.xi)
            self.step += 1
            self.u += deltau

            self.timer.stop("run_steps")

    def BH_mass(self):
        xi = self.xi
        R = self.R
        m = self.m
        U = self.U

        mOverR = (R**2 * m * self.Abar**2 * np.exp(2*(self.alpha-1)*xi))

        if(mOverR.max() < self.mOverR):
            print('2m/R is less than the threshold, no BH forms!')
            return -1

        xiprime = WENO_dfdA(xi, self.Abar, 1e100)
        Rprime = WENO_dfdA(R, self.Abar, 1e100)
        mprime = WENO_dfdA(m, self.Abar, 1e100)

        g = self.gamma(R, m, U, xi)
        r = self.rho(R, m, U, xi, g, xiprime, Rprime, mprime)
        exi = np.exp(xi)
        ephi = self.ephi(R, U, g, xi, xiprime, Rprime)
        elambda = self.elambda(ephi, exi, xiprime)
        epsi = self.epsi(R, U, g, xi, r, ephi)
        for pos in range(self.N):
            if epsi[pos] > 1e-6:
                break

        a = np.exp(self.alpha * self.xi)
        H = np.exp(-self.xi) / self.RH
        rho_b = a**(1+self.w0)
        Rb = a * self.A

        return (( (np.exp(-self.xi/2) * self.R **3 * self.Abar**3 * self.m ) / 2 )[pos])

    def BH_mass2(self):
        xi = self.xi
        R = self.R
        m = self.m
        U = self.U

        mOverR = (R**2 * m * self.Abar**2 * np.exp(2*(self.alpha-1)*xi))
        xs = np.arange(len(mOverR))
        xnearmax = max(2, min( mOverR.argmax(), len(mOverR)-2 ))
        fn_mOverR = interp1d(xs, -1*mOverR, kind='cubic')
        xatmax = minimize_scalar(fn_mOverR, bounds=(xnearmax-1, xnearmax+1), method='bounded')
        xatmax = xatmax.x

        mass_expr = (np.exp(-self.xi/2) * self.R **3 * self.Abar**3 * self.m ) / 2
        fn_mass_expr = interp1d(xs, mass_expr, kind='cubic')

        return fn_mass_expr(xatmax), -1.0*fn_mOverR(xatmax)


    def cfl_deltau(self, R, m, U, xi):
        xiprime = WENO_dfdA(xi, self.Abar, 1e100)
        Rprime = WENO_dfdA(R, self.Abar, 1e100)
        mprime = WENO_dfdA(m, self.Abar, 1e100)

        g = self.gamma(R, m, U, xi)
        r = self.rho(R, m, U, xi, g, xiprime, Rprime, mprime)
        #p = self.P(r)
        exi = np.exp(xi)
        ephi = self.ephi(R, U, g, xi, xiprime, Rprime)
        elambda = self.elambda(ephi, exi, xiprime)
        epsi = self.epsi(R, U, g, xi, r, ephi, self.w0)
        return ((1 / np.sqrt(self.w0) - 1)  * (self.Abar[1] - self.Abar[0]) * elambda / epsi).min()
