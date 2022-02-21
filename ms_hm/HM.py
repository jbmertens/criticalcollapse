import sys
import numpy as np
import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from ms_hm.utils import *


class HM:

    def __init__(self, MS, mOverR=0.99, sm_sigma=5):
        self.R = MS.R_hm
        self.m = MS.m_hm
        self.U = MS.U_hm
        self.xi = MS.xi_hm
        self.N = MS.R_hm.shape[0]

        self.w = MS.w
        self.A = MS.A[:self.N]
        self.alpha = MS.alpha


        self.t0 = MS.t0
        self.t = self.t0
        self.u = 0

        self.RH = self.t0 / self.alpha
        self.Abar = MS.Abar[:self.N]
        self.Abar_stg = self.to_stg(self.Abar)
        #self.Abar_stg = WENO_to_stg(self.Abar)
        self.Vbar = np.concatenate(
            ([MS.Abar[1] ], (MS.Abar[2:] - MS.Abar[0:self.N-1]) / 2) )

        self.kappa = 2
        self.Q = np.zeros(self.N)
        self.Q_du = np.zeros(self.N)
        self.Q_old = np.zeros(self.N)
        self.Qprime = np.zeros(self.N)

        self.deltau_i = self.cfl_deltau(self.R, self.m, self.U, self.xi) * 0.05
        self.deltau_adap = self.deltau_i

        self.mOverR = mOverR
        self.sm_sigma = sm_sigma
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
        return self.w * rho
    def rho(self, R, m, U, xi, g, xiprime, Rprime, mprime):
        temp = (g + self.Abar * R * U) / (g - (self.w + self.Q ) * self.Abar * R * U ) \
            * (m + self.Abar * R * hm_rho_term(R, m, self.Abar, xi, self.alpha) / 3)
        #temp = scipy.signal.savgol_filter(temp, 91, 3, mode='interp')
        temp=gaussian_filter1d(temp, sigma = self.sm_sigma, mode='nearest')
        return temp

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

        if self.Q.shape[0] > 1:
            Q_stg = self.to_stg(self.Q)
        else:
            Q_stg = self.Q

        temp = (g_stg + A_stg * R_stg * U_stg) / (g_stg - (self.w + Q_stg ) * A_stg * R_stg * U_stg ) \
            * (m_stg + A_stg * R_stg * rho_term_stg(R, m, self.Abar, xi, R_stg, m_stg, A_stg, self.alpha) / 3)
        #temp = scipy.signal.savgol_filter(temp, 31, 3, mode='interp')
        return temp

    def ephi(self, R, U, g, xi, xiprime, Rprime):
        return ephi_term(R, U, self.Abar, xi, g, self.alpha)

    def elambda(self, ephi, exi, xiprime):
        return self.alpha * ephi * exi * xiprime

    def epsi(self, R, U, g, xi, rho, ephi):
        c = self.alpha - 1 + ephi * self.Abar * R * rho * (1 + self.w + self.Q) /\
            ((g + self.Abar * R * U) * (1+self.w))
        offset = np.log(1/ephi[-1] * (g[-1] + self.Abar[-1] * R[-1] * U[-1]))
        temp = inv_derv_psi(xi, c, offset)
        return  (g + self.Abar * R * U) / np.exp(temp)

    def drho(self, R, m, U, g, xi, Rp, mp, xip):
        rho_stg = self.rho_stg(R, m, U, xi, g, xip, Rp, mp)
        return np.concatenate( ([0], stg_dfdA(rho_stg, self.to_stg(self.Abar)) ,[0]) )




    def k_coeffs(self, R, m, U, xi) :
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

        exi = np.exp(xi)
        ephi = self.ephi(R, U, g, xi, xiprime, Rprime)
        elambda = self.elambda(ephi, exi, xiprime)
        epsi = self.epsi(R, U, g, xi, r, ephi)


        drho = self.drho(R, m, U, g, xi, Rprime, mprime, xiprime)
        #drho  = dfdA(r, self.Abar, 1e100)
        drho[0] = 3 * elambda[0] / exi[0] * ((1 + self.w) * r[0] / ephi[0] - (r[0] + p[0]) * U[0])
        drho[-1] = 0

        kxi = epsi / ephi / np.exp(xi) / self.alpha

        kR = epsi / exi * R * (U - 1/ephi)

        km = 3 * epsi / exi * (1/ephi * m * (1+self.w) - U * (p +m))

        kU = - epsi / exi / (1 - self.w - self.Q) * (
            (m + 3 * p) / 2 + U**2 - U / self.alpha / ephi
        + (self.w + self.Q) * exi / elambda * Uprime + g * (self.w + self.Q) / ( np.concatenate( ([1], self.Abar[1:]) ) \
                                                                                * R * (1+self.w + self.Q)) * (
            3 * self.Q * U + exi * (self.Qprime / elambda - self.Q_du / epsi) / (self.w + self.Q)
         + 3 * (1 + self.w) * (U - 1 / ephi) + exi / elambda * drho / r) )

        # boundary conditions
        kxi[0] = epsi[0] / elambda[0] * (xi[1] - xi[0]) / ( (self.Abar[1] - self.Abar[0]) )
        kR[0] = epsi[0] / elambda[0] * (R[1] - R[0]) / ( (self.Abar[1] - self.Abar[0]) )
        km[0] = epsi[0] / elambda[0] * (m[1] - m[0]) / ( (self.Abar[1] - self.Abar[0]) )
        kU[0] = epsi[0] / elambda[0] * (U[1] - U[0]) / ( (self.Abar[1] - self.Abar[0]) )

        return kxi, kR, km, kU

    def adap_run_steps(self,n_steps, adj_intv=-1, tol=1e-7) :
        step = 0
        deltau = self.deltau_adap
        kxi1, kR1, km1, kU1 = self.k_coeffs(self.R, self.m, self.U,  self.xi)
        while(step < n_steps) :

            if (deltau < 1e-10):
                print("Warning, the time step is too small!")
                break
            if((self.R**2 * self.m * self.Abar**2 * np.exp(2*(self.alpha-1)*self.xi)).max() > self.mOverR):
                print('2m/R is larger than ' + str(self.mOverR))
                return 1

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

                step+=1
                self.u += deltau

            # Adjust step size.
            self.q = 0.8*np.min((max_err_xi/E_xi, max_err_R/E_R, max_err_m/E_m, max_err_U/E_U) )**(1/3)   # conservative optimal step factor
            self.q = min(self.q,10)               # limit stepsize growth
            deltau *= self.q
            self.deltau_adap = deltau


    def run_steps(self,n_steps, adj_intv=-1) :
        step = 0

        deltau = self.deltau_i
        while(step < n_steps) :
            if(adj_intv > 0 and step % adj_intv == 0):
                deltau = self.cfl_deltau(self.R, self.m, self.U, self.xi) * 0.05

            #der_U = dfdA(np.exp(self.xi * (self.alpha-1)) * self.Abar * self.R * self.U , self.Abar, 1e100)

            #self.Q = self.kappa * (self.Abar[1])**2 * der_U**2
            #self.Qprime = dfdA(self.Q , self.Abar, 1e100)
            #self.Q_du = (self.Q - self.Q_old) / deltau

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

            self.xi = self.xi + (deltau/6*(kxi1 + 2*kxi2 + 2*kxi3 + kxi4))
            self.R = self.R + (deltau/6*(kR1 + 2*kR2 + 2*kR3 + kR4))
            self.m = self.m + (deltau/6*(km1 + 2*km2 + 2*km3 + km4))
            self.U = self.U + (deltau/6*(kU1 + 2*kU2 + 2*kU3 + kU4))

            self.Q_old = self.Q

            step+=1
            self.u += deltau

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
        rho_b = a**(1+self.w)
        Rb = a * self.A

        return (( (np.exp(-self.xi/2) * self.R **3 * self.Abar**3 * self.m ) / 2 )[pos])

    def BH_mass2(self):
        xi = self.xi
        R = self.R
        m = self.m
        U = self.U

        mOverR = (R**2 * m * self.Abar**2 * np.exp(2*(self.alpha-1)*xi))


        return (( (np.exp(-self.xi/2) * self.R **3 * self.Abar**3 * self.m ) / 2 )[mOverR.argmax()])


    def cfl_deltau(self,R, m, U, xi):
        xiprime = dfdA(xi, self.Abar, 1e100)
        Rprime = dfdA(R, self.Abar, 1e100)
        mprime = dfdA(m, self.Abar, 1e100)

        g = self.gamma(R, m, U, xi)
        r = self.rho(R, m, U, xi, g, xiprime, Rprime, mprime)
        #p = self.P(r)
        exi = np.exp(xi)
        ephi = self.ephi(R, U, g, xi, xiprime, Rprime)
        elambda = self.elambda(ephi, exi, xiprime)
        epsi = self.epsi(R, U, g, xi, r, ephi)
        return ((1 / np.sqrt(self.w) - 1)  * (self.Abar[1] - self.Abar[0]) * elambda / epsi).min()
