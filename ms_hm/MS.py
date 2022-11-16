import sys
import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.ndimage import gaussian_filter1d

from ms_hm.utils import *
from ms_hm.QCD_EOS import *

class MS:
    """
    Class to integrate the Misner-Sharp equations to study
    critical collapse of a fluid to a black hole.

    Equations integrated are 50a - 50f in
    https://arxiv.org/pdf/1504.02071.pdf .
    """

    def __init__(self, Abar, rho0, amp,
                 trace_ray=False, BH_threshold=1, dt_frac=0.025, sm_sigma=0.0,
                 plot_interval=-1, fixw=False, use_turnaround=False,
                 od_size=1.6):

        # Initial coordinate grid & number of grid points
        self.Abar = Abar
        self.N = Abar.shape[0]
        self.step = 0

        # Equation of state & background information
        self.qcd = QCD_EOS()
        if fixw :
            self.qcd.fix_w(rho0, use_turnaround)
        self.w0 = self.qcd.P(rho0)/rho0
        self.alpha = (2/3)/(1 + self.w0)
        self.t0 = self.alpha * np.sqrt(3 / (8*np.pi*rho0))
        self.t = self.t0
        self.RH = self.t0 / self.alpha
        self.Abar_stg = self.to_stg(self.Abar)
        self.A = Abar * self.RH
        if fixw :
            print("Initial w is *fixed* at", self.w0, "and Horizon radius is", self.RH)
        else :
            print("Initial w is", self.w0, "and Horizon radius is", self.RH)

        # Initial field values
        self.od_size = od_size # overdensity "width"/size
        delta0 = amp * np.exp(-Abar**2 / 2 / (od_size)**2)
        delta0P = amp * delta0 * 2 * (-1 / 2 / (od_size)**2 ) * Abar
        self.m = 1 + delta0
        self.U = 1 - self.alpha * delta0 / 2
        self.R = 1 - self.alpha / 2 * (delta0 + self.w0 * Abar * delta0P / (1 + 3 * self.w0) )

        # Plot steps every so often
        self.plot_interval = plot_interval
        self.movie_plot_interval = 0
        self.movie_frames = []

        # Stability & integration parameters and such
        self.exc_pos = -1
        self.sm_sigma = sm_sigma # smooth the density field if needed
        self.xi = 0
        self.q = 1
        self.dt_frac = dt_frac
        self.deltau_i = self.cfl_deltau(self.R, self.m, self.U) * self.dt_frac
        self.deltau_adap = self.deltau_i

        # Data to supply to a hernandez-misner run
        # initialize the poton
        self.trace_ray = trace_ray

        r = self.rho(self.R, self.m) # rho for photon tracing

        self.Abar_p = self.Abar[0]
        self.U_p = self.U[0]
        self.m_p = self.m[0]
        self.R_p = self.R[0]
        self.rho_p = r[0]

        self.U_hm = np.zeros(self.N-1)
        self.m_hm = np.zeros(self.N-1)
        self.R_hm = np.zeros(self.N-1)
        self.xi_hm = np.zeros(self.N-1)
        self.rho_hm = np.zeros(self.N-1)

        self.U_hm[0] = self.U[0]
        self.m_hm[0] = self.m[0]
        self.R_hm[0] = self.R[0]
        self.xi_hm[0] = 0
        self.rho_hm[0] = r[0]
        self.r_old = 0

        self.BH_threshold = BH_threshold
        self.field_max = 0
        self.delta = -1


    # convert to half grid
    def to_stg(self,arr):
        return (arr[0:-1] + arr[1:]) / 2

    def to_cubic_stg(self,arr):
        a1 = arr[0:-1] ** 3
        a2 = arr[1:] ** 3
        return (a1 + a2) / (np.abs(a1 + a2)) * ( np.abs(a1 + a2)/ 2) **(1/3)

    def to_idx(self, pos):
        return np.searchsorted(self.Abar, pos, "right") - 1

    def gamma(self, R, m, U, xi):
        return np.sqrt( np.exp(2 * (1 - self.alpha) * xi)
                       + (self.Abar * R)**2 * (U**2 - m) )

    def P(self, rho) :
        """
        Compute (tilded) pressure as a function of (tilded) density.
        """
        H = np.exp(-self.xi) / self.RH
        rhob = 3 / (8*np.pi) * H**2
        realRho = rho * rhob
        realP = self.qcd.P(realRho)
        P = realP/rhob # P is Ptilde
        return P

    def rho(self, R, m):
        temp = m + ms_rho_term(R, m, self.Abar)
        if self.sm_sigma > 0 :
            temp = gaussian_filter1d(temp, sigma=self.sm_sigma, mode='nearest')
        return temp

    def psi(self, rho, p, Pprime):
        offset = + np.log(rho[-1]**(-3 * self.alpha * self.qcd.dPdrho(rho[-1]) / 2))
        psi = inv_derv_phi(rho, p, offset)
        if (psi>100).any() :
            raise ValueError("Large psi detected at step "+str(self.step)+"!")
        return psi

    def Pprime(self, R, m):
        """
        Return spatial derivative of the pressure field
        """
        # staggered derivative expression:
        R_stg = self.to_stg(R)
        m_stg = self.to_stg(m)
        rho_stg = m_stg + ms_rho_term_stg(R, m, self.Abar, R_stg, self.Abar_stg)

        P_stg = self.P(rho_stg)
        dPdAbar = np.concatenate( ([0], stg_dfdA(P_stg, self.Abar_stg), [0]) )

        return dPdAbar

    def k_coeffs(self, R, m, U, Abar_p, xi) :
        """
        Compute Runge-Kutta "k" coefficients
        """
        g = self.gamma(R, m, U, xi)
        r = self.rho(R, m)
        p = self.P(r)

        Pprime = self.Pprime(R, m)
        ep = np.exp(self.psi(r, p, Pprime))

        kR = self.alpha * R * (U * ep - 1)
        km = 2 * m - 3 * self.alpha * U * ep * (p + m)

        AR_prime = R + self.Abar * dfdA(R, self.Abar, 0, self.exc_pos)

        H = np.exp(-self.xi) / self.RH
        rhob = 3 / (8*np.pi) * H**2
        realRho_A0 = r[0] * rhob
        dpdrho_A0 = self.qcd.dPdrho(realRho_A0)

        kU = U - self.alpha * ep * \
            (   g**2 * np.concatenate((
                    [ 5/3 * dpdrho_A0 * (m[1] + m[1] - 2 * m[0]) / self.Abar[1]**2 ],
                    # [ (p[1] + p[1] - 2 * p[0])/self.Abar[1]**2 ] , # <-- less stable?
                    Pprime[1:] / self.Abar[1:]
            )) / (R * (AR_prime) * (r + p)) + (2 * U**2 + m + 3 * p) / 2)

        kA_p = self.alpha * np.interp(Abar_p, self.Abar, ep * g / AR_prime)

        return kR, km, kU, kA_p

    def BH_wont_form(self):
        """
        Check to see if a BH will NOT form.
        A zero or negative mass at the origin indicates some outflow or
        other error, so a BH won't form in the remainder of the run.
        """
        r = self.rho(self.R, self.m)[4]
        if r > self.field_max :
            self.field_max = r

        if self.m[0] < self.BH_threshold :
            print("Mass near origin is negative, so a black hole likely won't be forming! This occurred at step", self.step)
            return True
        elif r < self.field_max*1/5 :
            # Check for significant mass drop in density or mass near the origin?
            print("Density near origin has dropped significantly, so a black hole likely won't be forming! This occurred at step", self.step)
            return True
        else:
            return False

    def plot_fields(self, force_plot=False) :
        """
        Plot things every so often (according to the self.plot_interval class variable value),
        or if force_plot is true (force this function to plot)
        """
        if (self.step % self.plot_interval == 0 or force_plot) and self.plot_interval > 0 :
            # First figure shows m
            plt.figure(1)
            plt.semilogy(self.Abar, self.m)
            plt.title("Mass m")

            # Second figure shows rho
            plt.figure(2)
            r = self.rho(self.R, self.m)
            plt.semilogy(self.Abar, r)
            plt.title("Density rho")

            #Third figure shows R
            plt.figure(3)
            plt.semilogy(self.R)
            plt.title("R")

            #Fourth figure shows 2m/R
            plt.figure(4)
            plt.semilogy(self.Abar, self.R**2 * self.m * self.Abar**2 * np.exp(2 * (self.alpha-1) * self.xi))
            plt.title("2m/R")

            #Fifth figure shows Pprime
            plt.figure(5)
            Pprime = self.Pprime(self.R, self.m)
            plt.plot(Pprime)
            plt.title("Pprime")

            #Sixth figure shows P
            plt.figure(6)
            p = self.P(self.rho(self.R, self.m))
            plt.semilogy(p)
            plt.title("P")

            #Seventh figure shows psi
            plt.figure(7)
            psi = self.psi(r, p, Pprime)
            plt.semilogy(psi)
            plt.title("psi")

        # Plot additional fields if force_plot is true
        if force_plot or self.movie_plot_interval > 0 :

            a = np.exp(self.alpha * self.xi)
            H = np.exp(-self.xi) / self.RH

            g = self.gamma(self.R, self.m, self.U, self.xi)
            r = self.rho(self.R, self.m)
            p = self.P(r)
            Pprime = self.Pprime(self.R, self.m)

            two_m_over_R = self.R**2 * self.m * self.Abar**2 * np.exp(2 * (self.alpha-1) * self.xi)

            fields_plot = plt.figure(10+len(self.movie_frames))
            ax = fields_plot.add_subplot(111)
            ax.loglog(self.Abar, self.U, c='b', label='Velocity U')
            ax.loglog(self.Abar, -self.U, c='b', ls=':') # plot negative U values dashed
            ax.loglog(self.Abar, r, c='g', label='Density, rho')
            ax.loglog(self.Abar, -r, c='g', ls=':') # plot negative rho values dashed
            ax.loglog(self.Abar, self.m, c='r', label='Mass, m')
            ax.loglog(self.Abar, -self.m, c='r', ls=':') # plot negative mass values dashed
            ax.loglog(self.Abar, two_m_over_R, c='k', label='Horizon, 2m/R')
            ax.loglog(self.Abar, -two_m_over_R, c='k', ls=':') # plot negative mass values dashed
            ax.loglog(self.Abar, p, c='m', label='Pressure, p')
            ax.loglog(self.Abar, -p, c='m', ls=':') # plot negative pressure values dashed

            try :
                psi = self.psi(r, p, Pprime)
                ax.loglog(self.Abar, psi, c='c', label="Metric factor psi")
                ax.loglog(self.Abar, -psi, c='c', ls=':')
            except:
                pass

            ax.hlines(1.0, self.Abar[1], self.Abar[-1])
            ax.vlines(self.Abar_p, 10**-3, 10**4, colors='y', linestyles='solid')
            exc_pos = int(self.exc_pos)
            if exc_pos > 0 :
                ax.vlines(self.Abar[exc_pos], 10**-3, 10**4, colors='y', linestyles='dashed')
            ax.set_ylim(10**-3, 10**4)
            ax.legend()

            if (self.movie_plot_interval > 0) and (self.step % self.movie_plot_interval == 0) :
                print("Generating plot")
                self.movie_frames.append([ax])
                plt.close()

    def get_exc_arr(self) :
        """
        Get array of positions to evaluate=1, excise=0
        """
        if(self.to_idx(self.Abar_p) > 0.1*self.N and self.to_idx(self.Abar_p) < self.N ):
            horizon_pos = find_exc_pos(self.R**2 * self.m * self.Abar**2 * np.exp(2 * (self.alpha-1) * self.xi))
            # self.exc_pos = np.max([self.exc_pos, horizon_pos-1])
            self.exc_pos = np.max([self.exc_pos, horizon_pos, self.to_idx(self.Abar_p) - 10])
        
        return np.concatenate(( [0]*(self.exc_pos + 1), [1]*(self.N - self.exc_pos - 1) ))


    def run_steps(self, n_steps) :
        """

        """
        deltau = self.deltau_i
        if(self.trace_ray == True):
            print("Tracing ray is enabled and excision will be performed!")
        else:
            print('Not tracing ray and NO excision will be performed!')

        while(self.step < n_steps) :

            # Plot things every so often
            self.plot_fields()

            # Stop running if it becomes clear a BH won't form
            if (self.BH_wont_form() == True) :
                return -2

            # when density perturbation enters the cosmic horizon
            if(self.delta == -1):
                r = self.rho(self.R, self.m)
                pos = zero_crossing(self.Abar, (self.Abar - 1/np.exp((self.alpha-1) * self.xi) / self.R))
                if(pos > 0 and np.interp(pos, self.Abar, r) < 1):
                    self.delta = np.interp(pos, self.Abar, self.m) - 1

            exc_arr = self.get_exc_arr()

            kR1, km1, kU1, kA_p1 = self.k_coeffs(self.R, self.m, self.U, self.Abar_p, self.xi)
            kR2, km2, kU2, kA_p2 = self.k_coeffs(self.R + deltau/2*kR1, self.m + deltau/2*km1,
                                                 self.U + deltau/2*kU1, self.Abar_p + deltau/2*kA_p1,
                                                self.xi + deltau / 2)
            kR3, km3, kU3, kA_p3 = self.k_coeffs(self.R + deltau/2*kR2, self.m + deltau/2*km2,
                                                 self.U + deltau/2*kU2, self.Abar_p + deltau/2*kA_p2,
                                                self.xi + deltau/2)
            kR4, km4, kU4, kA_p4 = self.k_coeffs(self.R + deltau*kR3, self.m + deltau*km3,
                                                 self.U + deltau*kU3, self.Abar_p + deltau*kA_p3,
                                                self.xi + deltau)

            self.R = self.R + (deltau/6*(kR1 + 2*kR2 + 2*kR3 + kR4)) * exc_arr
            self.m = self.m + (deltau/6*(km1 + 2*km2 + 2*km3 + km4)) * exc_arr
            self.U = self.U + (deltau/6*(kU1 + 2*kU2 + 2*kU3 + kU4)) * exc_arr

            if(self.trace_ray == True):
                Abar_p_new = self.Abar_p + deltau/6*(kA_p1 + 2*kA_p2 + 2*kA_p3 + kA_p4)
                idx_p_new = self.to_idx(Abar_p_new)

                U_p_new = np.interp(Abar_p_new, self.Abar, self.U)
                m_p_new = np.interp(Abar_p_new, self.Abar, self.m)
                R_p_new = np.interp(Abar_p_new, self.Abar, self.R)
                rho_p_new = np.interp(Abar_p_new, self.Abar, self.rho(self.R, self.m))

                diff = idx_p_new - self.to_idx(self.Abar_p)
                if (diff > 1): # move across more than two grid pints!
                    print('Warning! ' + str(self.Abar_p) + ' ' + str(Abar_p_new))
                    print("Code stopped running on step", self.step)
                    return 2
                if ( diff > 0): # move across one grid point
                    interp_w = (self.Abar[idx_p_new] - self.Abar_p) / (Abar_p_new - self.Abar_p)
                    # linear interpolation
                    self.U_hm[idx_p_new] = U_p_new * interp_w + self.U_p * (1 - interp_w)
                    self.m_hm[idx_p_new] = m_p_new * interp_w + self.m_p * (1 - interp_w)
                    self.R_hm[idx_p_new] = R_p_new * interp_w + self.R_p * (1 - interp_w)
                    self.xi_hm[idx_p_new] = self.xi + deltau * interp_w
                    self.rho_hm[idx_p_new] = rho_p_new * interp_w + self.rho_p * (1 - interp_w)

                if(self.xi >= self.xi_hm[0]): # only start advancing photon when the system time is large enough
                    self.Abar_p = Abar_p_new
                    self.U_p = U_p_new
                    self.m_p = m_p_new
                    self.R_p = R_p_new
                    self.rho_p = rho_p_new

                if(idx_p_new == self.N-2): # going out of the boundary
                    print("Photon has gone out of the outter boundary!")
                    return 0

            self.step += 1
            self.xi += deltau

            if(self.step % 10 == 0 and self.trace_ray==False):
                if(find_exc_pos(self.R**2 * self.m * self.Abar**2 * np.exp(2 * (self.alpha-1) * self.xi)) > 0):
                    print("Horizon was found at step", self.step, "! code will be terminated.")
                    return -1

        self.plot_fields(force_plot=True)


    def adap_run_steps(self, n_steps, adjust_steps=100, tol=1e-7) :

        deltau = self.deltau_adap

        if(self.trace_ray == True):
            print("Tracing ray is enabled and excision will be performed!")
        else:
            print('Not Tracing ray and NO excision will be performed!')


        while(self.step < n_steps):

            # Plot things every so often
            self.plot_fields()

            # Stop running if it becomes clear a BH won't form.
            if(self.BH_wont_form() == True):
                return -2

            if (deltau < 1e-10):
                print("Warning, the time step is too small! Stopping run at step "
                    +str(self.step)+" with timestep "+str(deltau))
                return 1

            exc_arr = self.get_exc_arr()

            if(self.delta == -1):
                r = self.rho(self.R, self.m)
                pos = zero_crossing(self.Abar, (self.Abar - 1/np.exp((self.alpha-1) * self.xi) / self.R))
                if(pos > 0 and np.interp(pos, self.Abar, r) < 1):
                    self.delta = np.interp(pos, self.Abar, self.m) - 1


            kR1, km1, kU1, kA_p1 = self.k_coeffs(self.R, self.m, self.U, self.Abar_p, self.xi)
            kR2, km2, kU2, kA_p2 = self.k_coeffs(self.R + deltau/2*kR1, self.m + deltau/2*km1,
                                                 self.U + deltau/2*kU1, self.Abar_p + deltau/2*kA_p1,
                                                 self.xi + deltau/2)
            kR3, km3, kU3, kA_p3 = self.k_coeffs(self.R + 3/4*deltau*kR2, self.m + 3/4*deltau*km2,
                                                 self.U + 3/4*deltau*kU2, self.Abar_p + 3/4*deltau*kA_p2,
                                                self.xi + 3/4*deltau)


            Abar_p_new = self.Abar_p + deltau/9*(2*kA_p1 + 3*kA_p2 + 4*kA_p3 )
            R_new = self.R + deltau/9*(2*kR1 + 3*kR2 + 4*kR3 )
            m_new = self.m + deltau/9*(2*km1 + 3*km2 + 4*km3 )
            U_new = self.U + deltau/9*(2*kU1 + 3*kU2 + 4*kU3 )

            # calculating the difference of the photon position
            # diff is always zero when tracing_ray is false
            idx_p_new = self.to_idx(Abar_p_new)
            diff = idx_p_new - self.to_idx(self.Abar_p)

            kR4, km4, kU4, kA_p4 = self.k_coeffs(R_new , m_new , U_new, Abar_p_new, self.xi + deltau)

            E_R = np.max( np.abs((deltau * (-5*kR1/72 + kR2/12 + kR3/9 - kR4/8))) * exc_arr)
            E_m = np.max( np.abs(deltau * (-5*km1/72 + km2/12 + km3/9 - km4/8)) * exc_arr)
            E_U = np.max( np.abs(deltau * (-5*kU1/72 + kU2/12 + kU3/9 - kU4/8)) * exc_arr)

            max_err_R = np.max(np.abs(self.R) * exc_arr) * tol
            max_err_m = np.max(np.abs(self.m) * exc_arr) * tol
            max_err_U = np.max(np.abs(self.U) * exc_arr) * tol

            if(diff <= 1 and E_R < max_err_R and E_m < max_err_m and E_U < max_err_U):
                self.R = self.R + deltau/9*(2*kR1 + 3*kR2 + 4*kR3 ) * exc_arr
                self.m = self.m + deltau/9*(2*km1 + 3*km2 + 4*km3 ) * exc_arr
                self.U = self.U + deltau/9*(2*kU1 + 3*kU2 + 4*kU3 ) * exc_arr

                if(self.trace_ray == True):

                    U_p_new = np.interp(Abar_p_new, self.Abar, self.U)
                    m_p_new = np.interp(Abar_p_new, self.Abar, self.m)
                    R_p_new = np.interp(Abar_p_new, self.Abar, self.R)
                    rho_p_new = np.interp(Abar_p_new, self.Abar, self.rho(self.R, self.m))

                    if ( diff > 0): # move across one grid point
                        interp_w = (self.Abar[idx_p_new] - self.Abar_p) / (Abar_p_new - self.Abar_p)
                        # linear interpolation
                        self.U_hm[idx_p_new] = U_p_new * interp_w + self.U_p * (1 - interp_w)
                        self.m_hm[idx_p_new] = m_p_new * interp_w + self.m_p * (1 - interp_w)
                        self.R_hm[idx_p_new] = R_p_new * interp_w + self.R_p * (1 - interp_w)
                        self.xi_hm[idx_p_new] = self.xi + deltau * interp_w
                        self.rho_hm[idx_p_new] = rho_p_new * interp_w + self.rho_p * (1 - interp_w)

                    self.Abar_p = Abar_p_new
                    self.U_p = U_p_new
                    self.m_p = m_p_new
                    self.R_p = R_p_new
                    self.rho_p = rho_p_new

                kR1 = kR4.copy()
                km1 = km4.copy()
                kU1 = kU4.copy()
                kA_p1 = kA_p4

                self.step += 1
                self.xi += deltau
                if(idx_p_new == self.N-2): #going out of the boundary
                    print("Photon has gone out of the outter boundary at step", self.step)
                    return 0

            if( diff <= 1 ):
                # Adjust step size.
                self.q = 0.75*np.min((max_err_R/E_R, max_err_m/E_m, max_err_U/E_U) )**(1/3)   # conservative optimal step factor
                self.q = min(self.q, 5)               # limit stepsize growth
                deltau *= self.q

            else:
                # reducing the deltau since diff is too large
                deltau /= 2.0

            self.deltau_adap = deltau

            if(self.step % 10 == 0 and self.trace_ray==False):
                if(find_exc_pos(self.R**2 * self.m * self.Abar**2 * np.exp(2 * (self.alpha-1) * self.xi)) > 0):
                    print("Horizon is found, code will be terminated! Finished at step", self.step)
                    return -1

        self.plot_fields(force_plot=True)


    def cfl_deltau(self, R, m, U):
        a = np.exp(self.alpha * self.xi)
        H = np.exp(-self.xi) / self.RH

        g = self.gamma(R, m, U, self.xi)
        r = self.rho(R, m)
        p = self.P(r)

        Pprime = self.Pprime(R, m)

        ep = np.exp(self.psi(r, p, Pprime))
        el =  (dfdA(a * self.A * R, self.Abar, 1) / self.RH) /(a * H * self.RH * g)

        return (np.log(1 + el / ep / np.exp(self.xi)
                       * self.alpha * np.concatenate( ([1e10],(self.Abar[1:] - self.Abar[0:-1])) ) / np.sqrt(self.w0))).min()

    # def movie(self):

    #     fig = plt.figure()
    #     # plt.close()

    #     anim = animation.ArtistAnimation(fig, self.movie_frames,
    #         interval=100, repeat=False)
    #     HTML(anim.to_jshtml())
