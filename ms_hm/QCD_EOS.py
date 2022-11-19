import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as interp

class QCD_EOS:
    """
    Functionality for computing quantities associated with
    QCD-type matter (esp. equation of state).

    For PBH formation background in this setting, see:
    https://arxiv.org/pdf/1801.06138.pdf
    And for tabulated data used here, see table S2:
    https://arxiv.org/pdf/1606.07494.pdf
    """

    def __init__(self,
            min_rho=1e0, max_rho=1e23, # bounds to use the "correct" QCD EOS in (otherwise, w=1/3)
            mu=100 # Width of transition from QCD -> w=1/3
            ) :
        """
        Constructor to initialize class.
        """

        # Return P = rho/3 when outside this range
        self.min_rho = min_rho
        self.max_rho = max_rho

        # call fix_w() to use a fixed w instead of
        self.use_fixedw = False
        self.fixedw = 1/3

        mpl.rc('xtick', direction='in', top=True)
        mpl.rc('ytick', direction='in', right=True)
        mpl.rc('xtick.minor', visible=True)
        mpl.rc('ytick.minor', visible=True)

        # Useful arrays
        self.T = np.array([1, 3.1623, 10, 17.783, 39.811, 100, 141.254, 158.489,
                      251.189, 316.228, 1000, 10000, 19952.623, 39810.717,
                      100000, 281838.293])
        self.geff = np.array([10.71, 10.74, 10.76, 11.09, 13.68, 17.61, 24.07, 29.84,
                         47.83, 53.04, 73.48, 83.10, 85.56, 91.97, 102.17, 104.98])
        self.rho = np.pi**2 / 30 * self.geff * self.T**4
        self.hoverg = np.array([1.00228, 1.00029, 1.00048, 1.00505, 1.02159, 1.02324, 1.05423,
                               1.07578, 1.06118, 1.04690, 1.01778, 1.00123, 1.00589, 1.00887,
                               1.00750, 1.00023])
        self.Pressure = (4/3/self.hoverg - 1) * self.rho

        # Use polynomial interpolation and a spline
        # Probably want to generalize this so the function returns P = rho/3 outside of the tabulated range
        self.gOfT = interp.InterpolatedUnivariateSpline(self.T, self.geff, ext=3)
        self.logT_of_logrho = interp.InterpolatedUnivariateSpline(np.log(self.rho), np.log(self.T), ext=0)
        self.loglog_Pinterp = interp.InterpolatedUnivariateSpline(np.log(self.rho), np.log(self.Pressure) )

    def fix_w(self, rho0, use_turnaround=False) :
        if use_turnaround :
            rho_ta = 0.1*rho0 # approximate density at turnaround time,
                              # See eq. 9 in 1801.06138
            self.fixedw = self.dPdrho(rho_ta)
        else :
            self.fixedw = self.dPdrho(rho0)
        self.use_fixedw = True

    def H(self, rho, rho0) :
        """
        Transition function t
        """
        mu = 2
        return ( 1 + np.tanh( (np.log(rho) - np.log(rho0))/mu ) )/2

    def P(self, rho) :
        """
        Return the pressure for a given density rho
        """
        h = np.heaviside(rho - 1.0e-10, 0.0)
        rho = rho*h + 1.0e-10*(1-h) # minimum density floor
        if self.use_fixedw :
            return self.fixedw * rho
        else :
            P_in = np.exp(self.loglog_Pinterp(np.log(rho)))
            P_out = 1/3*rho
            P = P_out + self.H(rho, self.min_rho)*self.H(self.max_rho, rho)*(P_in - P_out)
            return P

    def TofRho(self, rho) :
        # return T of rho, scaled appropriately at the bounds.
        return np.exp(self.logT_of_logrho(np.log(rho)))

    def P_plot(self):
        # For plotting purpose we generate a fine grid
        rho_grid = np.logspace(-3, 27, 1000)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(self.rho, self.Pressure/self.rho, 'ko', label='Tabulated Values')
        ax.semilogx(rho_grid, self.P(rho_grid)/rho_grid, 'b-', lw=2, label='Spline')
        ax.legend(loc='best')
        ax.set_xlabel(r'Density, $\rho$ (MeV$^4$)')
        ax.set_ylabel(r'Ratio $P/\rho$');

    def dPdrho(self, rho) :
        """
        Return the pressure derivative wrt. rho, i.e. the equation of
        state parameter w in the constant w case.
        """
        h = np.heaviside(rho - 1.0e-10, 0.0)
        rho = rho*h + 1.0e-10*(1-h) # minimum density floor
        if self.use_fixedw :
            return np.ones_like(rho)*self.fixedw
        else :
            P = self.P(rho)
            dPdrho_in = P/rho*self.loglog_Pinterp.derivative()(np.log(rho))
            dPdrho_out = 1/3*np.ones_like(rho)
            dPdrho = dPdrho_out + self.H(rho, self.min_rho)*self.H(self.max_rho, rho)*(dPdrho_in - dPdrho_out)
            # dPdrho = dPdrho_in
            return dPdrho

    def dPdrho_plot(self):
        # For plotting purpose we generate a fine grid
        rho_grid = np.logspace(-3, 27, 1000)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(rho_grid, self.dPdrho(rho_grid), 'b-', lw=2, label='Spline')
        ax.legend(loc='best')
        ax.set_xlabel(r'Density, $\rho$ (MeV$^4$)')
        ax.set_ylabel(r'Derivative, $dP/d\rho$');
        
    def MH(self, T):
        # Convert temperature to a horizon mass
        Goverhc = 6.7e-45 # G in units of MeV^-2
        solarMassperMeV = 1 / 1.12e60
        return solarMassperMeV*( (8*np.pi*Goverhc/3)**(-3/2) * (np.pi**2*self.gOfT(T)/30)**(-1/2) * T**-2)

    def MH_plot(self):
        # Plot horizon mass as a function of temperature

        # For plotting purpose we generate a fine grid
        T_grid = np.logspace(0, 6, 1000)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.loglog(self.T, self.MH(self.T), 'ko', label='Tabulated Values')
        ax.loglog(T_grid, self.MH(T_grid), 'b-', lw=2, label='Spline')
        ax.legend(loc='best')
        ax.set_xlabel(r'Temperature, $T$ (MeV)')
        ax.set_ylabel(r'Horizon Mass');
        