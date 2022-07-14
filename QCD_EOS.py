import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as interp

class QCD_EOS:
    """
    Functionality for computing quantities associated with
    QCD-type matter (esp. equation of state).

    For PBH formation, see:
    https://arxiv.org/pdf/1801.06138.pdf
    """

    def __init__(self,
            min_rho=1e2, max_rho=1e17, # bounds to use the "correct" QCD EOS in (otherwise, w=1/3)
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
        self.Pinterp = interp.InterpolatedUnivariateSpline(self.rho, self.Pressure)

    def fix_w(self, rho0) :
        self.use_fixedw = True
        self.fixedw = self.dPdrho(rho0)

    def H(self, rho) :
        """
        Transition function t
        """
        mu = 1000
        return (np.tanh(rho/mu) + 1)/2

    def P(self, rho) :
        """
        Return the pressure for a given density rho
        """
        if self.use_fixedw :
            return self.fixedw * rho
        else :
            P_in = self.Pinterp(rho)
            P_out = 1/3*rho
            P = P_out + self.H(rho - self.min_rho)*self.H(self.max_rho - rho)*(P_in - P_out)
            return P

    def P_plot(self):
        # For plotting purpose we generate a fine grid
        rho_grid = np.logspace(0.01, 25, 1000)

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
        if self.use_fixedw :
            return (rho/rho)*self.fixedw
        else :
            dPdrho_in = self.Pinterp.derivative()(rho)
            dPdrho_out = 1/3*np.ones_like(rho)
            dPdrho = dPdrho_out + self.H(rho - self.min_rho)*self.H(self.max_rho - rho)*(dPdrho_in - dPdrho_out)
            return dPdrho

    def dPdrho_plot(self):
        # For plotting purpose we generate a fine grid
        rho_grid = np.logspace(0.01, 25, 1000)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(rho_grid, self.dPdrho(rho_grid), 'b-', lw=2, label='Spline')
        ax.legend(loc='best')
        ax.set_xlabel(r'Density, $\rho$ (MeV$^4$)')
        ax.set_ylabel(r'Derivative, $dP/d\rho$');
