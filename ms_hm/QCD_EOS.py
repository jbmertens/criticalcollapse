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

    def __init__(self, min_rho=1e2, max_rho=1e17) :

        # return P = rho/3 when outside this range
        self.min_rho = min_rho
        self.max_rho = max_rho

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


    def P(self, rho) :
        # Convert a number rho into an array, just so we know we're
        # working with an array
        if np.isscalar(rho) :
            rho = np.array([rho])
        # Pressure values by default are rho/3
        P = np.copy(rho)/3.0
        # For rho values in a certain range, use the special QCD EOS
        inbounds_rho = (rho < self.max_rho) & (rho > self.min_rho)
        if np.any(inbounds_rho) :
            P[inbounds_rho] = self.Pinterp(rho[inbounds_rho])
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
        if np.isscalar(rho) :
            rho = np.array([rho])
        dPdrho = np.ones_like(rho)/3.0

        inbounds_rho = (rho < self.max_rho) & (rho > self.min_rho)
        if np.any(inbounds_rho) :
            dPdrho[inbounds_rho] = self.Pinterp.derivative()(rho[inbounds_rho])

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
