/* INCLUDES */
#include <cmath>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <getopt.h>
#include <string>
#include <vector>
#include "spline.h"

typedef double real_t;

#define PI 3.14159265358979323846264338328
#define TOL (1.0e-7) // integration tolerance
#define RHO_B_I 1.0e25
#define NFIELDS 13


int NN = 3200;
bool USE_WENO = true;
bool SMOOTH_RHO = true;
bool USE_FIXW = false;
real_t FIXW = 1.0/3.0;

#define ALLOC_N(arr_n, ARR_SIZE) \
    real_t *arr_n; \
    arr_n = (real_t *) malloc( (ARR_SIZE) * ((long long) sizeof(real_t)));

#define ALLOC(arr_n) \
    ALLOC_N(arr_n, NN)

// Arrays for equation of state interpolation
std::vector<real_t> logrhos{-23.0258509, 1.25944027,  5.86743592, 10.47443832, 12.80727797, 16.24077033,
   20.17741006, 21.87146739, 22.54685293, 24.86073802, 25.88516263, 30.81629677,
   40.14966858, 42.94194384, 45.77729053, 49.56660234, 53.73838727, 92.1034037};
std::vector<real_t> logPs{-24.1244632, 0.15168708,  4.7676633 ,  9.37390511, 11.68836246, 15.05383501,
   18.98355404, 20.54248356, 21.11727849, 23.49996688, 24.58907975, 29.64524541,
   39.04613022, 41.81963086, 44.6428769 , 48.43776105, 52.63885477, 91.0047914};
// Interpolating function
tk::spline logPoflogrho(logrhos, logPs);
// min and max values in spline
real_t SPL_MIN_RHO=1e0;
real_t SPL_MAX_RHO=1e23;

template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

real_t max(real_t *f)
{
    real_t fmax = f[0];

#pragma omp parallel for reduction(max:fmax) 
    for(int i=0; i<NN; i++)
       fmax = fmax > f[i] ? fmax : f[i];

    return fmax;
}


// Background density as a function of ell
class RhoBIntegrator
{
public:
    real_t dl = 0.002538901;
    std::vector<real_t> ls;
    std::vector<real_t> logrho_of_ls;

    tk::spline spl;

    RhoBIntegrator()
    {
        ls.push_back(0.0);
        logrho_of_ls.push_back(std::log(RHO_B_I));

        while( logrho_of_ls.back() > -7 )
        {
            real_t l = ls.back();
            real_t logrho = logrho_of_ls.back();

            real_t k1 = -3.0*( 1.0 + std::exp( logPoflogrho(logrho) - logrho ) )*dl;
            real_t k2 = -3.0*( 1.0 + std::exp( logPoflogrho(logrho+k1/2.0) - (logrho+k1/2.0) ) )*dl;
            real_t k3 = -3.0*( 1.0 + std::exp( logPoflogrho(logrho+k2/2.0) - (logrho+k2/2.0) ) )*dl;
            real_t k4 = -3.0*( 1.0 + std::exp( logPoflogrho(logrho+k3) - (logrho+k3) ) )*dl;

            ls.push_back(l+dl);
            logrho_of_ls.push_back( logrho + (k1+2*k2+2*k3+k4)/6.0 );
        }

        std::cout << "Initializing background integrator. Max l was "<<ls.back()<<"\n";
        spl.set_points(ls, logrho_of_ls, tk::spline::cspline);
    }

    real_t eval(real_t l)
    {
        return spl(l);
    }
};
RhoBIntegrator logrhob_of_l = RhoBIntegrator();

/**
 * Background density at a given l = log(a), with a the scale factor.
 */
extern "C"
real_t rho_background(real_t l)
{
    if(USE_FIXW)
        return RHO_B_I*std::exp(-3.0*(1.0+FIXW)*l);

    return std::exp( logrhob_of_l.spl(l) );
}

/**
 * Background pressure at a given l = log(a), with a the scale factor.
 */
extern "C"
real_t P_background(real_t l)
{
    if(USE_FIXW)
        return FIXW*rho_background(l);

    real_t log_rho_b = logrhob_of_l.spl(l);
    return std::exp( logPoflogrho(log_rho_b) );
}

/**
 * Transition function, in logarithmic space. Used to smoothly "turn on"
 * a function. Transition occurs over "mu" orders of magnitude.
 * 
 * Returns ~0 before rho0, ~1 after rho0.
 */
real_t H(real_t rho, real_t rho0, real_t mu = 2)
{
    return ( 1 + std::tanh( (std::log(rho) - std::log(rho0))/mu ) )/2;
}

/**
 * Return P of \rho
 */
extern "C"
real_t P_of_rho(real_t rho)
{
    if(rho < 1.0e-10)
        rho = 1.0e-10;

    if(USE_FIXW)
        return FIXW*rho;

    real_t logrho = std::log(rho);

    if(logrho < -23.0 or logrho > 93.0)
        return rho/3.0;

    real_t P_in = std::exp(logPoflogrho(logrho));
    real_t P_out = rho/3.0;
    real_t P = P_out + H(rho, SPL_MIN_RHO)*H(SPL_MAX_RHO, rho)*(P_in - P_out);

    return P;
}

/**
 * Compute \tilde{P} of \tilde{\rho}, assuming a
 * specific background density rho_b.
 */
extern "C"
real_t Pt_of_rhot(real_t rhot, real_t rho_b)
{
    real_t rho = rhot*rho_b;
    real_t P = P_of_rho(rho);
    real_t Pt = P/rho_b;
    return Pt;
}

/**
 * Compute dP/d\rho, at a specific \rho (not tilded!)
 */
extern "C" 
real_t dPdrho(real_t rho)
{
    if(rho < 1.0e-10)
        rho = 1.0e-10;

    if(USE_FIXW)
        return FIXW;

    real_t P = P_of_rho(rho);
    real_t logrho = std::log(rho);

    real_t dPdrho_in = P/rho*logPoflogrho.deriv(1, logrho);
    real_t dPdrho_out = 1.0/3.0;
    real_t dPdrho = dPdrho_out + H(rho, SPL_MIN_RHO)*H(SPL_MAX_RHO, rho)*(dPdrho_in - dPdrho_out);

    return dPdrho;
}

/**
 * Compute dP/d\rho, at a specific \rho = \tilde{rho}*\rho_b.
 */
extern "C" 
real_t dPdrho_rhot(real_t rhot, real_t rho_b)
{
    real_t rho = rhot*rho_b;
    return dPdrho(rho);
}

/**
 * Compute derivatives of a function.
 * Assume derivatives are 0 at boundaries.
 */
void dfdA(real_t *f, real_t *A, real_t *dfdA)
{
    dfdA[0] = 0;

    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN-1; i++)
    {
        dfdA[i] = 0.5* (
            (f[i+1]-f[i]) / (A[i+1]-A[i])
            + (f[i]-f[i-1]) / (A[i]-A[i-1])
        );
    }

    dfdA[NN-1] = (f[NN-1]-f[NN-2]) / (A[NN-1]-A[NN-2]);
}

/**
 * Helper function to compute WENO weighted values on staggered grid
 * https://apps.dtic.mil/sti/tr/pdf/ADA390653.pdf
 * 
 * values in _stg grid are at f[i+1/2], assuming symmetric about i=0, and Neumann outer boundary
 */
void WENO_stg(real_t *f_in, real_t *f_stg)
{
    int i=0;

    ALLOC_N( f, NN+4.0 )
    f[0] = f_in[2];
    f[1] = f_in[1];
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
        f[i+2] = f_in[i];
    f[NN+2] = f_in[NN-1];
    f[NN+3] = f_in[NN-1];

#pragma omp parallel for default(shared) private(i)
    for(i=2; i<NN+2; i++)
    {
        real_t b1 = 13.0/12.0*std::pow(f[i-2] - 2*f[i-1] + f[i], 2)
            + 1.0/4.0*std::pow(f[i-2] - 4*f[i-1] + 3*f[i], 2);
        real_t b2 = 13.0/12.0*std::pow(f[i-1] - 2*f[i] + f[i+1], 2)
            + 1.0/4.0*std::pow(f[i-1] - f[i+1], 2);
        real_t b3 = 13.0/12.0*std::pow(f[i] - 2*f[i+1] + f[i+2], 2)
            + 1.0/4.0*std::pow(3*f[i] - 4*f[i+1] - f[i+2], 2);
        real_t eps = 1.0e-8;
        real_t g1 = 0.1, g2 = 0.6, g3 = 0.3;
        real_t wt1 = g1/std::pow(eps + b1, 2);
        real_t wt2 = g2/std::pow(eps + b2, 2);
        real_t wt3 = g3/std::pow(eps + b3, 2);
        real_t wtsum = wt1 + wt2 + wt3;
        real_t w1=wt1/wtsum, w2=wt2/wtsum, w3=wt3/wtsum;

        real_t f1 = 1.0/3.0*f[i-2] - 7.0/6.0*f[i-1] + 11.0/6.0*f[i];
        real_t f2 = -1.0/6.0*f[i-1] + 5.0/6.0*f[i] + 1.0/3.0*f[i+1];
        real_t f3 = 1.0/3.0*f[i] + 5.0/6.0*f[i+1] - 1.0/6.0*f[i+2];

        f_stg[i-2] = w1*f1 + w2*f2 + w3*f3;
    }

    free(f);
}

/**
 * Weighted essentially non-oscilliatory derivative
 * https://apps.dtic.mil/sti/tr/pdf/ADA390653.pdf
 * 
 * df/dA = (df/di)/(dA/di)
 */
void dfdA_WENO(real_t *f_in, real_t *A_in, real_t *dfdA)
{
    int i=0;

    ALLOC_N( f_stg, NN )
    WENO_stg( f_in, f_stg );

    ALLOC_N( A_stg, NN )
    WENO_stg( A_in, A_stg );

#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN; i++)
    {
        dfdA[i] = (f_stg[i]-f_stg[i-1])/(A_stg[i]-A_stg[i-1]);
    }
    dfdA[0] = 0;

    free(f_stg);
    free(A_stg);
}

/**
 * Compute dPtdAb
 * Uses a "staggered" grid to improve stability.
 * Assume dPtdAb is 0 at boundaries.
 */
void dPtdAbstg(real_t *agg, real_t *dPtdAb, real_t rho_b)
{
    int i=0;

    real_t *Ab = agg + 0;
    real_t *Rt = agg + 1*NN;
    real_t *mt = agg + 2*NN;

    real_t Pt_stg = 0, Ab_stg = 0;

    for(i=0; i<NN-1; i++)
    {
        real_t Ab_stg_prev = Ab_stg;
        Ab_stg = (Ab[i]+Ab[i+1])/2;

        real_t Pt_stg_prev = Pt_stg;
        real_t Rt_stg = (Rt[i]+Rt[i+1])/2;
        real_t mt_stg = (mt[i]+mt[i+1])/2;
        real_t rhot_stg = mt_stg + Ab_stg*Rt_stg*(mt[i+1] - mt[i])
                            / 3 / (Ab[i+1]*Rt[i+1] - Ab[i]*Rt[i]);
        Pt_stg = Pt_of_rhot(rhot_stg, rho_b);

        dPtdAb[i] = (Pt_stg - Pt_stg_prev)/(Ab_stg - Ab_stg_prev);
    }

    dPtdAb[0] = 0;
    dPtdAb[NN-1] = dPtdAb[NN-2];
}

/**
 * Compute dPtdAb using a WENO method.
 */
void dPtdAbstg_WENO(real_t *agg, real_t *dPtdAb, real_t rho_b)
{
    int i=0;

    real_t *Ab = agg + 0;
    real_t *rhot = agg + 7*NN;

    ALLOC_N( Ab_stg, NN )
    WENO_stg( Ab, Ab_stg );

    ALLOC_N( rhot_stg, NN )
    WENO_stg( rhot, rhot_stg );
    
    ALLOC_N( Pt_stg, NN )
    for(i=0; i<NN; i++)
        Pt_stg[i] = Pt_of_rhot(rhot_stg[i], rho_b);

    for(i=1; i<NN; i++)
        dPtdAb[i] = (Pt_stg[i] - Pt_stg[i-1])/(Ab_stg[i] - Ab_stg[i-1]);
    dPtdAb[0] = 0;

    free(Ab_stg);
    free(rhot_stg);
    free(Pt_stg);
}

/**
 * Smooth a field using nearby WENO-weighted values
 * assume an unchanged value at the origin
 */
void smooth(real_t *f)
{
    int i=0;

    ALLOC_N( f_stg, NN )
    WENO_stg( f, f_stg );

#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN; i++)
    {
        f[i] = (f_stg[i]+f_stg[i-1])/2.0;
    }

    free(f_stg);
}


/**
 * Populate the aggregate list of values ("agg"), structured as NFIELDS
 * concatenated arrays of length NN. Assumes The first 4 arrays already
 * contain Abar, Rtilde, mtilde, Utilde, and populate the remaining
 * NFIELDS - 4 arrays (detailed in code below).
 */
extern "C"
void agg_pop(real_t * agg, real_t l)
{
    int i=0;
    real_t e2l = std::exp(2*l);
    real_t rho_b = rho_background(l);

    // Populate all information; stored in "agg"regate array
    // Ab, Rt, mt, Ut are expected to be filled already.
    real_t *Ab     = agg + 0;
    real_t *Rt     = agg + 1*NN;
    real_t *mt     = agg + 2*NN;
    real_t *Ut     = agg + 3*NN;
    // The remainder are computed.
    real_t *dRtdAb = agg + 4*NN;
    real_t *dmtdAb = agg + 5*NN;
    real_t *gammab = agg + 6*NN;
    real_t *rhot   = agg + 7*NN;
    real_t *Pt     = agg + 8*NN;
    real_t *dPtdAb = agg + 9*NN;
    real_t *phi    = agg + 10*NN;
    real_t *m2oR   = agg + 11*NN;
    real_t *Qt     = agg + 12*NN;

    // Derivatives of R and m wrt. Ab, used later
    if(USE_WENO)
    {
        dfdA_WENO(Rt, Ab, dRtdAb);
        dfdA_WENO(mt, Ab, dmtdAb);
    }
    else
    {
        dfdA(Rt, Ab, dRtdAb);
        dfdA(mt, Ab, dmtdAb);
    }

    // compute gamma and 2*m/R, used later
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        real_t AbRt2 = Ab[i]*Rt[i]*Ab[i]*Rt[i];
        gammab[i] = std::sqrt( RHO_B_I/rho_b/e2l + AbRt2*(Ut[i]*Ut[i] - mt[i]) );
        m2oR[i] = e2l*rho_b/RHO_B_I*AbRt2*mt[i];
    }

    // Compute density (rho tilde)
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN-1; i++)
    {
        rhot[i] = mt[i] + Ab[i]*Rt[i]/3
                    *(mt[i+1] - mt[i-1]) / (Ab[i+1]*Rt[i+1] - Ab[i-1]*Rt[i-1]);
    }
    rhot[0] = mt[0];
    rhot[NN-1] = mt[NN-1] + Ab[NN-1]*Rt[NN-1]*(mt[NN-1] - mt[NN-2])
                / 3 / (Ab[NN-1]*Rt[NN-1] - Ab[NN-2]*Rt[NN-2]);
    if(SMOOTH_RHO)
    {
        smooth(rhot);
    }

    // Compute pressure (P tilde)
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
        Pt[i] = Pt_of_rhot(rhot[i], rho_b);

    // Compute derivative of pressure
    if(USE_WENO)
    {
        // This doesn't seem to work well...
        // dPtdAbstg_WENO(agg, dPtdAb, rho_b);
        dPtdAbstg(agg, dPtdAb, rho_b);
    }
    else
    {
        dPtdAbstg(agg, dPtdAb, rho_b);
    }

    // Compute (artificial) viscosity
    real_t kappa = 4.0;
    if(kappa > 0)
    {
#pragma omp parallel for default(shared) private(i)
        for(i=1; i<NN-1; i++)
        {
            real_t d2Ab = Ab[i+1] - Ab[i-1];
            real_t Utp = (Ut[i+1] - Ut[i-1]) / d2Ab;
            real_t AbRtp = (Ab[i+1]*Rt[i+1] - Ab[i-1]*Rt[i-1]) / d2Ab;
            real_t AbRtUtp = (Ab[i+1]*Rt[i+1]*Ut[i+1] - Ab[i-1]*Rt[i-1]*Ut[i-1]) / d2Ab;
            if(AbRtp*Ut[i] < -Ab[i]*Rt[i]*Utp)
                Qt[i] = kappa*(d2Ab/2)*(d2Ab/2)/e2l*AbRtUtp*AbRtUtp;
            else
                Qt[i] = 0;
        }
        Qt[0] = Qt[1];
        Qt[NN-1] = Qt[NN-2];

        // Smooth the viscosity curve a bit
        Qt[0] = (4*Qt[0] + 3*Qt[1] + 2*Qt[2] + Qt[3])/10.0;
        Qt[1] = (3*Qt[0] + 4*Qt[1] + 3*Qt[2] + 2*Qt[3] + Qt[4])/13.0;
        Qt[2] = (2*Qt[0] + 3*Qt[1] + 4*Qt[2] + 3*Qt[3] + 2*Qt[4] + Qt[5])/15.0;
        for(i=3; i<NN-3; i++)
            Qt[i] = (Qt[i-3]+2*Qt[i-2]+3*Qt[i-1]+4*Qt[i]+3*Qt[i+1]+2*Qt[i+2]+1*Qt[i+3])/16.0;

#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
            Pt[i] += Qt[i];
    }

    // Perform integration to compute metric (phi). The boundary value can be
    // chosen; shifting it by a constant amounts to re-scaling the time
    // coordinate. We use Eq. 51, Assuming radiation-dominated cosmology values
    // for the equation of state. Close to zero works, anyways.
    phi[NN-1] = -std::log(rhot[NN-1])/2.0;
    phi[NN-2] = phi[NN-1] + (Pt[NN-1] - Pt[NN-2])*2.0/(Pt[NN-2] + Pt[NN-1] + rhot[NN-2] + rhot[NN-1]);
    for(i=NN-3; i>=0; i--)
        phi[i] = (Pt[i+2] - Pt[i])/(Pt[i+1] + rhot[i+1]) + phi[i+2];
}

/**
 *  Compute Runge-Kutta 'k' coefficients
 *  kR_f = F( agg + deltal*kcoeff*k_p )
 */
void k_calc(real_t *agg, real_t *tmp_agg,
    real_t *kR_p, real_t *kR_f,
    real_t *km_p, real_t *km_f,
    real_t *kU_p, real_t *kU_f,
    real_t l, real_t deltal, real_t kcoeff)
{
    int i=0;

    // Populate temporary agg array with computed values
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        tmp_agg[i + 0*NN] = agg[i + 0*NN];
        tmp_agg[i + 1*NN] = agg[i + 1*NN] + kR_p[i]*deltal*kcoeff;
        tmp_agg[i + 2*NN] = agg[i + 2*NN] + km_p[i]*deltal*kcoeff;
        tmp_agg[i + 3*NN] = agg[i + 3*NN] + kU_p[i]*deltal*kcoeff;
    }
    l = l + deltal*kcoeff;
    agg_pop(tmp_agg, l);

    real_t *Ab     = tmp_agg + 0;
    real_t *Rt     = tmp_agg + 1*NN;
    real_t *mt     = tmp_agg + 2*NN;
    real_t *Ut     = tmp_agg + 3*NN;
    real_t *dRtdAb = tmp_agg + 4*NN;
    real_t *dmtdAb = tmp_agg + 5*NN;
    real_t *gammab = tmp_agg + 6*NN;
    real_t *rhot   = tmp_agg + 7*NN;
    real_t *Pt     = tmp_agg + 8*NN;
    real_t *dPtdAb = tmp_agg + 9*NN;
    real_t *phi    = tmp_agg + 10*NN;
    real_t *Qt     = tmp_agg + 12*NN;

    real_t rho_b = rho_background(l);
    real_t alpha = 2.0/3.0*rho_b/( rho_b + P_background(l) );

    // fill _f registers with final values
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        real_t ep = std::exp(phi[i]);
        real_t AbRt_prime = Rt[i] + Ab[i]*dRtdAb[i];

        real_t dPtdAboAb = 0;
        if(i == 0)
            dPtdAboAb = 5.0/3.0*dPdrho_rhot(rhot[0], rho_b)
                *(mt[1] + mt[1] - 2*mt[0]) / Ab[1] / Ab[1];
        else
            dPtdAboAb = dPtdAb[i] / Ab[i];

        // equations of motion
        kR_f[i] = Rt[i]*(Ut[i]*ep - 1);
        km_f[i] = 2.0/alpha*mt[i] - 3*Ut[i]*ep*(Pt[i] + mt[i]);
        kU_f[i] = Ut[i]/alpha - ep*(
            gammab[i]*gammab[i]*dPtdAboAb / (Rt[i]*AbRt_prime*(rhot[i] + Pt[i]))
            + (2*Ut[i]*Ut[i] + mt[i] + 3*Pt[i])/2.0
        );
    }

}

/**
 * Change Abar coordinates
 */
extern "C"
void regrid(real_t *agg, real_t nu)
{
    int i=0;

    real_t *Ab = agg + 0;
    real_t *Rt = agg + 1*NN;
    real_t *mt = agg + 2*NN;
    real_t *Ut = agg + 3*NN;
    real_t *dmtdAbi = agg + 5*NN;
    ALLOC(new_Ab)
    ALLOC(dA)

    // Want lots of resolution where mass changes rapidly
    dfdA_WENO(mt, Ab, dmtdAbi);
    smooth(dmtdAbi);
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN; i++)
    {
        dA[i] = Ab[i]-Ab[i-1];
        dmtdAbi[i] = 1.0/(std::abs(dmtdAbi[i])+1.0e-2);
    }
    dmtdAbi[0] = dmtdAbi[1];
    dA[0] = 0;

    // relax dA towards 1/dmtdAb
    real_t max_dmtdAbi = max(dmtdAbi);
    real_t max_dA = max(dA);
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN; i++)
        dA[i] = 0.03 + dA[i]/max_dA + nu*dmtdAbi[i]/max_dmtdAbi;
    smooth(dA);

    new_Ab[0] = 0;
    for(i=1; i<NN; i++)
        new_Ab[i] = new_Ab[i-1] + dA[i];

    real_t max_Ab = max(Ab);
    real_t max_new_Ab = max(new_Ab);
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN; i++)
        new_Ab[i] = new_Ab[i]*max_Ab/max_new_Ab;


    // interpolate fields at new Ab points
    std::vector<real_t> Ab_v ( NN+3.0 );
    std::vector<real_t> Rt_v ( NN+3.0 );
    std::vector<real_t> mt_v ( NN+3.0 );
    std::vector<real_t> Ut_v ( NN+3.0 );
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        Ab_v[i+3] = Ab[i];
        Rt_v[i+3] = Rt[i];
        mt_v[i+3] = mt[i];
        Ut_v[i+3] = Ut[i];
    }
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<3; i++)
    {
        Ab_v[i] = -Ab[3-i];
        Rt_v[i] = Rt[3-i];
        mt_v[i] = mt[3-i];
        Ut_v[i] = Ut[3-i];
    }

    tk::spline RtofAb(Ab_v, Rt_v);
    tk::spline mtofAb(Ab_v, mt_v);
    tk::spline UtofAb(Ab_v, Ut_v);

#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        Ab[i] = new_Ab[i];
        Rt[i] = RtofAb(new_Ab[i]);
        mt[i] = mtofAb(new_Ab[i]);
        Ut[i] = UtofAb(new_Ab[i]);
    }

    free(new_Ab);
    free(dA);
}

/**
 * Output agg array to a file
 */
void write_output(real_t *agg, real_t l, std::ofstream & output)
{
    agg_pop(agg, l);
    output.write((char *) agg, NFIELDS*NN*sizeof(real_t));
}

/**
 * Run a simulation --- given an initialized agg array and a
 * starting l, run for a number of steps, outputting per the interval.
 */
extern "C"
int run_sim(real_t *agg, real_t &l, real_t &deltaH, int steps, int output_interval,
    bool stop_on_horizon, real_t q_mult, bool smooth_rho, bool use_weno, real_t regrid_nu)
{
    // globals
    SMOOTH_RHO = smooth_rho;
    USE_WENO = use_weno;
    // variables for interpolating delta @ horizon crossing
    real_t rhot_horizon = 2.0, old_rhot_horizon = 2.0;
    real_t delta_horizon = 0.0, old_delta_horizon = 0.0;
    // return value flag
    int flag = 0;

    std::cout << "\n============\nRunning sim.\n============\n" << std::flush;
    std::cout << "\n";
    std::cout << "Using "<<steps<<" steps with "<<NN<<" gridpoints. Output every "
        <<output_interval<<" steps.\n";
    std::cout << " USE_FIXW is "<<USE_FIXW<<", SMOOTH_RHO is "<<SMOOTH_RHO<<", USE_WENO is "
        <<USE_WENO<<".\n";
    std::cout << "\n" << std::flush;

    real_t *Ab        = agg + 0;
    real_t *Rt        = agg + 1*NN;
    real_t *mt        = agg + 2*NN;
    real_t *Ut        = agg + 3*NN;
    ALLOC_N(tmp_agg, NFIELDS*NN)
    ALLOC(zeros)
    ALLOC(kR1) ALLOC(kR2) ALLOC(kR3) ALLOC(kR4) ALLOC(kRi)
    ALLOC(km1) ALLOC(km2) ALLOC(km3) ALLOC(km4) ALLOC(kmi)
    ALLOC(kU1) ALLOC(kU2) ALLOC(kU3) ALLOC(kU4) ALLOC(kUi)

    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
        zeros[i] = 0.0;

    // main integration loop
    std::ofstream output;
    output.open("output.dat", std::ios::out | std::ios::binary | std::ios::trunc);
    int s=0;
    real_t deltal = 3.67e-7;
    real_t max_rho0 = 0.0, prev_rho0 = 0.0, rho0 = 0.0;
    for(s=0; s<=steps; s++)
    {
        // Upkeep
        #pragma omp critical
        if(s%output_interval==0 or s==steps)
            if(output_interval > 0)
                write_output(agg, l, output);

        // Integration details
        k_calc(agg, tmp_agg,  zeros, kR1,  zeros, km1,  zeros, kU1,  l, deltal,  0.0);
        k_calc(agg, tmp_agg,  kR1, kR2,    km1, km2,    kU1, kU2,    l, deltal,  0.5);
        k_calc(agg, tmp_agg,  kR2, kR3,    km2, km3,    kU2, kU3,    l, deltal,  0.75);
#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
        {
            kRi[i] = ( 2*kR1[i] + 3*kR2[i] + 4*kR3[i] )/9.0;
            kmi[i] = ( 2*km1[i] + 3*km2[i] + 4*km3[i] )/9.0;
            kUi[i] = ( 2*kU1[i] + 3*kU2[i] + 4*kU3[i] )/9.0;
        }
        k_calc(agg, tmp_agg,  kRi, kR4,  kmi, km4,  kUi, kU4,  l, deltal,  1.0);

        // Determine maximum errors
        real_t E_R_max = 0, E_m_max = 0, E_U_max = 0,
               tol_R_max = 0, tol_m_max = 0, tol_U_max = 0;
        for(i=0; i<NN; i++)
        {
            real_t E_R = deltal*fabs(-5.0*kR1[i]/72.0 + kR2[i]/12.0 + kR3[i]/9.0 - kR4[i]/8.0);
            if(E_R > E_R_max) { E_R_max = E_R; }
            real_t E_m = deltal*fabs(-5.0*km1[i]/72.0 + km2[i]/12.0 + km3[i]/9.0 - km4[i]/8.0);
            if(E_m > E_m_max) { E_m_max = E_m; }
            real_t E_U = deltal*fabs(-5.0*kU1[i]/72.0 + kU2[i]/12.0 + kU3[i]/9.0 - kU4[i]/8.0);
            if(E_U > E_U_max) { E_U_max = E_U; }

            real_t tol_R = fabs(Rt[i])*TOL;
            if(tol_R > tol_R_max) { tol_R_max = tol_R; }
            real_t tol_m = fabs(mt[i])*TOL;
            if(tol_m > tol_m_max) { tol_m_max = tol_m; }
            real_t tol_U = fabs(Ut[i])*TOL;
            if(tol_U > tol_U_max) { tol_U_max = tol_U; }
        }

        // final field values at the end of the integration step.
        if(E_R_max < tol_R_max && E_m_max < tol_m_max && E_U_max < tol_U_max)
        {
            for(i=0; i<NN; i++)
            {
                Rt[i] = Rt[i] + deltal/9*(2*kR1[i] + 3*kR2[i] + 4*kR3[i] );
                mt[i] = mt[i] + deltal/9*(2*km1[i] + 3*km2[i] + 4*km3[i] );
                Ut[i] = Ut[i] + deltal/9*(2*kU1[i] + 3*kU2[i] + 4*kU3[i] );
            }
            l = l + deltal;
        }
        agg_pop(agg, l);


        // adjust step size for next step
        real_t q = -1.0;
        if(!(E_R_max == 0 || E_m_max == 0 || E_U_max == 0))
        {
            q = q_mult*std::pow( std::min(tol_R_max/E_R_max,
                std::min(tol_m_max/E_m_max, tol_U_max/E_U_max) ), 1.0/3.0);
            q = std::min(q, (real_t) 1.5); // limit stepsize growth
            deltal *= q;
        }

        // Flag when the density perturbation enters the cosmic horizon
        // (linear) interpolation of delta in both time and space directions
        real_t *Ab     = tmp_agg + 0;
        real_t *Rt     = tmp_agg + 1*NN;
        real_t *mt     = tmp_agg + 2*NN;
        real_t *rhot = agg + 7*NN;
        if(deltaH < 0)
        {
            real_t rho_b = rho_background(l);
            real_t AbRt_horizon = std::exp(-l)*std::sqrt( RHO_B_I/rho_b );

            for(i=0; i<NN - 1; i++)
            {
                real_t AbRt1 = Ab[i]*Rt[i];
                real_t AbRt2 = Ab[i+1]*Rt[i+1];
                bool at_horizon = AbRt1 < AbRt_horizon && AbRt2 > AbRt_horizon;

                if( at_horizon )
                {
                    old_rhot_horizon = rhot_horizon;
                    rhot_horizon = (AbRt_horizon - AbRt1)/(AbRt2 - AbRt1)*(rhot[i+1] - rhot[i]) + rhot[i];
                    old_delta_horizon = delta_horizon;
                    delta_horizon = (AbRt_horizon - AbRt1)/(AbRt2 - AbRt1)*(mt[i+1] - mt[i]) + mt[i] - 1;

                    if(old_rhot_horizon >= 1 && rhot_horizon < 1)
                    {
                        deltaH = old_delta_horizon
                            + (1.0 - old_rhot_horizon)/(rhot_horizon - old_rhot_horizon)*(delta_horizon - old_delta_horizon);
                        std::cout << "Density perturbation at cosmological horizon crossing found: \n  deltaH=" << deltaH
                            << " near l=" << l << ", rhot in (" << rhot_horizon << "," << old_rhot_horizon << "),"
                            << " mt in (" << mt[i] << "," << mt[i+1] << "), AbRt in (" << AbRt1 << "," << AbRt2 << "), "
                            << "and AbRt_horizon=" << AbRt_horizon << "\n";
                    }
                    break;
                }
            }
        }

        // Check for diverging rho (singularity forming), or shrinking rho (homogenizing)
        prev_rho0 = rho0;
        rho0 = rhot[3];
        if(rho0 > max_rho0)
            max_rho0 = rho0;
        if(rho0 > 1.0e12)
        {
            // Assume singularity is forming
            flag = 1;
            std::cout << "Density becoming singular at step "<<s<<". q was "<<q<<"\n";
            break;
        }
        if(rho0 < 0.25*max_rho0)
        {
            // Assume no BH is forming
            flag = 2;
            std::cout << "Density dropping substantially at step "<<s<<". q was "<<q<<"\n";
            break;
        }
        real_t *m2oR = agg + 11*NN;
        if( stop_on_horizon )
        {
            for(i=0; i<NN-1; i++)
            {
                if(m2oR[i]>1 && m2oR[i+1]<1)
                {
                    flag = 3;
                    std::cout << "Horizon formed at step "<<s<<". q was "<<q<<"\n";
                    break;
                }
            }
            if(flag == 3) break;
        }

        // Error checking
        bool hasnan = false;
        for(i=0; i<NN; i++)
            if(std::isnan(Rt[i]) || fabs(Rt[i]) > 1.0e10)
            {
                if(hasnan == false)
                    std::cout << "NaN detected.\n" << std::flush;
                hasnan = true;
            }
        if(deltal < 1.0e-12 || deltal > 1.0 || hasnan)
        {
            std::cout << "Error at step "<<s<<". q was "<<q<<", l was "<<l<<"\n";
            std::cout << "Tolerances were "<<tol_R_max<<", "<<tol_m_max<<", "<<tol_U_max<<"\n";
            std::cout << "Errors were     "<<E_R_max<<", "<<E_m_max<<", "<<E_U_max<<"\n";
            std::cout << "deltal was "<<deltal<<"\n";
            if(output_interval > 0)
                write_output(agg, l, output);
            flag = -1;
            break;
        }

        if(regrid_nu > 0)
        {
            regrid(agg, regrid_nu);
        }

    }

    std::cout << "\nFinal log(a) after step "<<s<<" was "<< l <<", delta log(a) was "<<deltal<<".\n";
    std::cout << "\n============\nDone running.\n============\n";

    output.close();

    agg_pop(agg, l);

    free(zeros);
    free(kR1); free(kR2); free(kR3); free(kR4); free(kRi);
    free(km1); free(km2); free(km3); free(km4); free(kmi);
    free(kU1); free(kU2); free(kU3); free(kU4); free(kUi);

    return flag;
}

/**
 * Set initial conditions. Valid only assuming a constant equation of state,
 * i.e. fixed. w. For radiation, this means significantly before the QCD
 * transition.
 * 
 * amp is the perturbation amplitude,
 * d is the perturbation size in horizon units.
 */
extern "C"
void ics(real_t *agg, real_t &l, real_t &deltaH, real_t amp, real_t d, int n,
    bool use_fixw)
{
    USE_FIXW = use_fixw;

    NN = n;
    l = 0;
    deltaH = -1.0;

    real_t *Ab      = agg + 0;
    real_t *Rt        = agg + 1*NN;
    real_t *mt        = agg + 2*NN;
    real_t *Ut        = agg + 3*NN;

    std::cout << "\n==================\nInitializing sim.\n==================\n" << std::flush;
    std::cout << "\n";
    std::cout << "Using "<<NN<<" gridpoints, amplitude "
                    <<amp<<", perturbation size "<<d<<".\n";
    std::cout << std::flush;


    real_t rho_b = rho_background(l);
    real_t alpha = 2.0/3.0*rho_b/( rho_b + P_background(l) ); // Eq. 36, sort of
    
    real_t L = 20.0*d; // Simulation size

    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        Ab[i] = i*L/NN;
        real_t deltam0 = amp*std::exp(-Ab[i]*Ab[i]/2/d/d);
        real_t deltam0P = -Ab[i]/d/d*deltam0;

        Rt[i] = 1 - alpha/2*(deltam0 + Ab[i]*deltam0P/6.0 );
        mt[i] = 1 + deltam0;
        Ut[i] = 1 - alpha*deltam0/2;
    }
}


int main(int argc, char **argv)
{

    int steps = 1000, output_interval=1000;
    real_t amp = 0.3, d = 1.6;

    static struct option long_options[] =
    {
        {"steps",  required_argument, 0, 's'},
        {"d",      required_argument, 0, 'd'},
        {"amp",    required_argument, 0, 'a'},
        {"N",      required_argument, 0, 'N'},
        {"output", required_argument, 0, 'o'},
        {"help",   no_argument,       0, 'h'}
    };

    int c = 0;
    while(1)
    {
        int option_index = 0;
        c = getopt_long(argc, argv, "s:r:a:N:o:h", long_options, &option_index);
        if(c == -1) // stop if done reading arguments
            break;

        switch(c)
        {
            case 's':
                steps = std::stoi(optarg);
                break;
            case 'd':
                d = (real_t) std::stod(optarg);
                break;
            case 'a':
                amp = (real_t) std::stod(optarg);
                break;
            case 'N':
                NN = std::stoi(optarg);
                break;
            case 'o':
                output_interval = std::stoi(optarg);
                break;
            case 'h':
            case '?':
                fprintf(stdout, "\nusage: %s -N [gridpoints] -s [steps] -d [size] -a [amp] -o [output_interval] \n", argv[0]);
                fprintf(stdout, "All options are optional; if not specified, defaults will be used.\n");
                return 0;
            default:
                fprintf(stdout, "Unrecognized option.\n");
                return 0;
        }
    }
    if(!output_interval) output_interval = steps;

    ALLOC_N(agg, NFIELDS*NN)

    real_t l, deltaH;
    ics(agg, l, deltaH, amp, d, NN, USE_FIXW);
    run_sim(agg, l, deltaH, steps, output_interval, false, 0.25,
        SMOOTH_RHO, USE_WENO, 0.0);

    free(agg);

    return 0;
}
