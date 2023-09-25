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
#define NFIELDS 13


int NN = 3200;
bool USE_WENO = true;
bool SMOOTH_RHO = true;
bool USE_FIXW = false;
real_t FIXW = 1.0/3.0;
real_t RHO_B_I = 1.0e27;

#define ONLY_ALLOC_N(arr_n, ARR_SIZE) \
    arr_n = (real_t *) malloc( (ARR_SIZE) * ((long long) sizeof(real_t)));

#define ALLOC_N(arr_n, ARR_SIZE) \
    real_t *arr_n; ONLY_ALLOC_N(arr_n, ARR_SIZE)

#define ONLY_ALLOC(arr_n) \
    ONLY_ALLOC_N(arr_n, NN)

#define ALLOC(arr_n) \
    ALLOC_N(arr_n, NN)

struct kCoefficientData {
  real_t *kR1, *kR2, *kR3, *kR4, *kR5, *kR6;
  real_t *km1, *km2, *km3, *km4, *km5, *km6;
  real_t *kU1, *kU2, *kU3, *kU4, *kU5, *kU6;
  real_t *agg_f;
};

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

real_t min(real_t *f)
{
    real_t fmin = f[0];
#pragma omp parallel for reduction(min:fmin)
    for(int i=0; i<NN; i++)
       fmin = fmin < f[i] ? fmin : f[i];
    return fmin;
}

real_t sum(real_t *f)
{
    real_t sum = 0;
#pragma omp parallel for reduction (+:sum)
    for(int i=0; i<NN; i++)
        sum = sum + f[i];
    return sum;
}

real_t mean(real_t *f)
{
    return sum(f)/NN;
}

real_t L1(real_t *f)
{
    real_t sum = 0;
#pragma omp parallel for reduction (+:sum)
    for(int i=0; i<NN; i++)
        sum = sum + std::fabs(f[i]);
    return sum;
}

real_t mean_L1(real_t *f)
{
    return L1(f)/NN;
}

// Background density as a function of ell
class RhoBIntegrator
{
public:
    real_t dl = 0.002538901;
    std::vector<real_t> ls;
    std::vector<real_t> logrho_of_ls;

    tk::spline spl;

    RhoBIntegrator(real_t rho_b_i, real_t l)
    {
        ls.push_back(l);
        logrho_of_ls.push_back(std::log(rho_b_i));

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
RhoBIntegrator logrhob_of_l = RhoBIntegrator(RHO_B_I, 0.0);

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
 * Linear growth function.
 * \delta_m(l) = G(l) * \delta_{m,0}
 */
extern "C"
real_t G(real_t l)
{
    real_t rho_b = rho_background(l);
    return std::exp(-2.0*l)*RHO_B_I/rho_b;
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
 * Second derivatives of a function.
 * Assume symmetry at the origin, zero at the outer boundary.
 */
void d2fdA2(real_t *f, real_t *A, real_t *d2fdA2)
{
    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN-1; i++)
    {
        d2fdA2[i] = (
            (f[i+1]-f[i]) / (A[i+1]-A[i])
            - (f[i]-f[i-1]) / (A[i]-A[i-1])
        )/( ( A[i+1] - A[i-1] )/2 );
    }

    d2fdA2[0] = 2*(f[1]-f[0])/A[1]/A[1];
    d2fdA2[NN-1] = 0.0;
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
void ENOify(real_t *f)
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
 * Smooth a field using nearby WENO-weighted values
 * assume an unchanged value at the origin
 */
void smooth(real_t *f)
{
    int i=0;

    ALLOC( f_new )

#pragma omp parallel for default(shared) private(i)
    for(i=2; i<NN-2; i++)
        f_new[i] = (f[i-2]+4*f[i-1]+6*f[i]+4*f[i+1]+f[i+2])/16.0;
    f_new[0] = (6*f[0]+4*f[1]+f[2])/11.0;
    f_new[NN-1] = (f[NN-3]+4*f[NN-2]+6*f[NN-1])/11.0;
    f_new[1] = (4*f[0]+6*f[1]+4*f[2]+f[3])/15.0;
    f_new[NN-2] = (f[NN-4]+4*f[NN-3]+6*f[NN-2]+4*f[NN-1])/15.0;
    
    // ENOify(f_new);

#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
        f[i] = f_new[i];

    free(f_new);
}

/**
 * Change Abar coordinates
 */
extern "C"
void regrid(real_t *agg, real_t lam, real_t nu)
{
    int i=0;

    real_t *Ab = agg + 0;
    real_t *Rt = agg + 1*NN;
    real_t *mt = agg + 2*NN;
    real_t *Ut = agg + 3*NN;

    ALLOC(new_Ab)
    ALLOC(cum_I_sigma) // field to try to sample regularly
    // use abs(d2mt/dAb2) + lam for sigma
    ALLOC(d2mtdAb2)
    d2fdA2(mt, Ab, d2mtdAb2);
    smooth(d2mtdAb2);
    cum_I_sigma[0] = 0;
    for(i=1; i<NN; i++)
    {
        // trapezoidal integration rule
        real_t I_sigma_i = ( std::tanh(std::fabs(d2mtdAb2[i])/100)+std::tanh(std::fabs(d2mtdAb2[i-1])/100) )/2 + lam;
        real_t dA = Ab[i] - Ab[i-1];
        cum_I_sigma[i] = cum_I_sigma[i-1] + I_sigma_i*dA;
    }

    // Spline to get Ab at even ds intervals
    std::vector<real_t> Ab_s ( Ab, Ab+NN );
    std::vector<real_t> cs_s ( cum_I_sigma, cum_I_sigma+NN );
    tk::spline Ab_at_cum_I_sigma( cs_s, Ab_s, tk::spline::cspline, true );
    // ds intervals are total sigma subdivided into NN-1 bins (def'd by NN points)
    real_t ds = cum_I_sigma[NN-1]/(NN-1);
    new_Ab[0] = 0;
#pragma omp parallel for default(shared) private(i)    
    for(i=1; i<NN; i++)
        new_Ab[i] = Ab_at_cum_I_sigma(i*ds);
#pragma omp parallel for default(shared) private(i)    
    for(i=0; i<NN; i++)
        new_Ab[i] = (1.0-nu)*Ab[i] + nu*new_Ab[i];

    // interpolating functions for fields
    std::vector<real_t> Ab_v ( NN+4.0 );
    std::vector<real_t> Rt_v ( NN+4.0 );
    std::vector<real_t> mt_v ( NN+4.0 );
    std::vector<real_t> Ut_v ( NN+4.0 );
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
    Ab_v[NN+3] = Ab[NN-1]+(Ab[NN-1]-Ab[NN-2]);
    Rt_v[NN+3] = Rt[NN-1];
    mt_v[NN+3] = mt[NN-1];
    Ut_v[NN+3] = Ut[NN-1];

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
    free(d2mtdAb2);
    free(cum_I_sigma);
}

/**
 * Populate the aggregate list of values ("agg"), structured as NFIELDS
 * concatenated arrays of length NN. Assumes The first 4 arrays already
 * contain Abar, Rtilde, mtilde, Utilde, and populate the remaining
 * NFIELDS - 4 arrays (detailed in code below).
 */
extern "C"
void agg_pop(real_t *agg, real_t l)
{
    int i=0;
    real_t e2l = std::exp(2*l);
    real_t G_l = G(l);
    real_t rho_b = rho_background(l);
    real_t P_b = P_background(l);

    // Populate all information; stored in "agg"regate array
    // Ab, Rt, mt, Ut are expected to be filled already.
    real_t *Ab      = agg + 0;
    real_t *Rt      = agg + 1*NN;
    real_t *mt      = agg + 2*NN;
    real_t *Ut      = agg + 3*NN;
    // The remainder are computed.
    real_t *dRtdAb  = agg + 4*NN;
    real_t *dmtdAb  = agg + 5*NN;
    real_t *gammab2 = agg + 6*NN;
    real_t *rhot    = agg + 7*NN;
    real_t *Pt      = agg + 8*NN;
    real_t *dPtdAb  = agg + 9*NN;
    real_t *phi     = agg + 10*NN;
    real_t *m2oR    = agg + 11*NN;
    real_t *Qt      = agg + 12*NN;

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
        gammab2[i] = G_l + AbRt2*(Ut[i]*Ut[i] - mt[i]);
        m2oR[i] = 1.0/G_l*AbRt2*mt[i];
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

    // Compute (artificial) viscosity
    real_t kappa = 2.0;
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
        smooth(Qt);

#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
            Pt[i] += Qt[i];
    }

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

    // Perform integration to compute metric (phi). The boundary value can be
    // chosen; shifting it by a constant amounts to re-scaling the time
    // coordinate. We use Eq. 51, Assuming radiation-dominated cosmology values
    // for the equation of state. Close to zero works, anyways.
    // phi[NN-1] = -std::log(rhot[NN-1])/2.0;
    phi[NN-1] = -std::log( (1 + Pt[NN-1]) / (1 + P_b/rho_b) );
    phi[NN-2] = phi[NN-1] + (Pt[NN-1] - Pt[NN-2])*2.0/(Pt[NN-2] + Pt[NN-1] + rhot[NN-2] + rhot[NN-1]);
    for(i=NN-3; i>=0; i--)
        phi[i] = (Pt[i+2] - Pt[i])/(Pt[i+1] + rhot[i+1]) + phi[i+2];
}

/**
 *  Compute Runge-Kutta 'k' coefficients
 */
void k_calc(real_t *agg, kCoefficientData *ks,  real_t l_in, real_t deltal, real_t A,
    real_t *kR_f, real_t *km_f, real_t *kU_f,
    real_t B1, real_t B2, real_t B3, real_t B4, real_t B5)
{
    int i=0;
    real_t l = l_in + A*deltal;

    // Populate temporary agg array with computed values
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        ks->agg_f[i + 0*NN] = agg[i + 0*NN]; // coordinates don't change
        ks->agg_f[i + 1*NN] = agg[i + 1*NN] + B1*ks->kR1[i] + B2*ks->kR2[i] + B3*ks->kR3[i] + B4*ks->kR4[i] + B5*ks->kR5[i];
        ks->agg_f[i + 2*NN] = agg[i + 2*NN] + B1*ks->km1[i] + B2*ks->km2[i] + B3*ks->km3[i] + B4*ks->km4[i] + B5*ks->km5[i];
        ks->agg_f[i + 3*NN] = agg[i + 3*NN] + B1*ks->kU1[i] + B2*ks->kU2[i] + B3*ks->kU3[i] + B4*ks->kU4[i] + B5*ks->kU5[i];
    }
    agg_pop(ks->agg_f, l);

    real_t *Ab      = ks->agg_f + 0;
    real_t *Rt      = ks->agg_f + 1*NN;
    real_t *mt      = ks->agg_f + 2*NN;
    real_t *Ut      = ks->agg_f + 3*NN;
    real_t *dRtdAb  = ks->agg_f + 4*NN;
    real_t *dmtdAb  = ks->agg_f + 5*NN;
    real_t *gammab2 = ks->agg_f + 6*NN;
    real_t *rhot    = ks->agg_f + 7*NN;
    real_t *Pt      = ks->agg_f + 8*NN;
    real_t *dPtdAb  = ks->agg_f + 9*NN;
    real_t *phi     = ks->agg_f + 10*NN;
    real_t *Qt      = ks->agg_f + 12*NN;

    real_t rho_b = rho_background(l);
    real_t alpha = 2.0/3.0*rho_b/( rho_b + P_background(l) );

    // fill _f register with computed values
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
        kR_f[i] = deltal*Rt[i]*(Ut[i]*ep - 1);
        km_f[i] = deltal*(2.0/alpha*mt[i] - 3*Ut[i]*ep*(Pt[i] + mt[i]));
        kU_f[i] = deltal*(Ut[i]/alpha - ep*(
            gammab2[i]*dPtdAboAb / (Rt[i]*AbRt_prime*(rhot[i] + Pt[i]))
            + (2*Ut[i]*Ut[i] + mt[i] + 3*Pt[i])/2.0
        ));
    }
}

/**
 * Compute the black hole mass at a specific index.
 * Mass is in units of R_H^I.
 */
real_t mass_at(real_t *agg, real_t l, int i)
{
    real_t *Ab = agg + 0;
    real_t *Rt = agg + 1*NN;
    real_t *mt = agg + 2*NN;
    return std::exp(l)/2.0/G(l)*std::pow(Ab[i]*Rt[i],3)*mt[i];
}

/**
 * Return Misher-Sharp mass
 * (BH mass at outermost trapped surface).
 */
real_t ms_mass(real_t *agg, real_t l)
{
    int i=0;
    real_t *m2oR = agg + 11*NN;

    for(i=0; i<NN-1; i++)
    {
        if(m2oR[i] > 1 && m2oR[i+1] <= 1)
        {
            real_t m2oR_frac = (1 - m2oR[i])/(m2oR[i+1] - m2oR[i]);
            return m2oR_frac*(mass_at(agg, l, i+1) - mass_at(agg, l, i)) + mass_at(agg, l, i);
        }
    }

    return 0;
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
int run_sim(real_t *agg, real_t &l, real_t &deltaH, real_t &max_rho0, real_t &bh_mass,
    int steps, int output_interval,
    bool stop_on_horizon, real_t q_mult, bool smooth_rho, bool use_weno,
    int regrid_interval, real_t regrid_lam, real_t regrid_nu, real_t TOL)
{
    // globals
    SMOOTH_RHO = smooth_rho;
    USE_WENO = use_weno;
    // variables for interpolating delta @ horizon crossing
    real_t rhot_horizon = 2.0, old_rhot_horizon = 2.0;
    real_t delta_horizon = 0.0, old_delta_horizon = 0.0;
    real_t m_horizon = 0.0, old_m_horizon = 0.0;
    // Tracking density and BH formation
    real_t prev_rho0 = 0.0, rho0 = 0.0;
    real_t prev_bh_mass = 0.0;
    // return value flag
    int flag = 0;
    // Integration tolerances
    real_t deltal = 1.5e-4, q = -1.0;
    real_t rel_TE_max = 0.0;
    // reusable iterator
    int i = 0;

    std::cout << "\n============\nRunning sim.\n" << std::flush;
    std::cout << "\n";
    std::cout << "Using "<<steps<<" steps with "<<NN<<" gridpoints. Output every "
        <<output_interval<<" steps.\n";
    std::cout << " USE_FIXW is "<<USE_FIXW<<" ("<<FIXW<<"), SMOOTH_RHO is "<<SMOOTH_RHO<<", USE_WENO is "
        <<USE_WENO<<".\n";
    std::cout << "\n" << std::flush;

    real_t *Ab = agg + 0;
    real_t *Rt = agg + 1*NN;
    real_t *mt = agg + 2*NN;
    real_t *Ut = agg + 3*NN;

    kCoefficientData ks;
    ONLY_ALLOC_N(ks.agg_f, NFIELDS*NN)
    ONLY_ALLOC(ks.kR1) ONLY_ALLOC(ks.kR2) ONLY_ALLOC(ks.kR3) ONLY_ALLOC(ks.kR4) ONLY_ALLOC(ks.kR5) ONLY_ALLOC(ks.kR6)
    ONLY_ALLOC(ks.km1) ONLY_ALLOC(ks.km2) ONLY_ALLOC(ks.km3) ONLY_ALLOC(ks.km4) ONLY_ALLOC(ks.km5) ONLY_ALLOC(ks.km6)
    ONLY_ALLOC(ks.kU1) ONLY_ALLOC(ks.kU2) ONLY_ALLOC(ks.kU3) ONLY_ALLOC(ks.kU4) ONLY_ALLOC(ks.kU5) ONLY_ALLOC(ks.kU6)

    ALLOC(TER) ALLOC(TEm) ALLOC(TEU)
    ALLOC(delta_l_CFL)

    // main integration loop
    std::ofstream output;
    output.open("output.dat", std::ios::out | std::ios::binary | std::ios::trunc);
    int s = 0, isteps = 0;
    for(s=0; s<=steps; s++)
    {
        // Upkeep
        #pragma omp critical
        if(s%output_interval==0 or s==steps)
            if(output_interval > 0)
                write_output(agg, l, output);

        // Zero RK coefficients
#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
        {
            ks.kR1[i] = 0.0; ks.kR2[i] = 0.0; ks.kR3[i] = 0.0; ks.kR4[i] = 0.0; ks.kR5[i] = 0.0; ks.kR6[i] = 0.0;
            ks.km1[i] = 0.0; ks.km2[i] = 0.0; ks.km3[i] = 0.0; ks.km4[i] = 0.0; ks.km5[i] = 0.0; ks.km6[i] = 0.0;
            ks.kU1[i] = 0.0; ks.kU2[i] = 0.0; ks.kU3[i] = 0.0; ks.kU4[i] = 0.0; ks.kU5[i] = 0.0; ks.kU6[i] = 0.0;
        }

        // Compute K coefficients
        // RKF(4,5) method https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
        k_calc(agg, &ks,  l, deltal, 0,
            ks.kR1, ks.km1, ks.kU1, 0, 0, 0, 0, 0);  // Compute k1 coefficients
        for(i=0; i<13*NN; i++) { if(std::isnan(ks.agg_f[i])) { std::cout << "k1 nan at " << i << ", " << s << "\n"; return -1; } }
        k_calc(agg, &ks,  l, deltal, 2.0/9.0,
            ks.kR2, ks.km2, ks.kU2, 2.0/9.0, 0, 0, 0, 0);  // Compute k2 coefficients
        for(i=0; i<13*NN; i++) { if(std::isnan(ks.agg_f[i])) { std::cout << "k2 nan at " << i << ", " << s << "\n"; return -1; } }
        k_calc(agg, &ks,  l, deltal, 1.0/3.0,
            ks.kR3, ks.km3, ks.kU3, 1.0/12.0, 1.0/4.0, 0, 0, 0);  // Compute k3 coefficients
        for(i=0; i<13*NN; i++) { if(std::isnan(ks.agg_f[i])) { std::cout << "k3 nan at " << i << ", " << s << "\n"; return -1; } }
        k_calc(agg, &ks,  l, deltal, 3.0/4.0,
            ks.kR4, ks.km4, ks.kU4, 69.0/128.0, -243.0/128.0, 135.0/64.0, 0, 0);  // Compute k4 coefficients
        for(i=0; i<13*NN; i++) { if(std::isnan(ks.agg_f[i])) { std::cout << "k4 nan at " << i << ", " << s << "\n"; return -1; } }
        k_calc(agg, &ks,  l, deltal, 1.0,
            ks.kR5, ks.km5, ks.kU5, -17.0/12.0, 27.0/4.0, -27.0/5.0, 16.0/15.0, 0);  // Compute k5 coefficients
        for(i=0; i<13*NN; i++) { if(std::isnan(ks.agg_f[i])) { std::cout << "k5 nan at " << i << ", " << s << "\n"; return -1; } }
        k_calc(agg, &ks,  l, deltal, 5.0/6.0,
            ks.kR6, ks.km6, ks.kU6, 65.0/432.0, -5.0/16.0, 13.0/16.0, 4.0/27.0, 5.0/144.0);  // Compute k6 coefficients
        for(i=0; i<13*NN; i++) { if(std::isnan(ks.agg_f[i])) { std::cout << "k6 nan at " << i << ", " << s << "\n"; return -1; } }

        // Truncation error estimate
#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
        {
            TER[i] = std::fabs( 1.0/150.0*ks.kR1[i] - 3.0/100.0*ks.kR3[i] + 16.0/75.0*ks.kR4[i] + 1.0/20.0*ks.kR5[i] - 6.0/25.0*ks.kR6[i] );
            TEm[i] = std::fabs( 1.0/150.0*ks.km1[i] - 3.0/100.0*ks.km3[i] + 16.0/75.0*ks.km4[i] + 1.0/20.0*ks.km5[i] - 6.0/25.0*ks.km6[i] );
            TEU[i] = std::fabs( 1.0/150.0*ks.kU1[i] - 3.0/100.0*ks.kU3[i] + 16.0/75.0*ks.kU4[i] + 1.0/20.0*ks.kU5[i] - 6.0/25.0*ks.kU6[i] );
        }
        real_t TER_max = max(TER), TEm_max = max(TEm), TEU_max = max(TEU);
        real_t TE_max = std::max( TER_max, std::max( TEm_max, TEU_max ) );

        real_t R_mean_L1 = mean_L1(Rt)+1.0e-15, m_mean_L1 = mean_L1(mt)+1.0e-15, U_mean_L1 = mean_L1(Ut)+1.0e-15;
        rel_TE_max = std::max( TER_max/R_mean_L1, std::max( TEm_max/m_mean_L1, TEU_max/U_mean_L1 ) );

        if(rel_TE_max < TOL)
        {
            // finalize step, otherwise don't and the stepsize gets smaller.
#pragma omp parallel for default(shared) private(i)
            for(i=0; i<NN; i++)
            {
                Rt[i] = Rt[i] + 47.0/450.0*ks.kR1[i] + 12.0/25.0*ks.kR3[i] + 32.0/225.0*ks.kR4[i] + 1.0/30.0*ks.kR5[i] + 6.0/25.0*ks.kR6[i];
                mt[i] = mt[i] + 47.0/450.0*ks.km1[i] + 12.0/25.0*ks.km3[i] + 32.0/225.0*ks.km4[i] + 1.0/30.0*ks.km5[i] + 6.0/25.0*ks.km6[i];
                Ut[i] = Ut[i] + 47.0/450.0*ks.kU1[i] + 12.0/25.0*ks.kU3[i] + 32.0/225.0*ks.kU4[i] + 1.0/30.0*ks.kU5[i] + 6.0/25.0*ks.kU6[i];
            }
            l = l + deltal;
            isteps++;
        }
        agg_pop(agg, l);

        q = -1.0;
        if(rel_TE_max > 0)
        {
            q = q_mult*std::pow(TOL/rel_TE_max, 1.0/5.0);
            q = std::min(q, (real_t) 1.5); // limit stepsize growth
            deltal *= q;
        }
        
        real_t *dRtdAb  = agg + 4*NN;
        real_t *gammab2 = agg + 6*NN;
        real_t *phi     = agg + 10*NN;
#pragma omp parallel for default(shared) private(i)
        for(i=1; i<NN; i++)
        {
            real_t dA = Ab[i] - Ab[i-1];
            real_t emp = std::exp(-phi[i]);
            real_t gammab = gammab2[i] < 0 ? 1.0e-12 : std::sqrt(gammab2[i]);
            real_t fac = std::fabs((Rt[i]+Ab[i]*dRtdAb[i])/gammab);

            // numerical factor 5 is just for QCD.
            // Happens to satisfy inequality. Could do better, would need to for w=0.
            delta_l_CFL[i] = 5*dA*emp*fac;
        }
        delta_l_CFL[0] = delta_l_CFL[1];
        real_t min_delta_l_CFL = min(delta_l_CFL);
        if(min_delta_l_CFL < deltal)
            deltal = min_delta_l_CFL;

        // Flag when the density perturbation enters the cosmic horizon
        // (linear) interpolation of delta in both time and space directions
        real_t *rhot = agg + 7*NN;
        if(deltaH < 0)
        {
            real_t AbRt_horizon = std::sqrt( G(l) );

            for(i=0; i<NN - 1; i++)
            {
                real_t AbRt1 = Ab[i]*Rt[i];
                real_t AbRt2 = Ab[i+1]*Rt[i+1];

                if( AbRt1 < AbRt_horizon && AbRt2 > AbRt_horizon ) // at horizon
                {
                    // interpolate rhot and delta at the horizon.
                    real_t AbRt_frac = (AbRt_horizon - AbRt1)/(AbRt2 - AbRt1);

                    old_rhot_horizon = rhot_horizon;
                    rhot_horizon = AbRt_frac*(rhot[i+1] - rhot[i]) + rhot[i];
                    old_delta_horizon = delta_horizon;
                    delta_horizon = AbRt_frac*(mt[i+1] - mt[i]) + mt[i] - 1;
                    old_m_horizon = m_horizon;
                    m_horizon = AbRt_frac*(mass_at(agg, l, i+1) - mass_at(agg, l, i)) + mass_at(agg, l, i);

                    if(old_rhot_horizon >= 1 && rhot_horizon < 1)
                    {
                        // interpolate deltaH at the time of horizon crossing (when the underdensity crosses).
                        real_t rhot_frac = (1.0 - old_rhot_horizon)/(rhot_horizon - old_rhot_horizon);
                        deltaH = rhot_frac*(delta_horizon - old_delta_horizon) + old_delta_horizon;
                        real_t mH = rhot_frac*(m_horizon - old_m_horizon) + old_m_horizon;
                        std::cout << "Density perturbation at cosmological horizon crossing found: \n  deltaH=" << deltaH
                            << " and mH=" << mH << " near l=" << l
                            << ", rhot in (" << rhot_horizon << "," << old_rhot_horizon << ")"
                            << ", mt in (" << mt[i] << "," << mt[i+1] << "), AbRt in (" << AbRt1 << "," << AbRt2 << ")"
                            << ", and AbRt_horizon=" << AbRt_horizon << " at step "<<s<<"\n";
                    }
                    break;
                }
            }
        }

        // Check for diverging rho (singularity forming), or shrinking rho (homogenizing)
        prev_rho0 = rho0;
        rho0 = max(rhot);
        if(rho0 > max_rho0)
            max_rho0 = rho0;
        prev_bh_mass = bh_mass;
        bh_mass = ms_mass(agg, l);
        real_t rho0_thresh = 1.0e9;
        if(rho0 > rho0_thresh)
        {
            // Assume singularity is forming
            flag = 1;
            std::cout << "Density becoming singular at step "<<s<<". q was "<<q<<"\n";
            real_t rho0_frac = (rho0_thresh - prev_rho0)/(rho0 - prev_rho0);
            bh_mass = rho0_frac*(bh_mass - prev_bh_mass) + prev_bh_mass;
            std::cout << "BH mass near singularity was " << bh_mass << "\n";
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
                    real_t rho0_frac = (rho0_thresh - prev_rho0)/(rho0 - prev_rho0);
                    bh_mass = rho0_frac*(bh_mass - prev_bh_mass) + prev_bh_mass;
                    std::cout << "BH mass near singularity was" << bh_mass << "\n";
                    break;
                }
            }
            if(flag == 3) break;
        }

        // Error checking
        bool hasnan = false;
        for(i=0; i<NN; i++)
        {
            if(std::isnan(Rt[i]))
            {
                if(hasnan == false)
                    std::cout << "Rt NaN detected.\n" << std::flush;
                hasnan = true;
            }
            if(std::isnan(mt[i]))
            {
                if(hasnan == false)
                    std::cout << "mt NaN detected.\n" << std::flush;
                hasnan = true;
            }
            if(std::isnan(Ut[i]))
            {
                if(hasnan == false)
                    std::cout << "Ut NaN detected.\n" << std::flush;
                hasnan = true;
            }
            if(std::fabs(Rt[i]) > 1.0e12)
            {
                if(hasnan == false)
                    std::cout << "Diverging Rt detected.\n" << std::flush;
                hasnan = true;
            }
        }
        if(deltal < 1.0e-11 || deltal > 1.0 || hasnan || rel_TE_max <= 0 || std::isnan(rel_TE_max))
        {
            std::cout << "Error at step "<<s<<". q="<<q<<", l="<<l<<", hasnan="<<hasnan<<".\n";
            std::cout << "  L1 vals were:  "<<R_mean_L1<<", "<<m_mean_L1<<", "<<U_mean_L1<<"\n";
            std::cout << "   Errors were:  "<<TER_max<<", "<<TEm_max<<", "<<TEU_max<<"\n";
            std::cout << "    deltal was:  "<<deltal<<"\n";
            if(output_interval > 0)
                write_output(agg, l, output);
            flag = -1;
            break;
        }

        if(regrid_interval > 0 && s%regrid_interval == 0)
        {
            regrid(agg, regrid_lam, regrid_nu);
        }

    }

    std::cout << "\nFinal log(a) after step "<<s<<" (isteps="<<isteps<<") was "<< l <<", delta log(a) was "<<deltal<<" (q="<<q<<").\n";
    std::cout << "  max_rho0 was "<<max_rho0<<", rel_TE_max was "<<rel_TE_max<<".\n";
    std::cout << "\nDone running.\n=============\n";

    output.close();

    free(ks.agg_f);
    free(ks.kR1); free(ks.kR2); free(ks.kR3); free(ks.kR4); free(ks.kR5); free(ks.kR6);
    free(ks.km1); free(ks.km2); free(ks.km3); free(ks.km4); free(ks.km5); free(ks.km6);
    free(ks.kU1); free(ks.kU2); free(ks.kU3); free(ks.kU4); free(ks.kU5); free(ks.kU6);
    free(TER); free(TEm); free(TEU);
    free(delta_l_CFL);

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
void ics(real_t *agg, real_t &l, real_t &deltaH, real_t &max_rho0, real_t &bh_mass,
    real_t amp, real_t d, int n, real_t Ld, bool use_fixw)
{
    NN = n;
    deltaH = -1.0;
    max_rho0 = 0.0;
    bh_mass = 0.0;

    USE_FIXW = false;
    real_t rho_b = rho_background(l);
    real_t P_b = P_background(l);
    FIXW = dPdrho(rho_b);
    USE_FIXW = use_fixw;

    real_t alpha = 2.0/3.0*rho_b/( rho_b + P_b ); // Eq. 36, generalized
    real_t L = Ld*d; // Simulation size

    real_t *Ab = agg + 0;
    real_t *Rt = agg + 1*NN;
    real_t *mt = agg + 2*NN;
    real_t *Ut = agg + 3*NN;

    std::cout << "\n==================\nInitializing sim.\n==================\n" << std::flush;
    std::cout << "\n";
    std::cout << "Using "<<NN<<" gridpoints, amplitude "
                    <<amp<<", perturbation size "<<d<<", alpha = "<<alpha<<".\n";
    std::cout << std::flush;

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

    agg_pop(agg, l);
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

    real_t l=0, deltaH, max_rho0, bh_mass;
    ics(agg, l, deltaH, max_rho0, bh_mass, amp, d, NN, 20.0, USE_FIXW);
    run_sim(agg, l, deltaH, max_rho0, bh_mass, steps, output_interval, false, 0.25,
        SMOOTH_RHO, USE_WENO, 0, 0.0, 0.0, 1.0e-7);

    free(agg);

    return 0;
}
