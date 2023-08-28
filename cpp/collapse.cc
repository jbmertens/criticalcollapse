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

typedef float real_t;

int NN = 3200;

#define PI 3.14159265358979323846264338328
#define TOL (1.0e-7) // integration tolerance
#define RHO_B_I 0.1

#define NFIELDS 13

#define ALLOC_N(arr_n, ARR_SIZE) \
    real_t *arr_n; \
    arr_n = (real_t *) malloc( (ARR_SIZE) * ((long long) sizeof(real_t)));

#define ALLOC(arr_n) \
    ALLOC_N(arr_n, NN)

// Arrays for equation of state interpolation
std::vector<real_t> logrho{-23.0258509, 1.25944027,  5.86743592, 10.47443832, 12.80727797, 16.24077033,
   20.17741006, 21.87146739, 22.54685293, 24.86073802, 25.88516263, 30.81629677,
   40.14966858, 42.94194384, 45.77729053, 49.56660234, 53.73838727, 92.1034037};
std::vector<real_t> logP{-24.1244632, 0.15168708,  4.7676633 ,  9.37390511, 11.68836246, 15.05383501,
   18.98355404, 20.54248356, 21.11727849, 23.49996688, 24.58907975, 29.64524541,
   39.04613022, 41.81963086, 44.6428769 , 48.43776105, 52.63885477, 91.0047914};
// Interpolating function
tk::spline loglog_Pinterp(logrho, logP);
// 
real_t SPL_MIN_RHO=1e0;
real_t SPL_MAX_RHO=1e23;


bool use_fixw = false;
real_t fixw = 1.0/3.0;

/**
 * Background density at a given l = log(a), with a the scale factor.
 */
real_t rho_background(real_t l)
{
    // TODO
    return 1.0;
}

/**
 * Background pressure at a given l = log(a), with a the scale factor.
 */
real_t P_background(real_t l)
{
    real_t log_rho_b = std::log(rho_background(l));
    return std::exp(loglog_Pinterp(log_rho_b));
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
 * Compute \tilde{P} of \tilde{\rho}, assuming a
 * specific background density rho_b.
 */
extern "C"
real_t Pt_of_rhot(real_t rhot, real_t rho_b)
{
    if(rhot < 1.0e-10)
        rhot = 1.0e-10;

    if(use_fixw)
        return fixw*rhot;

    real_t rho = rhot*rho_b;
    real_t logrho = std::log(rho);

    if(logrho < -23.0 or logrho > 93.0)
        return rho/3.0;

    real_t P_in = std::exp(loglog_Pinterp(logrho));
    real_t P_out = rho/3.0;
    real_t P = P_out + H(rho, SPL_MIN_RHO)*H(SPL_MAX_RHO, rho)*(P_in - P_out);

    real_t Pt = P/rho_b;
    return Pt;
}

/**
 * Compute d\tilde{P}/d\tilde{\rho} (or, equivalently, dP/drho) assuming a
 * specific background density rho_b.
 */
extern "C" 
real_t dPdrho(real_t rhot, real_t rho_b)
{
    if(rhot < 1.0e-10)
        rhot = 1.0e-10;

    if(use_fixw)
        return fixw;

    real_t Pt = Pt_of_rhot(rhot, rho_b);
    real_t rho = rhot*rhob;
    real_t P = Pt*rhob;
    real_t logrho = std::log(rho);

    real_t dPdrho_in = P/rho*loglog_Pinterp.deriv(1, logrho);
    real_t dPdrho_out = 1.0/3.0;
    real_t dPdrho = dPdrho_out + H(rho, SPL_MIN_RHO)*H(SPL_MAX_RHO, rho)*(dPdrho_in - dPdrho_out);

    return dPdrho;
}
real_t dPtdrhot(real_t rhot, real_t rho_b)
{
    return dPdrho(rhot, rho_b);
}

/**
 * Compute derivatives of a function.
 * Assume derivatives are 0 at boundaries.
 */
void dfdA(real_t *f, real_t *A, real_t *dfdA)
{
    dfdA[0] = 0;
    dfdA[NN-1] = 0;

    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN-1; i++)
    {
        dfdA[i] = 0.5* (
            (f[i+1]-f[i]) / (A[i+1]-A[i])
            + (f[i]-f[i-1]) / (A[i]-A[i-1])
        );
    }
}

/**
 * Compute dPtdAb
 * Uses a "staggered" grid to improve stability.
 * Assume dPtdAb is 0 at boundaries.
 */
void dPtdAbstg(real_t *agg, real_t *dPtdAb)
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
        Pt_stg = Pt_of_rhot(rhot_stg);

        dPtdAb[i] = (Pt_stg - Pt_stg_prev)/(Ab_stg - Ab_stg_prev);
    }

    dPtdAb[0] = 0;
    dPtdAb[NN-1] = 0;
}

/**
 * Populate the aggregate list of values ("agg"), structured as NFIELDS
 * concatenated arrays of length NN. Assumes The first 4 arrays already
 * contain Abar, Rtilde, mtilde, Utilde, and populate the remaining
 * NFIELDS - 4 arrays (detailed in code below).
 */
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
    real_t *Pr     = agg + 8*NN;
    real_t *dPtdAb = agg + 9*NN;
    real_t *phi    = agg + 10*NN;
    real_t *m2oR   = agg + 11*NN;
    real_t *Qt     = agg + 12*NN;

    // Derivatives of R and m wrt. Ab, used later
    dfdA(Rt, Ab, dRtdA);
    dfdA(mt, Ab, dmtdA);

    // compute gamma and 2*m/R, used later
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        real_t AbRt = Ab[i]*Rt[i];
        gammab[i] = std::sqrt( RHO_B_I/rho_b/e2l + AbRt*AbRt*(Ut[i]*Ut[i] - mt[i]) );
        m2oR[i] = e2l*(rho_b/RHO_B_I)*Ab[i]*Ab[i]*Rt[i]*Rt[i]*mt[i];
    }


    // Compute density (rho tilde)
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN-1; i++)
    {
        rhot[i] = mt[i] + Ab[i]*Rt[i]*(mt[i+1] - mt[i-1])
                / 3 / (Ab[i+1]*Rt[i+1] - Ab[i-1]*Rt[i-1]);
    }
    rhot[0] = mt[0];
    rhot[NN-1] = mt[NN-1] + Ab[NN-1]*Rt[NN-1]*(mt[NN-1] - mt[NN-2])
                / 3 / (Ab[NN-1]*Rt[NN-1] - Ab[NN-2]*Rt[NN-2]);


    // Compute pressure (P tilde)
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
        Pt[i] = Pt_of_rhot(rhot[i]);

    // Compute derivative of pressure, using staggered grid
    dPtdAbstg(Rt, mt, Ab, dPtdAb);

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
                Qt[i] = kappa*(d2Ab/2)*(d2Ab/2)/e2ax*AbRtUtp*AbRtUtp;
            else
                Qt[i] = 0;
        }
        Qt[NN-1] = Qt[NN-2];
        Qt[0] = Qt[1];

        // Smooth the viscosity curve a bit
        for(i=2; i<NN-2; i++)
            Qt[i] = (Qt[i-2]+2*Qt[i-1]+3*Qt[i]+2*Qt[i+1]+Qt[i+2])/9.0;

#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
            Pt[i] += Qt[i];
    }

    // Perform integration to compute metric (phi). The boundary value can be
    // chosen; shifting it by a constant amounts to re-scaling the time
    // coordinate. We use Eq. 51, Assuming radiation-dominated cosmology values
    // for the equation of state. Close to zero works, anyways.
    phi[NN-1] = -std::log(rhot[NN-1])/4.0;
    phi[NN-2] = phi[NN-1] + (Pt[NN-1] - Pt[NN-2])/(Pt[NN-2] + rhot[NN-2]);
    for(i=NN-3; i>=0; i--)
        phi[i] = (Pt[i+2] - Pt[i])/(Pt[i+1] + rhot[i+1]) + phi[i+2];
}


// Compute Runge-Kutta 'k' coefficients
void k_calc(real_t *agg,
    real_t *kR_p, real_t *kR_f,
    real_t *km_p, real_t *km_f,
    real_t *kU_p, real_t *kU_f,
    real_t l, real_t deltal, real_t kcoeff)
{
    int i=0;

    // Populate temporary agg array with computed values
    ALLOC_N(tmp_agg, NFIELDS*NN)
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
    real_t *gammat = tmp_agg + 6*NN;
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
            dPtdAboAb = 5.0/3.0*dPdrho(rhot[0])
                *(mt[1] + mt[1] - 2*mt[0]) / Ab[1] / Ab[1];
        else
            dPtdAboAb = dPtdAb[i] / Ab[i];

        kR_f[i] = Rt[i]*(Ut[i]*ep - 1);
        km_f[i] = 2/alpha*mt[i] - 3*Ut[i]*ep*(Pt[i] + mt[i]);
        kU_f[i] = Ut[i]/alpha - ep*(
            gamma[i]*gamma[i]*dPtdAboAb / (Rt[i]*AbRt_prime*(rhot[i] + Pt[i]))
            + (2*Ut[i]*Ut[i] + mt[i] + 3*Pt[i])/2.0
        );
    }

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
int run_sim(real_t *agg, real_t &l, int steps, int output_interval)
{
    int flag = 0;

    std::cout << "\n============\nRunning sim.\n============\n" << std::flush;
    std::cout << "\n";
    std::cout << "Using "<<steps<<" steps with "<<NN<<" gridpoints. Output every "
        <<output_interval<<" steps.\n";
    std::cout << "\n" << std::flush;

    real_t *Abar      = agg + 0;
    real_t *Rt        = agg + 1*NN;
    real_t *mt        = agg + 2*NN;
    real_t *Ut        = agg + 3*NN;
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
    real_t deltal = 3.67e-5;
    real_t max_rho0 = 0.0, prev_rho0 = 0.0;
    for(s=0; s<=steps; s++)
    {
        // Upkeep
        #pragma omp critical
        if(s%output_interval==0 or s==steps)
            if(output_interval > 0)
                write_output(agg, l, output);

        // Integration details
        k_calc(agg,  zeros, kR1,  zeros, km1,  zeros, kU1,  l, deltal,  0.0);
        k_calc(agg,  kR1, kR2,    km1, km2,    kU1, kU2,    l, deltal,  0.5);
        k_calc(agg,  kR2, kR3,    km2, km3,    kU2, kU3,    l, deltal,  0.75);
#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
        {
            kRi[i] = ( 2*kR1[i] + 3*kR2[i] + 4*kR3[i] )/9.0;
            kmi[i] = ( 2*km1[i] + 3*km2[i] + 4*km3[i] )/9.0;
            kUi[i] = ( 2*kU1[i] + 3*kU2[i] + 4*kU3[i] )/9.0;
        }
        k_calc(agg,  kRi, kR4,  kmi, km4,  kUi, kU4,  l, deltal,  1.0);

        // Determine maximum errors
        real_t E_R_max = 0, E_m_max = 0, E_U_max = 0,
               tol_R_max = 0, tol_m_max = 0, tol_U_max = 0;
        for(i=0; i<NN; i++)
        {
            real_t E_R = deltal*fabs(-5*kR1[i]/72 + kR2[i]/12 + kR3[i]/9 - kR4[i]/8);
            if(E_R > E_R_max) { E_R_max = E_R; }
            real_t E_m = deltal*fabs(-5*km1[i]/72 + km2[i]/12 + km3[i]/9 - km4[i]/8);
            if(E_m > E_m_max) { E_m_max = E_m; }
            real_t E_U = deltal*fabs(-5*kU1[i]/72 + kU2[i]/12 + kU3[i]/9 - kU4[i]/8);
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
                Rt[i] = Rt[i] + deltaxi/9*(2*kR1[i] + 3*kR2[i] + 4*kR3[i] );
                mt[i] = mt[i] + deltaxi/9*(2*km1[i] + 3*km2[i] + 4*km3[i] );
                Ut[i] = Ut[i] + deltaxi/9*(2*kU1[i] + 3*kU2[i] + 4*kU3[i] );
            }
            l = l + deltal;
        }
        agg_pop(agg, l);


        // adjust step size for next step
        real_t q = 0.75*std::pow( std::min(tol_R_max/E_R_max,
            std::min(tol_m_max/E_m_max, tol_U_max/E_U_max) ), 1.0/3.0);
        q = std::min(q, (real_t) 5.0); // limit stepsize growth
        deltal *= q;


        // if(self.delta == -1):
        //     r = self.rho(self.R, self.m)
        //     pos = zero_crossing(self.Abar, (self.Abar - 1/np.exp((self.alpha-1) * self.xi) / self.R))
        //     if(pos > 0 and np.interp(pos, self.Abar, r) < 1):
        //         self.delta = np.interp(pos, self.Abar, self.m) - 1

        // Check for diverging rho (singularity forming), or shrinking rho (homogenizing)
        real_t *rhot = agg + 7*NN;
        prev_rho0 = rho0;
        rho0 = rhot[3];
        if(rho0 > max_rho0)
            max_rho0 = rho0;
        if(rho0 > 1.0e10)
        {
            // Assume singularity is forming
            flag = 1;
            std::cout << "Density becoming singular at step "<<s<<". q was "<<q<<"\n";
            break;
        }
        if(rho0 < 0.25*max_rho0)
        {
            // Assume no BH is forming
            flag = -1;
            std::cout << "Density dropping substantially at step "<<s<<". q was "<<q<<"\n";
            break;
        }

        // Error checking
        bool hasnan = false;
        for(i=0; i<NN; i++)
            if(std::isnan(Rt[i]))
                hasnan = true;
        if(deltal < 1.0e-11 || deltal > 1.0 || hasnan)
        {
            std::cout << "Error at step "<<s<<". q was "<<q<<"\n";
            std::cout << "Errors were "<<err_R_max<<", "<<err_m_max<<", "<<err_U_max<<"\n";
            std::cout << "            "<<E_R_max<<", "<<E_m_max<<", "<<E_U_max<<"\n";
            std::cout << "deltal was "<<deltal<<"\n";
            if(output_interval > 0)
                write_output(agg, xi, output);
            break;
        }

    }

    std::cout << "\nFinal log(a) after step "<<s<<" was "<< l <<", delta log(a) was "<<deltal<<".\n";
    std::cout << "\n============\nDone running.\n============\n";

    output.close();

    agg_pop(agg, l);

    free(zeros); free(Abar); free(Rt); free(mt); free(Ut);
    free(kR1); free(kR2); free(kR3); free(kR4); free(Rnew);
    free(km1); free(km2); free(km3); free(km4); free(mnew);
    free(kU1); free(kU2); free(kU3); free(kU4); free(Unew);

    return flag;
}

/**
 * Set initial conditions. Assumes a pure-radiation era, i.e.
 * significantly before the QCD transition.
 * amp is the perturbation amplitude,
 * d is the perturbation size in horizon units.
 */
extern "C"
void ics(real_t *agg, real_t &l, real_t amp, real_t d, int n)
{
    std::cout << "\n==================\nInitializing sim.\n==================\n" << std::flush;
    std::cout << "\n";
    std::cout << "Using "<<NN<<" gridpoints, amplitude "
                    <<amp<<", perturbation size "<<d<<".\n";
    std::cout << "\n" << std::flush;

    NN = n;
    l = 0;

    real_t *Abar      = agg + 0;
    real_t *Rt        = agg + 1*NN;
    real_t *mt        = agg + 2*NN;
    real_t *Ut        = agg + 3*NN;

    real_t w_rad = 1.0/3.0;
    real_t alpha = 0.5; // Eq. 36
    
    real_t L = 20.0*d; // Simulation size

    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        Abar[i] = i*L/NN;
        real_t deltam0 = amp*std::exp(-Abar[i]*Abar[i]/2/d/d);
        real_t deltam0P = -Abar[i]/d/d*deltam0;

        Rt[i] = 1 - alpha/2*(deltam0 + Abar[i]*deltam0P/6.0 );
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

    real_t l;
    ics(agg, l, amp, d, NN);
    run_sim(agg, l, steps, output_interval);

    free(agg);

    return 0;
}
