/* INCLUDES */
#include <cmath>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <getopt.h>
#include <string>

// #include <boost/math/interpolators/makima.hpp>
// using boost::math::interpolators::makima;
// #include <utility>
#include <vector>
#include "spline.h"

typedef float real_t;

int NN = 3200;

#define PI 3.14159265358979323846264338328

#define L 21.0
#define OD_SIZE 1.6

#define OUTPUT true

#define ALPHA (2.0/3.0/(1 + fixw))
#define T0 (ALPHA * std::sqrt(3 / (8*PI*rho0)))
#define RH (T0 / ALPHA)

#define TOL (1.0e-7) // integration tolerance


#define ALLOC_N(arr_n, ARR_SIZE) \
    real_t *arr_n; \
    arr_n = (real_t *) malloc( (ARR_SIZE) * ((long long) sizeof(real_t)));

#define ALLOC(arr_n) \
    ALLOC_N(arr_n, NN)

std::vector<real_t> logrho{-23.0258509, 1.25944027,  5.86743592, 10.47443832, 12.80727797, 16.24077033,
   20.17741006, 21.87146739, 22.54685293, 24.86073802, 25.88516263, 30.81629677,
   40.14966858, 42.94194384, 45.77729053, 49.56660234, 53.73838727, 92.1034037};
   
std::vector<real_t> logP{-24.1244632, 0.15168708,  4.7676633 ,  9.37390511, 11.68836246, 15.05383501,
   18.98355404, 20.54248356, 21.11727849, 23.49996688, 24.58907975, 29.64524541,
   39.04613022, 41.81963086, 44.6428769 , 48.43776105, 52.63885477, 91.0047914};

tk::spline loglog_Pinterp(logrho, logP);


real_t min_rho=1e0;
real_t max_rho=1e23;
real_t mu=2;

bool use_fixw = false;
real_t fixw = 1.0/3.0;

real_t H(real_t rho, real_t rho0)
{
    return ( 1 + std::tanh( (std::log(rho) - std::log(rho0))/mu ) )/2;
}

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

extern "C"
real_t P_of_rho(real_t rho)
{
    if(rho < 1.0e-10)
        rho = 1.0e-10;

    if(use_fixw)
        return fixw*rho;

    real_t logrho = std::log(rho);
    if(logrho < -23.0 or logrho > 93.0)
        return rho/3.0;

    real_t P_in = std::exp(loglog_Pinterp(logrho));
    real_t P_out = rho/3.0;
    real_t P = P_out + H(rho, min_rho)*H(max_rho, rho)*(P_in - P_out);

    return P;
}

extern "C"
real_t dPdrho(real_t rho)
{
    if(rho < 1.0e-10)
        rho = 1.0e-10;

    if(use_fixw)
        return fixw;

    real_t P = P_of_rho(rho);
    real_t logrho = std::log(rho);

    real_t dPdrho_in = P/rho*loglog_Pinterp.deriv(1, logrho);
    real_t dPdrho_out = 1.0/3.0;
    real_t dPdrho = dPdrho_out + H(rho, min_rho)*H(max_rho, rho)*(dPdrho_in - dPdrho_out);

    return dPdrho;
}

void dPdAstg(real_t *R, real_t *m, real_t *A, real_t *dPdA)
{
    int i=0;

    real_t P_stg = 0, A_stg = 0;

    for(i=0; i<NN-1; i++)
    {
        real_t A_stg_prev = A_stg;
        A_stg = (A[i]+A[i+1])/2;

        real_t P_stg_prev = P_stg;
        real_t R_stg = (R[i]+R[i+1])/2;
        real_t m_stg = (m[i]+m[i+1])/2;
        real_t rho_stg = m_stg + A_stg*R_stg*(m[i+1] - m[i])
                            / 3 / (A[i+1]*R[i+1] - A[i]*R[i]);
        P_stg = P_of_rho(rho_stg);

        dPdA[i] = (P_stg - P_stg_prev)/(A_stg - A_stg_prev);
    }

    dPdA[0] = 0;
    dPdA[NN-1] = 0;
}


void agg_pop(
    real_t *Abar_in, real_t *Rtilde_in, real_t *mtilde_in, real_t *Utilde_in,
    real_t xi,
    real_t * agg)
{
    int i=0;
    real_t e2ax = std::exp(2*(1 - ALPHA)*xi);

    // Populate all information; stored in "agg"regate array
    real_t *Abar      = agg + 0;
    real_t *Rtilde    = agg + 1*NN;
    real_t *mtilde    = agg + 2*NN;
    real_t *Utilde    = agg + 3*NN;
    real_t *dRtildedA = agg + 4*NN;
    real_t *dmtildedA = agg + 5*NN;
    real_t *gamma     = agg + 6*NN;
    real_t *rho       = agg + 7*NN;
    real_t *P         = agg + 8*NN;
    real_t *dPdA      = agg + 9*NN;
    real_t *phi       = agg + 10*NN;
    real_t *m2oR      = agg + 11*NN;
    real_t *Q         = agg + 12*NN;

    // A, R, m, U are read in
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        Abar[i] = Abar_in[i];
        Rtilde[i] = Rtilde_in[i];
        mtilde[i] = mtilde_in[i];
        Utilde[i] = Utilde_in[i];
    }

    // Derivatives of R and m, used later
    dfdA(Rtilde, Abar, dRtildedA);
    dfdA(mtilde, Abar, dmtildedA);

    // compute gamma and 2*m/R, used later
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        real_t AR = Abar[i] * Rtilde[i];
        gamma[i] = std::sqrt( e2ax + AR*AR * (Utilde[i]*Utilde[i] - mtilde[i]) );
        m2oR[i] = Rtilde[i]*Rtilde[i]*mtilde[i]*Abar[i]*Abar[i]/e2ax;
    }


    // Compute density (rho)
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<NN-1; i++)
    {
        rho[i] = mtilde[i] + Abar[i]*Rtilde[i]*(mtilde[i+1] - mtilde[i-1])
                / 3 / (Abar[i+1]*Rtilde[i+1] - Abar[i-1]*Rtilde[i-1]);
    }
    rho[0] = mtilde[0];
    rho[NN-1] = mtilde[NN-1] + Abar[NN-1]*Rtilde[NN-1]*(mtilde[NN-1] - mtilde[NN-2])
                / 3 / (Abar[NN-1]*Rtilde[NN-1] - Abar[NN-2]*Rtilde[NN-2]);


    // Compute pressure
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
        P[i] = P_of_rho(rho[i]);

    // Compute (artificial) viscosity
    real_t kappa = 2.0;
    if(kappa > 0)
    {
#pragma omp parallel for default(shared) private(i)
        for(i=1; i<NN-1; i++)
        {
            real_t d2A = Abar[i+1] - Abar[i-1];
            real_t Up = (Utilde[i+1] - Utilde[i-1]) / d2A;
            real_t ARp = (Abar[i+1]*Rtilde[i+1] - Abar[i-1]*Rtilde[i-1]) / d2A;
            real_t ARUp = (Abar[i+1]*Rtilde[i+1]*Utilde[i+1] - Abar[i-1]*Rtilde[i-1]*Utilde[i-1]) / d2A;
            if(ARp*Utilde[i] < -Abar[i]*Rtilde[i]*Up)
                Q[i] = kappa*(d2A/2)*(d2A/2)/e2ax*ARUp*ARUp;
            else
                Q[i] = 0;
        }
        Q[NN-1] = Q[NN-2];
        Q[0] = Q[1];

        for(i=2; i<NN-2; i++)
            Q[i] = (Q[i-2]+2*Q[i-1]+3*Q[i]+2*Q[i+1]+Q[i+2])/9.0;

#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
            P[i] += Q[i];
    }

    // Compute derivative of pressure, using staggered grid
    dPdAstg(Rtilde, mtilde, Abar, dPdA);

    // Perform integration to compute metric (phi)
    phi[NN-1] = -1.5*ALPHA*dPdrho(rho[NN-1])*std::log(rho[NN-1]);
    phi[NN-2] = phi[NN-1] + (P[NN-1] - P[NN-2])/(P[NN-2] + rho[NN-2]);
    for(i=NN-3; i>=0; i--)
        phi[i] = (P[i+2] - P[i])/(P[i+1] + rho[i+1]) + phi[i+2];
}


// Compute Runge-Kutta 'k' coefficients
void k_calc(real_t *Rtilde, real_t *kR_p, real_t *kR_f,
    real_t *mtilde, real_t *km_p, real_t *km_f,
    real_t *Utilde, real_t *kU_p, real_t *kU_f,
    real_t *Abar, real_t *agg, real_t xi, real_t deltaxi,
    real_t kcoeff)
{
    int i=0;

    // Store intermediate values in _f register
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        kR_f[i] = Rtilde[i] + kR_p[i]*deltaxi*kcoeff;
        km_f[i] = mtilde[i] + km_p[i]*deltaxi*kcoeff;
        kU_f[i] = Utilde[i] + kU_p[i]*deltaxi*kcoeff;
    }
    xi = xi + deltaxi*kcoeff;

    // Populate "agg" with computed values,
    // extract them in relevant arrays
    agg_pop(Abar, kR_f, km_f, kU_f, xi, agg);
    real_t *dRtildedA = agg + 4*NN;
    real_t *dmtildedA = agg + 5*NN;
    real_t *gamma     = agg + 6*NN;
    real_t *rho       = agg + 7*NN;
    real_t *P         = agg + 8*NN;
    real_t *dPdA      = agg + 9*NN;
    real_t *phi       = agg + 10*NN;
    real_t *Q         = agg + 12*NN;

    // fill _f registers with final values
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        real_t R = kR_f[i];
        real_t m = km_f[i];
        real_t U = kU_f[i];
        real_t ep = std::exp(phi[i]);
        real_t AR_prime = R + Abar[i]*dRtildedA[i];

        real_t dPdAoA = 0;
        if(i == 0)
            dPdAoA = 5.0/3.0*dPdrho(rho[0])
                *(km_f[1] + km_f[1] - 2*km_f[0]) / Abar[1] / Abar[1];
        else
            dPdAoA = dPdA[i] / Abar[i];

        kR_f[i] = ALPHA*R*(U*ep - 1);
        km_f[i] = 2*m - 3*ALPHA*U*ep*(P[i] + m);
        kU_f[i] = U - ALPHA*ep*(
            gamma[i]*gamma[i]*dPdAoA / (R*AR_prime*(rho[i] + P[i]))
            + (2*U*U + m + 3*P[i])/2
        );
    }

}

// Write output to file
void write_output(real_t *Abar, real_t *Rtilde, real_t *mtilde, real_t *Utilde,
    real_t *agg,
    real_t xi, std::ofstream & output)
{
    agg_pop(Abar, Rtilde, mtilde, Utilde, xi, agg);
    output.write((char *) agg, 13*NN*sizeof(real_t));
}


extern "C"
int run_sim(int n, int steps, real_t amp, real_t rho0, int output_interval,
    real_t *Abar, real_t *Rtilde, real_t *mtilde, real_t *Utilde,
    real_t *agg, real_t &xi)
{
    NN = n;
    
    std::cout << "\n============\nRunning sim.\n============\n" << std::flush;
    std::cout << "\n";
    std::cout << "Using "<<steps<<" steps with "<<NN<<" gridpoints. Output every "
        <<output_interval<<" steps.\n";
    std::cout << "  amp = "<<amp<<"\n";
    std::cout << "  rho0 = "<<rho0<<"\n";
    std::cout << "\n" << std::flush;

    ALLOC(zeros)
    ALLOC(kR1) ALLOC(kR2) ALLOC(kR3) ALLOC(kR4) ALLOC(Rnew)
    ALLOC(km1) ALLOC(km2) ALLOC(km3) ALLOC(km4) ALLOC(mnew)
    ALLOC(kU1) ALLOC(kU2) ALLOC(kU3) ALLOC(kU4) ALLOC(Unew)

    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
        zeros[i] = 0.0;

    // main integration loop
    std::ofstream output;
    output.open("output.dat", std::ios::out | std::ios::binary | std::ios::trunc);
    int s=0;
    real_t deltaxi = 3.67e-5;
    for(s=0; s<=steps; s++)
    {
        // Run upkeep

            // # Stop running if it becomes clear a BH won't form.
            // if(self.BH_wont_form() == True):
            //     self.timer.stop("adap_run_steps")
            //     return -2

            // if(self.delta == -1):
            //     r = self.rho(self.R, self.m)
            //     pos = zero_crossing(self.Abar, (self.Abar - 1/np.exp((self.alpha-1) * self.xi) / self.R))
            //     if(pos > 0 and np.interp(pos, self.Abar, r) < 1):
            //         self.delta = np.interp(pos, self.Abar, self.m) - 1

            // self.m2oR_prev = self.m2oR
            // self.m2oR = self.R**2 * self.m * self.Abar**2 * np.exp(2 * (self.alpha-1) * self.xi)
            // self.exc_pos = find_exc_pos(self.m2oR)
            // if(self.exc_pos > 0 and self.BH_term == True):
            //     print("Horizon is found, MS run will be terminated! Finished at step", self.step)
            //     self.timer.stop("adap_run_steps")
            //     return -1

            #pragma omp critical
            if(s%output_interval==0 or s==steps)
                if(output_interval > 0)
                    write_output(Abar, Rtilde, mtilde, Utilde, agg, xi, output);

        // Integration details

        k_calc(Rtilde, zeros, kR1,  mtilde, zeros, km1,
            Utilde, zeros, kU1,  Abar, agg, xi, deltaxi,  0.0);

        k_calc(Rtilde, kR1, kR2,  mtilde, km1, km2,
            Utilde, kU1, kU2,  Abar, agg, xi, deltaxi,  0.5);

        k_calc(Rtilde, kR2, kR3,  mtilde, km2, km3,
            Utilde, kU2, kU3,  Abar, agg, xi, deltaxi,  0.75);

#pragma omp parallel for default(shared) private(i)
        for(i=0; i<NN; i++)
        {
            Rnew[i] = Rtilde[i] + deltaxi/9*(2*kR1[i] + 3*kR2[i] + 4*kR3[i] );
            mnew[i] = mtilde[i] + deltaxi/9*(2*km1[i] + 3*km2[i] + 4*km3[i] );
            Unew[i] = Utilde[i] + deltaxi/9*(2*kU1[i] + 3*kU2[i] + 4*kU3[i] );
        }

        k_calc(Rnew, zeros, kR4,  mnew, zeros, km4,
            Unew, zeros, kU4,  Abar, agg, xi, deltaxi,  1.0);

        real_t E_R_max = 0, E_m_max = 0, E_U_max = 0,
               err_R_max = 0, err_m_max = 0, err_U_max = 0;
        for(i=0; i<NN; i++)
        {
            real_t E_R = deltaxi*fabs(-5*kR1[i]/72 + kR2[i]/12 + kR3[i]/9 - kR4[i]/8);
            if(E_R > E_R_max) { E_R_max = E_R; }
            real_t E_m = deltaxi*fabs(-5*km1[i]/72 + km2[i]/12 + km3[i]/9 - km4[i]/8);
            if(E_m > E_m_max) { E_m_max = E_m; }
            real_t E_U = deltaxi*fabs(-5*kU1[i]/72 + kU2[i]/12 + kU3[i]/9 - kU4[i]/8);
            if(E_U > E_U_max) { E_U_max = E_U; }

            real_t err_R = fabs(Rtilde[i])*TOL;
            if(err_R > err_R_max) { err_R_max = err_R; }
            real_t err_m = fabs(mtilde[i])*TOL;
            if(err_m > err_m_max) { err_m_max = err_m; }
            real_t err_U = fabs(Utilde[i])*TOL;
            if(err_U > err_U_max) { err_U_max = err_U; }
        }

        // final field values at the end of the integration step.
        if(E_R_max < err_R_max and E_m_max < err_m_max and E_U_max < err_U_max)
        {
            for(i=0; i<NN; i++)
            {
                Rtilde[i] = Rtilde[i] + deltaxi/9*(2*kR1[i] + 3*kR2[i] + 4*kR3[i] );
                mtilde[i] = mtilde[i] + deltaxi/9*(2*km1[i] + 3*km2[i] + 4*km3[i] );
                Utilde[i] = Utilde[i] + deltaxi/9*(2*kU1[i] + 3*kU2[i] + 4*kU3[i] );
            }
            xi = xi + deltaxi;
        }

        // adjust step size for next step
        real_t q = 0.75*std::pow( std::min(err_R_max/E_R_max,
            std::min(err_m_max/E_m_max, err_U_max/E_U_max) ), 1.0/3.0);
        q = std::min(q, (real_t) 5.0); // limit stepsize growth
        deltaxi *= q;

        bool hasnan = false;
        for(i=0; i<NN; i++)
            if(std::isnan(Rtilde[i]))
                hasnan = true;
        if(deltaxi < 1.0e-11 || deltaxi > 1.0 || hasnan)
        {
            std::cout << "Error at step "<<s<<". q was "<<q<<"\n";
            std::cout << "Errors were "<<err_R_max<<", "<<err_m_max<<", "<<err_U_max<<"\n";
            std::cout << "            "<<E_R_max<<", "<<E_m_max<<", "<<E_U_max<<"\n";
            std::cout << "deltaxi was "<<deltaxi<<"\n";
            if(output_interval > 0)
                write_output(Abar, Rtilde, mtilde, Utilde, agg, xi, output);
            break;
        }

    }

    std::cout << "\nFinal xi after step "<<s<<" was "<< xi <<", deltaxi was "<<deltaxi<<".\n";
    std::cout << "\n============\nDone running.\n============\n";

    output.close();

    agg_pop(Abar, Rtilde, mtilde, Utilde, xi, agg);

    free(zeros);
    free(kR1); free(kR2); free(kR3); free(kR4); free(Rnew);
    free(km1); free(km2); free(km3); free(km4); free(mnew);
    free(kU1); free(kU2); free(kU3); free(kU4); free(Unew);

    return 0;
}

extern "C"
void ics(real_t *Abar, real_t *Rtilde, real_t *mtilde, real_t *Utilde,
    real_t &xi, real_t amp, int n)
{
    NN = n;
    xi = 0;

    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<NN; i++)
    {
        Abar[i] = i*L/NN;
        real_t delta0 = amp * std::exp(-Abar[i]*Abar[i] / 2 / OD_SIZE / OD_SIZE);
        real_t delta0P = amp * delta0 * 2 * (-1 / 2 / OD_SIZE / OD_SIZE ) * Abar[i];
        Rtilde[i] = 1 - ALPHA / 2 * (delta0 + fixw * Abar[i] * delta0P / (1 + 3*fixw) );
        mtilde[i] = 1 + delta0;
        Utilde[i] = 1 - ALPHA * delta0 / 2;
    }
}


int main(int argc, char **argv)
{

    int steps = 1000, output_interval=1000;
    real_t amp = 0.3, rho0 = 1.0;

    static struct option long_options[] =
    {
        {"steps",  required_argument, 0, 's'},
        {"rho0",   required_argument, 0, 'r'},
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
            case 'r':
                rho0 = (real_t) std::stod(optarg);
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
                fprintf(stdout, "\nusage: %s -N [gridpoints] -s [steps] -r [rho_0] -a [amp] -o [output_interval] \n", argv[0]);
                fprintf(stdout, "All options are optional; if not specified, defaults will be used.\n");
                return 0;
            default:
                fprintf(stdout, "Unrecognized option.\n");
                return 0;
        }
    }
    if(!output_interval) output_interval = steps;


    ALLOC(Abar)
    ALLOC(Rtilde)
    ALLOC(mtilde)
    ALLOC(Utilde)
    ALLOC_N(agg, 13*NN)

    real_t xi;
    ics(Abar, Rtilde, mtilde, Utilde, xi, amp, NN);

    run_sim(NN, steps, amp, rho0, output_interval,
        Abar, Rtilde, mtilde, Utilde, agg, xi);

    free(Abar);
    free(Rtilde);
    free(mtilde);
    free(Utilde);
    free(agg);

    return 0;
}
