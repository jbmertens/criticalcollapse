/* INCLUDES */
#include <cmath>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <getopt.h>
#include <string>

typedef double real_t;

int N = 3200;

#define PI 3.14159265358979323846264338328

#define L 21.0
#define OD_SIZE 1.6

#define OUTPUT true

#define W0 (1.0/3.0)  //( P(RHO0)/RHO0 )
#define ALPHA (2.0/3.0/(1 + W0))
#define T0 (ALPHA * std::sqrt(3 / (8*PI*rho0)))
#define RH (T0 / ALPHA)

#define TOL (1.0e-7) // integration tolerance


#define ALLOC(arr_n) \
    real_t *arr_n; \
    arr_n = (real_t *) malloc(N * ((long long) sizeof(real_t)));

#define AUX_ALLOC(aux_n) \
    aux_t aux_n; \
    aux_n.dRtildedA = (real_t *) malloc(N * ((long long) sizeof(real_t))); \
    aux_n.dmtildedA = (real_t *) malloc(N * ((long long) sizeof(real_t))); \
    aux_n.gamma = (real_t *) malloc(N * ((long long) sizeof(real_t))); \
    aux_n.rho = (real_t *) malloc(N * ((long long) sizeof(real_t))); \
    aux_n.P = (real_t *) malloc(N * ((long long) sizeof(real_t))); \
    aux_n.dPdA = (real_t *) malloc(N * ((long long) sizeof(real_t))); \
    aux_n.phi = (real_t *) malloc(N * ((long long) sizeof(real_t))); \
    aux_n.m2oR = (real_t *) malloc(N * ((long long) sizeof(real_t)));

#define AUX_FREE(aux_n) \
    free(aux_n.dRtildedA);\
    free(aux_n.dmtildedA);\
    free(aux_n.gamma);\
    free(aux_n.rho);\
    free(aux_n.P);\
    free(aux_n.dPdA);\
    free(aux_n.phi);\
    free(aux_n.m2oR);

typedef struct {
    real_t *dRtildedA;
    real_t *dmtildedA;

    real_t *gamma;
    real_t *rho;

    real_t *P;
    real_t *dPdA;

    real_t *phi;

    real_t *m2oR;
} aux_t;


void dfdA(real_t *f, real_t *A, real_t *dfdA)
{
    dfdA[0] = 0;
    dfdA[N-1] = 0;

    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<N-1; i++)
    {
        dfdA[i] = 0.5* (
            (f[i+1]-f[i]) / (A[i+1]-A[i])
            + (f[i]-f[i-1]) / (A[i]-A[i-1])
        );
    }
}

real_t Prho(real_t rho)
{
    return W0*rho;
}

real_t dPdrho(real_t rho)
{
    return W0;
}

void dPdAstg(real_t *R, real_t *m, real_t *A, real_t *dPdA)
{
    int i=0;

    real_t P_stg = 0, A_stg = 0;

    for(i=0; i<N-1; i++)
    {
        real_t A_stg_prev = A_stg;
        A_stg = (A[i]+A[i+1])/2;

        real_t P_stg_prev = P_stg;
        real_t R_stg = (R[i]+R[i+1])/2;
        real_t m_stg = (m[i]+m[i+1])/2;
        real_t rho_stg = m_stg + A_stg*R_stg*(m[i+1] - m[i])
                            / 3 / (A[i+1]*R[i+1] - A[i]*R[i]);
        P_stg = Prho(rho_stg);

        dPdA[i] = (P_stg - P_stg_prev)/(A_stg - A_stg_prev);
    }

    dPdA[0] = 0;
    dPdA[N-1] = 0;
}


void aux_pop(real_t *Rtilde, real_t *mtilde, real_t *Utilde,
    real_t *Abar, real_t xi, aux_t aux)
{
    int i=0;

    dfdA(Rtilde, Abar, aux.dRtildedA);
    dfdA(mtilde, Abar, aux.dmtildedA);

    real_t *gamma = aux.gamma;
    real_t *m2oR = aux.m2oR;
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<N; i++)
    {
        real_t e2ax = std::exp(2 * (1 - ALPHA) * xi);
        real_t AR = Abar[i] * Rtilde[i];
        gamma[i] = std::sqrt( e2ax + AR*AR * (Utilde[i]*Utilde[i] - mtilde[i]) );
        m2oR[i] = Rtilde[i]*Rtilde[i]*mtilde[i]*Abar[i]*Abar[i]/e2ax;
    }


    real_t *rho = aux.rho;
#pragma omp parallel for default(shared) private(i)
    for(i=1; i<N-1; i++)
    {
        rho[i] = mtilde[i] + Abar[i]*Rtilde[i]*(mtilde[i+1] - mtilde[i-1])
                / 3 / (Abar[i+1]*Rtilde[i+1] - Abar[i-1]*Rtilde[i-1]);
    }
    rho[0] = mtilde[0];
    rho[N-1] = mtilde[N-1] + Abar[N-1]*Rtilde[N-1]*(mtilde[N-1] - mtilde[N-2])
                / 3 / (Abar[N-1]*Rtilde[N-1] - Abar[N-2]*Rtilde[N-2]);


    real_t *P = aux.P;
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<N; i++)
        P[i] = Prho(rho[i]);

    dPdAstg(Rtilde, mtilde, Abar, aux.dPdA);

    real_t *phi = aux.phi;
    phi[N-1] = -1.5*ALPHA*dPdrho(rho[N-1])*std::log(rho[N-1]);
    phi[N-2] = phi[N-1] + (P[N-1] - P[N-2])/(P[N-2] + rho[N-2]);
    for(i=N-3; i>=0; i--)
        phi[i] = (P[i+2] - P[i])/(P[i+1] + rho[i+1]) + phi[i+2];

}

void k_calc(real_t *Rtilde, real_t *kR_p, real_t *kR_f,
    real_t *mtilde, real_t *km_p, real_t *km_f,
    real_t *Utilde, real_t *kU_p, real_t *kU_f,
    real_t *Abar, aux_t aux, real_t xi, real_t deltaxi,
    real_t kcoeff)
{
    // Compute Runge-Kutta "k" coefficients
    int i=0;

    // Store intermediate values in _f register
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<N; i++)
    {
        kR_f[i] = Rtilde[i] + kR_p[i]*deltaxi*kcoeff;
        km_f[i] = mtilde[i] + km_p[i]*deltaxi*kcoeff;
        kU_f[i] = Utilde[i] + kU_p[i]*deltaxi*kcoeff;
    }
    xi = xi + deltaxi*kcoeff;

    // Populate auxiliary field values
    aux_pop(kR_f, km_f, kU_f, Abar, xi, aux);

    // fill _f registers with final values
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<N; i++)
    {
        real_t R = kR_f[i];
        real_t m = km_f[i];
        real_t U = kU_f[i];
        real_t ep = std::exp(aux.phi[i]);
        real_t AR_prime = R + Abar[i]*aux.dRtildedA[i];

        real_t dPdAoA = 0;
        if(i == 0)
            dPdAoA = 5.0/3.0*dPdrho(aux.rho[0])
                *(km_f[1] + km_f[1] - 2*km_f[0]) / Abar[1] / Abar[1];
        else
            dPdAoA = aux.dPdA[i] / Abar[i];

        kR_f[i] = ALPHA*R*(U*ep - 1);
        km_f[i] = 2*m - 3*ALPHA*U*ep*(aux.P[i] + m);
        kU_f[i] = U - ALPHA*ep*(
            aux.gamma[i]*aux.gamma[i]*dPdAoA / (R*AR_prime*(aux.rho[i] + aux.P[i]))
            + (2*U*U + m + 3*aux.P[i])/2
        );
    }

}


void write_output(real_t *Rtilde, real_t *mtilde, real_t *Utilde,
    real_t *Abar, real_t xi, std::ofstream & output)
{
    AUX_ALLOC(aux)
    aux_pop(Rtilde, mtilde, Utilde, Abar, xi, aux);
    int i=0;

    for(i=0; i<N; i++) output << Rtilde[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << mtilde[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << Utilde[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << Abar[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << aux.dRtildedA[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << aux.dmtildedA[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << aux.gamma[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << aux.rho[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << aux.P[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << aux.dPdA[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << aux.phi[i] << " ";
    output << "\n";
    for(i=0; i<N; i++) output << aux.m2oR[i] << " ";
    output << "\n";

    AUX_FREE(aux)
}




int main(int argc, char **argv)
{

    int steps = 1000;
    real_t amp = 0.3, rho0 = 1.0;

    static struct option long_options[] =
    {
        {"steps", required_argument, 0, 's'},
        {"rho0",  required_argument, 0, 'r'},
        {"amp",   required_argument, 0, 'a'},
        {"N",     required_argument, 0, 'N'},
        {"help",  no_argument,       0, 'h'}
    };

    int c = 0;
    while(1)
    {
        int option_index = 0;
        c = getopt_long(argc, argv, "s:r:a:N:h", long_options, &option_index);
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
                N = std::stoi(optarg);
                break;
            case 'h':
            case '?':
                fprintf(stderr, "usage: %s \n", argv[0]);
                fprintf(stderr, "All options are optional; if not specified defaults will be used.\n");
                return 0;
            default:
                fprintf(stderr, "Unrecognized option.\n");
                return 0;
        }
    }

    std::cout << "\n============\nRunning sim.\n============\n";
    std::cout << "\n";
    std::cout << "Using "<<steps<<" steps with "<<N<<" gridpoints.\n";
    std::cout << "  amp = "<<amp<<"\n";
    std::cout << "  rho0 = "<<rho0<<"\n";

    ALLOC(zeros)
    ALLOC(Abar) ALLOC(Rtilde) ALLOC(mtilde) ALLOC(Utilde)
    ALLOC(kR1) ALLOC(kR2) ALLOC(kR3) ALLOC(kR4) ALLOC(Rnew)
    ALLOC(km1) ALLOC(km2) ALLOC(km3) ALLOC(km4) ALLOC(mnew)
    ALLOC(kU1) ALLOC(kU2) ALLOC(kU3) ALLOC(kU4) ALLOC(Unew)

    AUX_ALLOC(aux)

    real_t xi = 0.0;

    // initial field values
    int i=0;
#pragma omp parallel for default(shared) private(i)
    for(i=0; i<N; i++)
    {
        zeros[i] = 0.0;
        Abar[i] = i*L/N;
        real_t delta0 = amp * std::exp(-Abar[i]*Abar[i] / 2 / OD_SIZE / OD_SIZE);
        real_t delta0P = amp * delta0 * 2 * (-1 / 2 / OD_SIZE / OD_SIZE ) * Abar[i];
        mtilde[i] = 1 + delta0;
        Utilde[i] = 1 - ALPHA * delta0 / 2;
        Rtilde[i] = 1 - ALPHA / 2 * (delta0 + W0 * Abar[i] * delta0P / (1 + 3*W0) );
    }


    // main integration loop
    std::ofstream output;
    output.open("output.dat", std::ios::out | std::ios::trunc);
    int s=0;
    real_t deltaxi = 3.67e-5;
    for(s=0; s<=steps; s++)
    {
        // Run upkeep

            // # Stop running if it becomes clear a BH won't form.
            // if(self.BH_wont_form() == True):
            //     self.timer.stop("adap_run_steps")
            //     return -2

            // self.timer.start("delta_calc")
            // if(self.delta == -1):
            //     r = self.rho(self.R, self.m)
            //     pos = zero_crossing(self.Abar, (self.Abar - 1/np.exp((self.alpha-1) * self.xi) / self.R))
            //     if(pos > 0 and np.interp(pos, self.Abar, r) < 1):
            //         self.delta = np.interp(pos, self.Abar, self.m) - 1
            // self.timer.stop("delta_calc")

            // self.m2oR_prev = self.m2oR
            // self.m2oR = self.R**2 * self.m * self.Abar**2 * np.exp(2 * (self.alpha-1) * self.xi)
            // self.exc_pos = find_exc_pos(self.m2oR)
            // if(self.exc_pos > 0 and self.BH_term == True):
            //     print("Horizon is found, MS run will be terminated! Finished at step", self.step)
            //     self.timer.stop("adap_run_steps")
            //     return -1

            #pragma omp critical
            if(s%1000==0 or s==steps)
                write_output(Rtilde, mtilde, Utilde, Abar, xi, output);

        // Integration details

        k_calc(Rtilde, zeros, kR1,  mtilde, zeros, km1,
            Utilde, zeros, kU1,  Abar, aux, xi, deltaxi,  0.0);

        k_calc(Rtilde, kR1, kR2,  mtilde, km1, km2,
            Utilde, kU1, kU2,  Abar, aux, xi, deltaxi,  0.5);

        k_calc(Rtilde, kR2, kR3,  mtilde, km2, km3,
            Utilde, kU2, kU3,  Abar, aux, xi, deltaxi,  0.75);

#pragma omp parallel for default(shared) private(i)
        for(i=0; i<N; i++)
        {
            Rnew[i] = Rtilde[i] + deltaxi/9*(2*kR1[i] + 3*kR2[i] + 4*kR3[i] );
            mnew[i] = mtilde[i] + deltaxi/9*(2*km1[i] + 3*km2[i] + 4*km3[i] );
            Unew[i] = Utilde[i] + deltaxi/9*(2*kU1[i] + 3*kU2[i] + 4*kU3[i] );
        }

        k_calc(Rnew, zeros, kR4,  mnew, zeros, km4,
            Unew, zeros, kU4,  Abar, aux, xi, deltaxi,  1.0);

        real_t E_R_max = 0, E_m_max = 0, E_U_max = 0,
               err_R_max = 0, err_m_max = 0, err_U_max = 0;
        for(i=0; i<N; i++)
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
            for(i=0; i<N; i++)
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
        for(i=0; i<N; i++)
            if(std::isnan(Rtilde[i]))
                hasnan = true;
        if(deltaxi < 1.0e-11 || deltaxi > 1.0 || hasnan)
        {
            std::cout << "Error at step "<<s<<". q was "<<q<<"\n";
            std::cout << "Errors were "<<err_R_max<<", "<<err_m_max<<", "<<err_U_max<<"\n";
            std::cout << "            "<<E_R_max<<", "<<E_m_max<<", "<<E_U_max<<"\n";
            std::cout << "deltaxi was "<<deltaxi<<"\n";
            write_output(Rtilde, mtilde, Utilde, Abar, xi, output);
            break;
        }

    }

    std::cout << "\n============\nDone running.\n============\n";
    std::cout << "Final xi after step "<<s<<" was "<< xi <<", deltaxi was "<<deltaxi<<".\n";

    output.close();

    free(zeros);
    free(Abar); free(Rtilde); free(mtilde); free(Utilde);
    free(kR1); free(kR2); free(kR3); free(kR4); free(Rnew);
    free(km1); free(km2); free(km3); free(km4); free(mnew);
    free(kU1); free(kU2); free(kU3); free(kU4); free(Unew);
    AUX_FREE(aux)

    return 0;
}
