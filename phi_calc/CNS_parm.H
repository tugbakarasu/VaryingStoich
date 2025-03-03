#ifndef CNS_PARM_H_
#define CNS_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

#include <cmath>

struct Parm
{
    amrex::Real eos_gamma = 1.4;
    amrex::Real eos_mu = 0.02897;  // mean molecular weight

    amrex::Real cv;
    amrex::Real cp;

    amrex::Real Pr  = 0.72;     // Prandtl number
    amrex::Real Le  = 1.0;      // Lewis number
    amrex::Real C_S = 1.458e-5; // constant in Sutherland's law
    amrex::Real T_S = 110.4;    // Sutherland temperature

    amrex::Real smallr = 1.e-6;
    amrex::Real smallp = 1.e-3;

    amrex::Real minro = 1.e-5;
    amrex::Real minp = 1.e-3;

    amrex::Real maxro = 15.0;
    amrex::Real maxp = 150*101325;

    amrex::Real const_visc_mu = -1.0;
    amrex::Real const_visc_ki = -1.0;
    amrex::Real const_lambda  = -1.0;
    amrex::Real OF_st = 34.32; 

    static constexpr int level_mask_interior   = 0; // valid cells
    static constexpr int level_mask_covered    = 1; // ghost cells covered by valid cells of this level
    static constexpr int level_mask_notcovered = 2; // ghost cells not covered
    static constexpr int level_mask_physbnd    = 3; // outside domain

    // Specific gas constant
    amrex::Real Rsp;

    // Universal gas constant
    const amrex::Real Ru = amrex::Real(8.314462618);

    amrex::Real start_sfoil_time = 0.0;


    // REACTION PARAMETERS

    mutable amrex::Real pre_exp;
    mutable amrex::Real q_nd;
    mutable amrex::Real Ea_nd;
    mutable amrex::Real kappa_0;

   // amrex::Real pre_exp = 0.0;

    // Specifying variable for heat release
    //amrex::Real q_nd = 0.0;
    amrex::Real q_dim = 0.0;

    //amrex::Real Ea_nd = 1.e30;
    amrex::Real Ea_dim = 1.e30;

    amrex::Real Tref = 298.0;
    amrex::Real pref = 1.01325e5;

    AMREX_GPU_DEVICE
    inline void updateCDMParameters(amrex::Real phi) {
       pre_exp = -1.68e13 * pow(phi, 4) + 8.92e13 * pow(phi, 3) - 1.70e14 * pow(phi, 2) +
                 1.41e14 * phi - 4.26e13;

       Ea_nd = -215.90 * pow(phi, 4) + 1349.40 * pow(phi, 3) - 2688.36 * pow(phi, 2) +
                2339.21 * phi - 744.58;

       q_nd  = -14.13 * pow(phi, 4) + 193.02 * pow(phi, 3) - 245.39 * pow(phi, 2) +
                209.56 * phi - 78.67;

       kappa_0 = -1.44e-5 * pow(phi, 4) + 6.50e-5 * pow(phi, 3) - 7.55e-5 * pow(phi, 2) +
                 7.87e-5 * phi - 3.10e-5;

       //pre_exp = -2.85e11 + 5.45e12 * phi - 2.46e13 * pow(phi, 2) +
                 //4.80e13 * pow(phi, 3) - 4.59e13 * pow(phi, 4) +
                 //2.11e13 * pow(phi, 5) - 3.75e12 * pow(phi, 6);

       //Ea_nd =  97.17 - 288.37 * phi + 667.25 * pow(phi, 2) -
                 //537.39 * pow(phi, 3) + 140.32 * pow(phi, 4);

       //q_nd  =  49.34 - 181.94 * phi + 417.35 * pow(phi, 2) -
                //325.84 * pow(phi, 3) + 83.09 * pow(phi, 4);

      // kappa_0 = -6.80e-6 + 2.83e-5 * phi - 3.99e-5 * pow(phi, 2) +
                 //2.49e-5 * pow(phi, 3) - 5.87e-6 * pow(phi, 4);

    }
 
    // Specififying kappa_0 ( kappa = kappa_0 * Cp * T^(0.7) )
    //amrex::Real kappa_0 = 0.0; // units in CGS

    // Some parameters for NSCBC
    amrex::Real sigma = 0.25;
    amrex::Real beta = 1.0;
    amrex::Real eta[5] = {2.0, 2.0, 2.0, 2.0, 2.0};

    // Some target values for NSCBC
    amrex::Real ptarg = 101325.;
    amrex::Real utarg = 0.0;
    amrex::Real vtarg = 0.0;
#if AMREX_SPACEDIM==3
    amrex::Real wtarg = 0.0;
#endif
    amrex::Real Ttarg = 298.;
    amrex::Real Ytarg = 0.0;

    int do_nscbc_lo[AMREX_SPACEDIM] = {0, 0
#if AMREX_SPACEDIM==3        
        , 0
#endif
    };
    int do_nscbc_hi[AMREX_SPACEDIM] = {0, 0
#if AMREX_SPACEDIM==3
        , 0
#endif
    };

    amrex::Real mindt = 1.e-20;

    bool is_visc = false;
    bool is_const_visc = false;

    bool eb_wallloss = false;

    amrex::Real ksolid = 0.0;
    amrex::Real tempsolidwall = 298;

    void Initialize ();
};

#endif
