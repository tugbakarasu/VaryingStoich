#ifndef FCT_ADVANCE_BOX_H
#define FCT_ADVANCE_BOX_H

#include "AMReX.H"
#include "cns_prob_parm.H"
#include <AMReX_REAL.H>
using namespace amrex;

using amrex::Real;

AMREX_GPU_DEVICE
void phiAndYReact(Real yO, Real yF, Real yP, Real& phi, Real& Y, const ProbParm* prob_parm);

// Declare Calculate_CDM_Parameters function
AMREX_GPU_DEVICE
void Calculate_CDM_Parameters(Real phi, Real& pre_exp_out, Real& Ea_nd_out, 
                              Real& q_nd_out, Real& kappa_0_out);

#endif // FCT_ADVANCE_BOX_H