#include "cns_prob_parm.H"
#include "CNS.H"
#include "CNS_index_macros.H"

#include <AMReX_Arena.H>

ProbParm::ProbParm ()
{
// Calculate stoichiometric mass fractions
    Y_fuel_st = 1.0 / (1.0 + OF_st);
    Y_oxid_st = OF_st / (1.0 + OF_st);
}

ProbParm::~ProbParm ()
{
}
