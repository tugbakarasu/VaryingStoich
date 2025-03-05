
#include <CNS_parm.H>

void Parm::Initialize ()
{
    // constexpr amrex::Real Ru = amrex::Real(8.314462618);
    Rsp = Ru / eos_mu;
    cv = Rsp / ( (eos_gamma-amrex::Real(1.0)));
    cp = eos_gamma * Rsp / ( (eos_gamma-amrex::Real(1.0)));
}
