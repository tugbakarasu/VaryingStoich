#ifndef CNS_K_H_
#define CNS_K_H_

#include "CNS_index_macros.H"
#include <AMReX_FArrayBox.H>
#include <limits>
#include <cmath>

#include "cns_prob.H"
#include "CNS_parm.H"
#include "CNS_K.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real
cns_estdt (int i, int j, int k, amrex::Array4<Real const> const& state,
           amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
           Parm const& parm) noexcept
{
    using amrex::Real;

    Real rho = state(i,j,k,URHO);
    Real rhoinv = Real(1.0)/amrex::max(rho,parm.smallr);
    Real mx  = state(i,j,k,UMX);
    Real my  = state(i,j,k,UMY);
#if AMREX_SPACEDIM==2
    Real velmod = rhoinv*std::sqrt( mx*mx + my*my );
#endif
#if AMREX_SPACEDIM==3
    Real mz  = state(i,j,k,UMZ);
    Real velmod = rhoinv*std::sqrt( mx*mx + my*my + mz*mz );
#endif
    Real ei  = state(i,j,k,UEINT);
    Real p = amrex::max((parm.eos_gamma-Real(1.0))*ei, parm.smallp);
    Real cs = std::sqrt(parm.eos_gamma*p*rhoinv);
#if AMREX_SPACEDIM==2
    Real dt_return = amrex::min(dx[0],dx[1])/(amrex::Math::abs(velmod)+cs);
#endif
#if AMREX_SPACEDIM==3
    Real dt_return = amrex::min(dx[0],amrex::min(dx[1],dx[2]))/(amrex::Math::abs(velmod)+cs);
#endif

    if(parm.is_visc){
        Real nu = rhoinv * parm.kappa_0 * parm.Pr * std::pow(state(i,j,k,UTEMP), Real(0.7));
        dt_return = amrex::min(dt_return, 0.5 * dx[0]*dx[0] / nu);
    }

    return dt_return;

}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
cns_compute_temperature (int i, int j, int k, amrex::Array4<amrex::Real> const& u,
                         Parm const& parm, int do_sootfoil, amrex::Real cur_time) noexcept
{
    using amrex::Real;

    // Correction for negative densities
    u(i,j,k,URHO) = amrex::max(parm.minro, u(i,j,k,URHO));
    u(i,j,k,URHO) = amrex::min(parm.maxro, u(i,j,k,URHO));
    Real rhoinv = Real(1.0)/u(i,j,k,URHO);
    Real mx = u(i,j,k,UMX);
    Real my = u(i,j,k,UMY);

#if (AMREX_SPACEDIM == 2)
    Real kineng = Real(0.5) * rhoinv * (mx*mx+my*my);
#else
    Real mz = u(i,j,k,UMZ);
    Real kineng = Real(0.5) * rhoinv * (mx*mx+my*my+mz*mz);
#endif
    Real pre  = amrex::max(parm.minp, (parm.eos_gamma-1.0) * ( u(i,j,k,UEDEN) - kineng ));
    pre  = amrex::min(pre, parm.maxp);
    u(i,j,k,UEDEN) = (pre/(parm.eos_gamma-1.0)) + kineng;

#if (AMREX_SPACEDIM == 2)
    kineng = Real(0.5) * rhoinv * (mx*mx+my*my);
    u(i,j,k,UEINT) = u(i,j,k,UEDEN) - Real(0.5) * rhoinv * (mx*mx+my*my);
#else
    kineng = Real(0.5) * rhoinv * (mx*mx+my*my+mz*mz);
    u(i,j,k,UEINT) = u(i,j,k,UEDEN) - Real(0.5) * rhoinv * (mx*mx+my*my+mz*mz);
#endif
    u(i,j,k,UTEMP) = rhoinv * u(i,j,k,UEINT) * (Real(1.0)/parm.cv);

    u(i,j,k,URHOY_FUEL) = amrex::min(u(i,j,k,URHOY_FUEL),u(i,j,k,URHO));
    u(i,j,k,URHOY_FUEL) = std::fabs(amrex::max(u(i,j,k,URHOY_FUEL),Real(0.0)));
    if(u(i,j,k,URHOY_FUEL) < Real(1.e-10)) u(i,j,k,URHOY_FUEL) = Real(0.0);

    u(i,j,k,URHOY_OXID) = amrex::min(u(i,j,k,URHOY_OXID),u(i,j,k,URHO));
    u(i,j,k,URHOY_OXID) = std::fabs(amrex::max(u(i,j,k,URHOY_OXID),Real(0.0)));
    if(u(i,j,k,URHOY_OXID) < Real(1.e-10)) u(i,j,k,URHOY_OXID) = Real(0.0);

    u(i,j,k,URHOY_PROD) = amrex::min(u(i,j,k,URHOY_PROD),u(i,j,k,URHO));
    u(i,j,k,URHOY_PROD) = std::fabs(amrex::max(u(i,j,k,URHOY_PROD),Real(0.0)));
    if(u(i,j,k,URHOY_PROD) < Real(1.e-10)) u(i,j,k,URHOY_PROD) = Real(0.0);



    if(do_sootfoil == 1 && cur_time > parm.start_sfoil_time ){
        u(i,j,k,SFOIL) = amrex::max(u(i,j,k,SFOIL), pre);
    }

    if(u(i,j,k,URHO) < Real(0.0) || u(i,j,k,UTEMP) < Real(0.0) || u(i,j,k,UEINT) < Real(0.0)){
        amrex::Abort(" Negative density or temperature or energy density reached in cns_compute_temperature");
    }
}

#endif
