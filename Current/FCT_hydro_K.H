#ifndef FCT_HYDRO_K_H_
#define FCT_HYDRO_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_ctoprim (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Array4<amrex::Real const> const& u,
             amrex::Array4<amrex::Real> const& q,
             Parm const& parm) noexcept
{
    using amrex::Real;

    Real rho = amrex::max(u(i,j,k,URHO),parm.minro);
    rho = amrex::min(rho,parm.maxro);
    Real rhoinv = Real(1.0)/rho;
    AMREX_D_TERM(Real ux = u(i,j,k,UMX)*rhoinv;,
                 Real uy = u(i,j,k,UMY)*rhoinv;,
                 Real uz = u(i,j,k,UMZ)*rhoinv;);

    AMREX_D_TERM(q(i,j,k,QU) = ux;,
                 q(i,j,k,QV) = uy;,
                 q(i,j,k,QW) = uz;);

#if (AMREX_SPACEDIM == 2)
    Real kineng = Real(0.5)*rho*(ux*ux+uy*uy);
#else
    Real kineng = Real(0.5)*rho*(ux*ux+uy*uy+uz*uz);
#endif
    Real ei = u(i,j,k,UEDEN) - kineng;
    if (ei <= Real(0.0)) ei = u(i,j,k,UEINT);

    Real p = amrex::max((parm.eos_gamma-Real(1.0))*ei,parm.minp);
    p = amrex::min(p,parm.maxp);
    q(i,j,k,QPRES) = p;

    // q(i,j,k,QEINT) is e, not (rho e)
    ei *= rhoinv;

    rho = amrex::max(uo(i,j,k,URHO),parm.minro);
    rho = amrex::min(rho,parm.maxro);
    rhoinv = Real(1.0)/rho;
    
    AMREX_D_TERM(ux = uo(i,j,k,UMX)*rhoinv;,
                 uy = uo(i,j,k,UMY)*rhoinv;,
                 uz = uo(i,j,k,UMZ)*rhoinv;);

#if (AMREX_SPACEDIM == 2)
    kineng = Real(0.5)*rho*(ux*ux+uy*uy);
#else
    kineng = Real(0.5)*rho*(ux*ux+uy*uy+uz*uz);
#endif

    ei = uo(i,j,k,UEDEN) - kineng;
    if (ei <= Real(0.0)) ei = uo(i,j,k,UEINT);
    q(i,j,k,QRHO) = rho;
    q(i,j,k,QEINT) = ei;
    q(i,j,k,QCS) = std::sqrt(parm.eos_gamma*p*rhoinv);
    q(i,j,k,QTEMP) = ei/parm.cv;

    if( q(i,j,k,QRHO) < Real(0.0) || q(i,j,k,QEINT) < Real(0.0)
        || q(i,j,k,QPRES) < Real(0.0) || q(i,j,k,QTEMP) < Real(0.0)
        || q(i,j,k,QCS) < Real(0.0) ){
        amrex::Abort("Negative density/pressure/temperature/sound-speed/internal energy sent to fct_ctoprim()");
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_flux_to_dudt (int i, int j, int k, int n,
                  amrex::Array4<amrex::Real> const& dudt,
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fx,
                               amrex::Array4<amrex::Real const> const& fy,
                               amrex::Array4<amrex::Real const> const& fz),
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
#if (AMREX_SPACEDIM == 2)
    dudt(i,j,k,n) = dxinv[0] * (fx(i,j,k,n) - fx(i+1,j,k,n))
        +           dxinv[1] * (fy(i,j,k,n) - fy(i,j+1,k,n));
#else
    dudt(i,j,k,n) = dxinv[0] * (fx(i,j,k,n) - fx(i+1,j,k,n))
        +           dxinv[1] * (fy(i,j,k,n) - fy(i,j+1,k,n))
        +           dxinv[2] * (fz(i,j,k,n) - fz(i,j,k+1,n));
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_con_flux_x (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fcx,
             amrex::Array4<amrex::Real> const& u0,
             amrex::Array4<amrex::Real const> const& uo) noexcept
{
    using amrex::Real;

    Real ux = Real(0.5)*(q(i-1,j,k,QU) + q(i,j,k,QU));
    Real pint = Real(0.5)*(q(i-1,j,k,QPRES) + q(i,j,k,QPRES));
    Real puint = Real(0.5)*( (q(i-1,j,k,QPRES)*q(i-1,j,k,QU)) + (q(i,j,k,QPRES)*q(i,j,k,QU)) );
    fcx(i,j,k,URHO)  = ux*Real(0.5)*( uo(i-1,j,k,URHO) + uo(i,j,k,URHO) );
    
    AMREX_D_TERM(fcx(i,j,k,UMX)   = pint + ux*Real(0.5)*( uo(i-1,j,k,UMX) + uo(i,j,k,UMX) );,
                 fcx(i,j,k,UMY)   = ux*Real(0.5)*( uo(i-1,j,k,UMY) + uo(i,j,k,UMY) );,
                 fcx(i,j,k,UMZ)   = ux*Real(0.5)*( uo(i-1,j,k,UMZ) + uo(i,j,k,UMZ) ););

    fcx(i,j,k,UEDEN) = puint + (ux*Real(0.5)*( uo(i-1,j,k,UEDEN) + uo(i,j,k,UEDEN) )); 
    
    fcx(i,j,k,URHOY_F) = ux*Real(0.5)*( uo(i-1,j,k,URHOY_F) + uo(i,j,k,URHOY_F) ); 
    fcx(i,j,k,URHOY_O) = ux*Real(0.5)*( uo(i-1,j,k,URHOY_O) + uo(i,j,k,URHOY_O) );
    fcx(i,j,k,URHOY_P) = ux*Real(0.5)*( uo(i-1,j,k,URHOY_P) + uo(i,j,k,URHOY_P) );
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_con_flux_y (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fcy,
             amrex::Array4<amrex::Real> const& u0,
             amrex::Array4<amrex::Real const> const& uo) noexcept
{
    using amrex::Real;

    Real uy = Real(0.5)*(q(i,j-1,k,QV) + q(i,j,k,QV));
    Real pint = Real(0.5)*(q(i,j-1,k,QPRES) + q(i,j,k,QPRES));
    Real pvint = Real(0.5)*(q(i,j-1,k,QPRES)*q(i,j-1,k,QV) + q(i,j,k,QPRES)*q(i,j,k,QV));
    fcy(i,j,k,URHO)  = uy*Real(0.5)*(uo(i,j-1,k,URHO) + uo(i,j,k,URHO));
    
    AMREX_D_TERM(fcy(i,j,k,UMX) = uy*Real(0.5)*(uo(i,j-1,k,UMX) + uo(i,j,k,UMX));,
                 fcy(i,j,k,UMY) = pint + uy*Real(0.5)*(uo(i,j-1,k,UMY) + uo(i,j,k,UMY));,
                 fcy(i,j,k,UMZ) = uy*Real(0.5)*(uo(i,j-1,k,UMZ) + uo(i,j,k,UMZ)););

    fcy(i,j,k,UEDEN) = pvint + uy*Real(0.5)*(uo(i,j-1,k,UEDEN) + uo(i,j,k,UEDEN)); 
    
    fcy(i,j,k,URHOY_F)  = uy*Real(0.5)*(uo(i,j-1,k,URHOY_F) + uo(i,j,k,URHOY_F));
    fcy(i,j,k,URHOY_O)  = uy*Real(0.5)*(uo(i,j-1,k,URHOY_O) + uo(i,j,k,URHOY_O));
    fcy(i,j,k,URHOY_P)  = uy*Real(0.5)*(uo(i,j-1,k,URHOY_P) + uo(i,j,k,URHOY_P));
}

#if AMREX_SPACEDIM==3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_con_flux_z (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fcz,
             amrex::Array4<amrex::Real> const& u0,
             amrex::Array4<amrex::Real const> const& uo) noexcept
{
    using amrex::Real;

    Real uz = Real(0.5)*(q(i,j,k-1,QW) + q(i,j,k,QW));
    Real pint = Real(0.5)*(q(i,j,k-1,QPRES) + q(i,j,k,QPRES));
    Real pwint = Real(0.5)*(q(i,j,k-1,QPRES)*q(i,j,k-1,QW) + q(i,j,k,QPRES)*q(i,j,k,QW));
    fcz(i,j,k,URHO)  = uz*Real(0.5)*(uo(i,j,k-1,URHO) + uo(i,j,k,URHO));
    fcz(i,j,k,UMX)   = uz*Real(0.5)*(uo(i,j,k-1,UMX) + uo(i,j,k,UMX));
    fcz(i,j,k,UMY)   = uz*Real(0.5)*(uo(i,j,k-1,UMY) + uo(i,j,k,UMY));
    fcz(i,j,k,UMZ)   = pint + uz*Real(0.5)*(uo(i,j,k-1,UMZ) + uo(i,j,k,UMZ));
    fcz(i,j,k,UEDEN) = pwint + uz*Real(0.5)*(uo(i,j,k-1,UEDEN) + uo(i,j,k,UEDEN)); 
    
    fcz(i,j,k,URHOY_F) = uz*Real(0.5)*(uo(i,j,k-1,URHOY_F) + uo(i,j,k,URHOY_F));
    fcz(i,j,k,URHOY_O) = uz*Real(0.5)*(uo(i,j,k-1,URHOY_O) + uo(i,j,k,URHOY_O));
    fcz(i,j,k,URHOY_P) = uz*Real(0.5)*(uo(i,j,k-1,URHOY_P) + uo(i,j,k,URHOY_P));
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_diff_flux_x (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fdx,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Real dxinv, amrex::Real dt,
             const amrex::Real nudiff) noexcept
{
    using amrex::Real;

    Real ux = Real(0.5)*(q(i-1,j,k,QU) + q(i,j,k,QU));
    Real epsx = dt*dxinv*ux;
    Real nuxx = nudiff + (Real(1.0/3.0)*epsx*epsx);
    for(int n = URHO; n < NEQNS; ++n){
        fdx(i,j,k,n) = nuxx*(uo(i,j,k,n) - uo(i-1,j,k,n));
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_diff_flux_y (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fdy,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Real dyinv, amrex::Real dt,
             const amrex::Real nudiff) noexcept
{
    using amrex::Real;

    Real uy = Real(0.5)*(q(i,j-1,k,QV) + q(i,j,k,QV));
    Real epsy = dt*dyinv*uy;
    Real nuyy = nudiff + (Real(1.0/3.0)*epsy*epsy);
    for(int n = URHO; n < NEQNS; ++n){
        fdy(i,j,k,n) = nuyy*(uo(i,j,k,n) - uo(i,j-1,k,n));
    }
}

#if AMREX_SPACEDIM==3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_diff_flux_z (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fdz,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Real dzinv, amrex::Real dt,
             const amrex::Real nudiff) noexcept
{
    using amrex::Real;

    Real uz = Real(0.5)*(q(i,j,k-1,QW) + q(i,j,k,QW));
    Real epsz = dt*dzinv*uz;
    Real nuzz = nudiff + (Real(1.0/3.0)*epsz*epsz);
    for(int n = URHO; n < NEQNS; ++n){
        fdz(i,j,k,n) = nuzz*(uo(i,j,k,n) - uo(i,j,k-1,n));
    }
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_ad_flux_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& q,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real const> const& uo,
                amrex::Array4<amrex::Real> const& ucx,
                amrex::Real dxinv, amrex::Real dt, 
                amrex::Real diff1,
                const amrex::Real mudiff) noexcept
{
    using amrex::Real;

    Real ux      = Real(0.5)*( q(i-1,j,k,QU) + q(i,j,k,QU) );
    Real epsx    = dt*dxinv*ux;
    Real muxx    = (Real(1.0/6.0) - (Real(1.0/6.0)*epsx*epsx)); 
    for(int n = URHO; n < NEQNS; ++n){
        Real resdiff  = mudiff * (uo(i,j,k,n) - uo(i-1,j,k,n));
        flax(i,j,k,n) = diff1*( (muxx*(ucx(i,j,k,n) -  ucx(i-1,j,k,n))) ) - resdiff;
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_ad_flux_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& q,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real const> const& uo,
                amrex::Array4<amrex::Real> const& ucy,
                amrex::Real dyinv, amrex::Real dt,
                amrex::Real diff1,
                const amrex::Real mudiff) noexcept
{
    using amrex::Real;

    Real uy      = Real(0.5)*( q(i,j-1,k,QV) + q(i,j,k,QV) );
    Real epsy    = dt*dyinv*uy;
    Real muyy    = (Real(1.0/6.0) - (Real(1.0/6.0)*epsy*epsy));
    for(int n = URHO; n < NEQNS; ++n){
        Real resdiff = mudiff * (uo(i,j,k,n) - uo(i,j-1,k,n));
        flay(i,j,k,n) = diff1*(muyy*(ucy(i,j,k,n) -  ucy(i,j-1,k,n))) - resdiff ;
    }
}

#if AMREX_SPACEDIM==3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_ad_flux_z (int i, int j, int k,
                amrex::Array4<amrex::Real> const& q,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real const> const& uo,
                amrex::Array4<amrex::Real> const& ucz,
                amrex::Real dzinv, amrex::Real dt,
                amrex::Real diff1,
                const amrex::Real mudiff) noexcept
{
    using amrex::Real;

    Real uz      = Real(0.5)*( q(i,j,k-1,QW) + q(i,j,k,QW) );
    Real epsz    = dt*dzinv*uz;
    Real muzz    = (Real(1.0/6.0) - (Real(1.0/6.0)*epsz*epsz));
    for(int n = URHO; n < NEQNS; ++n){
        Real resdiff = mudiff * (uo(i,j,k,n) - uo(i,j,k-1,n));
        flaz(i,j,k,n) = diff1*(muzz*(ucz(i,j,k,n) -  ucz(i,j,k-1,n))) - resdiff ;
    }
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_prelimit_ad_flux_x (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& udx) noexcept
{
    using amrex::Real;

    Real fltmp    = amrex::Math::abs(flax(i,j,k,n));
    Real sgn      = amrex::Math::copysign(1.0, udx(i,j,k,n) - udx(i-1,j,k,n));
    flax(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udx(i-1,j,k,n) - udx(i-2,j,k,n)),
                                   sgn*(udx(i+1,j,k,n) - udx(i,j,k,n))  ));     
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_prelimit_ad_flux_y (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& udy) noexcept
{
    using amrex::Real;

    Real fltmp = amrex::Math::abs(flay(i,j,k,n));
    Real sgn = amrex::Math::copysign(1.0, udy(i,j,k,n) - udy(i,j-1,k,n));
    flay(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udy(i,j-1,k,n) - udy(i,j-2,k,n)),
                                   sgn*(udy(i,j+1,k,n) - udy(i,j,k,n))  ));
}

#if AMREX_SPACEDIM==3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_prelimit_ad_flux_z (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real> const& udz) noexcept
{
    using amrex::Real;

    Real fltmp = amrex::Math::abs(flaz(i,j,k,n));
    Real sgn = amrex::Math::copysign(1.0, udz(i,j,k,n) - udz(i,j,k-1,n));
    flaz(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udz(i,j,k-1,n) - udz(i,j,k-2,n)),
                                   sgn*(udz(i,j,k+1,n) - udz(i,j,k,n))  ));
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_compute_frac_fluxes (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& flay,
#if AMREX_SPACEDIM==3
                amrex::Array4<amrex::Real> const& flaz,
#endif
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout,
                amrex::Array4<amrex::Real> const& ud) noexcept
{
    using amrex::Real;

    Real fintmp  = amrex::max(flax(i,j,k,n),Real(0.0)) - amrex::min(flax(i+1,j,k,n),Real(0.0))
                 + amrex::max(flay(i,j,k,n),Real(0.0)) - amrex::min(flay(i,j+1,k,n),Real(0.0))
#if AMREX_SPACEDIM==3
                 + amrex::max(flaz(i,j,k,n),Real(0.0)) - amrex::min(flaz(i,j,k+1,n),Real(0.0))
#endif
                 ;

    Real fouttmp = amrex::max(flax(i+1,j,k,n),Real(0.0)) - amrex::min(flax(i,j,k,n),Real(0.0))
                 + amrex::max(flay(i,j+1,k,n),Real(0.0)) - amrex::min(flay(i,j,k,n),Real(0.0))
#if AMREX_SPACEDIM==3
                 + amrex::max(flaz(i,j,k+1,n),Real(0.0)) - amrex::min(flaz(i,j,k,n),Real(0.0))
#endif
                 ;

    Real umin    = amrex::min(ud(i-1,j,k,n),ud(i,j-1,k,n),ud(i,j,k,n),
                              ud(i+1,j,k,n),ud(i,j+1,k,n));
    Real umax    = amrex::max(ud(i-1,j,k,n),ud(i,j-1,k,n),ud(i,j,k,n),
                              ud(i+1,j,k,n),ud(i,j+1,k,n)); 

#if AMREX_SPACEDIM==3
    umin = amrex::min(umin, ud(i,j,k-1,n), ud(i,j,k+1,n));
    umax = amrex::max(umax, ud(i,j,k-1,n), ud(i,j,k+1,n));
#endif

    frin(i,j,k,n)  = (umax - ud(i,j,k,n))/(Real(1E-50) + fintmp);
    frout(i,j,k,n) = (ud(i,j,k,n) - umin)/(Real(1E-50) + fouttmp); 
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_correct_fluxes_x (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout) noexcept
{
    using amrex::Real;

    Real fltmp = flax(i,j,k,n);
    if(fltmp >= Real(0.0))
        flax(i,j,k,n) = fltmp*amrex::min(frout(i-1,j,k,n),frin(i,j,k,n),Real(1.0));
    else
        flax(i,j,k,n) = fltmp*amrex::min( frin(i-1,j,k,n),frout(i,j,k,n),Real(1.0));    
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_correct_fluxes_y (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout) noexcept
{
    using amrex::Real;

    Real fltmp = flay(i,j,k,n);
    if(fltmp >= Real(0.0))
        flay(i,j,k,n) = fltmp*amrex::min(frout(i,j-1,k,n),frin(i,j,k,n),Real(1.0));
    else
        flay(i,j,k,n) = fltmp*amrex::min( frin(i,j-1,k,n),frout(i,j,k,n),Real(1.0));
    
}

#if AMREX_SPACEDIM==3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_correct_fluxes_z (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout) noexcept
{
    using amrex::Real;

    Real fltmp = flaz(i,j,k,n);
    if(fltmp >= Real(0.0))
        flaz(i,j,k,n) = fltmp*amrex::min(frout(i,j,k-1,n),frin(i,j,k,n),Real(1.0));
    else
        flaz(i,j,k,n) = fltmp*amrex::min( frin(i,j,k-1,n),frout(i,j,k,n),Real(1.0));
    
}
#endif

#endif
