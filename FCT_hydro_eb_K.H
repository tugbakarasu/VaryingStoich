#ifndef FCT_HYDRO_EB_K_H_
#define FCT_HYDRO_EB_K_H_

#include "CNS_index_macros.H"
#include "CNS.H"
#include "CNS_parm.H"
#include "CNS_hydro_K.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_con_flux_eb_x (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fcx,
             amrex::Array4<amrex::Real> const& u0,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::Array4<amrex::Real const> const& vfrc,
             const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i-1,j,k).isRegular()){
        Real ux    = Real(0.5)*(q(i-1,j,k,QU) + q(i,j,k,QU));
        Real pint  = Real(0.5)*(q(i-1,j,k,QPRES) + q(i,j,k,QPRES));
        Real puint = Real(0.5)*(q(i-1,j,k,QPRES)*q(i-1,j,k,QU) + q(i,j,k,QPRES)*q(i,j,k,QU));
        fcx(i,j,k,URHO)  = ux*Real(0.5)*(uo(i-1,j,k,URHO) + uo(i,j,k,URHO));
        AMREX_D_TERM(fcx(i,j,k,UMX)   = pint + ux*Real(0.5)*(uo(i-1,j,k,UMX) + uo(i,j,k,UMX));,
                     fcx(i,j,k,UMY)   = ux*Real(0.5)*(uo(i-1,j,k,UMY) + uo(i,j,k,UMY));,
                     fcx(i,j,k,UMZ)   = ux*Real(0.5)*(uo(i-1,j,k,UMZ) + uo(i,j,k,UMZ)););
        fcx(i,j,k,UEDEN) = puint + ux*Real(0.5)*(uo(i-1,j,k,UEDEN) + uo(i,j,k,UEDEN));
         
        fcx(i,j,k,URHOY_F) = ux * Real(0.5) * (uo(i-1,j,k,URHOY_F) + uo(i,j,k,URHOY_F));
        fcx(i,j,k,URHOY_O) = ux * Real(0.5) * (uo(i-1,j,k,URHOY_O) + uo(i,j,k,URHOY_O));
        fcx(i,j,k,URHOY_P) = ux * Real(0.5) * (uo(i-1,j,k,URHOY_P) + uo(i,j,k,URHOY_P));
    }else{
        Real fxrho = 0.0, fxmx = 0.0, fxmy = 0.0, fxeden = 0.0;
        Real fxrhoy_fuel = 0.0, fxrhoy_oxid = 0.0, fxrhoy_prod = 0.0;
#if AMREX_SPACEDIM==3
        Real fxmz = 0.0;
#endif
        if(flag(i,j,k).isConnected(-1,0,0) && vfrc(i,j,k) > vfmin && vfrc(i-1,j,k) > vfmin){
            Real ux    = Real(0.5)*(q(i-1,j,k,QU) + q(i,j,k,QU));
            Real pint  = Real(0.5)*(q(i-1,j,k,QPRES) + q(i,j,k,QPRES));
            Real puint = Real(0.5)*(q(i-1,j,k,QPRES)*q(i-1,j,k,QU) + q(i,j,k,QPRES)*q(i,j,k,QU));

            fxrho  = ux*Real(0.5)*(uo(i-1,j,k,URHO) + uo(i,j,k,URHO));
            AMREX_D_TERM(fxmx   = pint + (ux*Real(0.5)*(uo(i-1,j,k,UMX) + uo(i,j,k,UMX)));,
                         fxmy   = ux*Real(0.5)*(uo(i-1,j,k,UMY) + uo(i,j,k,UMY));,
                         fxmz   = ux*Real(0.5)*(uo(i-1,j,k,UMZ) + uo(i,j,k,UMZ)););
            fxeden = puint + ux*Real(0.5)*(uo(i-1,j,k,UEDEN) + uo(i,j,k,UEDEN)); 
            fxrhoy_fuel = ux * Real(0.5) * (uo(i-1,j,k,URHOY_F) + uo(i,j,k,URHOY_F));
            fxrhoy_oxid = ux * Real(0.5) * (uo(i-1,j,k,URHOY_O) + uo(i,j,k,URHOY_O));
            fxrhoy_prod = ux * Real(0.5) * (uo(i-1,j,k,URHOY_P) + uo(i,j,k,URHOY_P));
        }
        fcx(i,j,k,URHO)  = fxrho;
        AMREX_D_TERM(fcx(i,j,k,UMX)   = fxmx;,
                     fcx(i,j,k,UMY)   = fxmy;,
                     fcx(i,j,k,UMZ)   = fxmz;);
        fcx(i,j,k,UEDEN) = fxeden;
        
        fcx(i,j,k,URHOY_F) = fxrhoy_fuel;
        fcx(i,j,k,URHOY_O) = fxrhoy_oxid;
        fcx(i,j,k,URHOY_P) = fxrhoy_prod;
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_con_flux_eb_y (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fcy,
             amrex::Array4<amrex::Real> const& u0,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::Array4<amrex::Real const> const& vfrc,
             const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i,j-1,k).isRegular()){
        Real uy    = Real(0.5)*( q(i,j-1,k,QV) + q(i,j,k,QV) );
        Real pint  = Real(0.5)*( q(i,j-1,k,QPRES) + q(i,j,k,QPRES) );
        Real pvint = Real(0.5)*( (q(i,j-1,k,QPRES)*q(i,j-1,k,QV)) + (q(i,j,k,QPRES)*q(i,j,k,QV)) );
        fcy(i,j,k,URHO)  = uy*Real(0.5)*( uo(i,j-1,k,URHO) + uo(i,j,k,URHO) );
        AMREX_D_TERM(fcy(i,j,k,UMX)   = uy*Real(0.5)*( uo(i,j-1,k,UMX) + uo(i,j,k,UMX) );,
                     fcy(i,j,k,UMY)   = pint + uy*Real(0.5)*( uo(i,j-1,k,UMY) + uo(i,j,k,UMY) );,
                     fcy(i,j,k,UMZ)   = uy*Real(0.5)*( uo(i,j-1,k,UMZ) + uo(i,j,k,UMZ) ););
        fcy(i,j,k,UEDEN) = pvint + uy*Real(0.5)*( uo(i,j-1,k,UEDEN) + uo(i,j,k,UEDEN) ); 
        
        fcy(i,j,k,URHOY_F) = uy * Real(0.5) * (uo(i,j-1,k,URHOY_F) + uo(i,j,k,URHOY_F));
        fcy(i,j,k,URHOY_O) = uy * Real(0.5) * (uo(i,j-1,k,URHOY_O) + uo(i,j,k,URHOY_O));
        fcy(i,j,k,URHOY_P) = uy * Real(0.5) * (uo(i,j-1,k,URHOY_P) + uo(i,j,k,URHOY_P));
    }else{
        Real fyrho = 0.0, fymx = 0.0, fymy = 0.0, fyeden = 0.0;
        Real fyrhoy_fuel = 0.0, fyrhoy_oxid = 0.0, fyrhoy_prod = 0.0;
        
#if AMREX_SPACEDIM==3
        Real fymz = 0.0;
#endif
        if(flag(i,j,k).isConnected(0,-1,0) && vfrc(i,j,k) > vfmin && vfrc(i,j-1,k) > vfmin){
            Real uy     = Real(0.5)*( q(i,j-1,k,QV) + q(i,j,k,QV) );
            Real pint   = Real(0.5)*( q(i,j-1,k,QPRES) + q(i,j,k,QPRES) );
            Real pvint  = Real(0.5)*( (q(i,j-1,k,QPRES)*q(i,j-1,k,QV)) 
                        + (q(i,j,k,QPRES)*q(i,j,k,QV)) );

            fyrho  = uy*Real(0.5)*( uo(i,j-1,k,URHO) + uo(i,j,k,URHO) );
            AMREX_D_TERM(fymx   = uy*Real(0.5)*( uo(i,j-1,k,UMX) + uo(i,j,k,UMX) );,
                         fymy   = pint  + ( uy*Real(0.5)*( uo(i,j-1,k,UMY) + uo(i,j,k,UMY) ));,
                         fymz   = uy*Real(0.5)*( uo(i,j-1,k,UMZ) + uo(i,j,k,UMZ) ););
            fyeden = pvint + ( uy*Real(0.5)*( uo(i,j-1,k,UEDEN) + uo(i,j,k,UEDEN) )); 
            
            fyrhoy_fuel = uy * Real(0.5) * (uo(i,j-1,k,URHOY_F) + uo(i,j,k,URHOY_F));
            fyrhoy_oxid = uy * Real(0.5) * (uo(i,j-1,k,URHOY_O) + uo(i,j,k,URHOY_O));
            fyrhoy_prod = uy * Real(0.5) * (uo(i,j-1,k,URHOY_P) + uo(i,j,k,URHOY_P)); 
        }
        fcy(i,j,k,URHO)  = fyrho;
        AMREX_D_TERM(fcy(i,j,k,UMX)   = fymx;,
                     fcy(i,j,k,UMY)   = fymy;,
                     fcy(i,j,k,UMZ)   = fymz;);
        fcy(i,j,k,UEDEN) = fyeden; 
        
        fcy(i,j,k,URHOY_F) = fyrhoy_fuel;
        fcy(i,j,k,URHOY_O) = fyrhoy_oxid;
        fcy(i,j,k,URHOY_P) = fyrhoy_prod;
    }
}

#if AMREX_SPACEDIM==3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_con_flux_eb_z (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fcz,
             amrex::Array4<amrex::Real> const& u0,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::Array4<amrex::Real const> const& vfrc,
             const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i,j,k-1).isRegular()){
        Real uz    = Real(0.5)*(q(i,j,k-1,QW) + q(i,j,k,QW));
        Real pint  = Real(0.5)*(q(i,j,k-1,QPRES) + q(i,j,k,QPRES));
        Real pwint = Real(0.5)*(q(i,j,k-1,QPRES)*q(i,j,k-1,QW) + q(i,j,k,QPRES)*q(i,j,k,QW));
        fcz(i,j,k,URHO)  = uz*Real(0.5)*(uo(i,j,k-1,URHO) + uo(i,j,k,URHO));
        fcz(i,j,k,UMX)   = uz*Real(0.5)*(uo(i,j,k-1,UMX) + uo(i,j,k,UMX));
        fcz(i,j,k,UMY)   = uz*Real(0.5)*(uo(i,j,k-1,UMY) + uo(i,j,k,UMY));
        fcz(i,j,k,UMZ)   = pint  + uz*Real(0.5)*(uo(i,j,k-1,UMZ) + uo(i,j,k,UMZ));
        fcz(i,j,k,UEDEN) = pwint + uz*Real(0.5)*(uo(i,j,k-1,UEDEN) + uo(i,j,k,UEDEN)); 
        
        fcz(i,j,k,URHOY_F) = uz * Real(0.5) * (uo(i,j,k-1,URHOY_F) + uo(i,j,k,URHOY_F));
        fcz(i,j,k,URHOY_O) = uz * Real(0.5) * (uo(i,j,k-1,URHOY_O) + uo(i,j,k,URHOY_O));
        fcz(i,j,k,URHOY_P) = uz * Real(0.5) * (uo(i,j,k-1,URHOY_P) + uo(i,j,k,URHOY_P));
    }else{
        Real fzrho = 0.0, fzmx = 0.0, fzmy = 0.0, fzmz = 0.0, fzeden = 0.0;
        Real fzrhoy_fuel = 0.0, fzrhoy_oxid = 0.0, fzrhoy_prod = 0.0;
        
        if(flag(i,j,k).isConnected(0,0,-1) && vfrc(i,j,k) > vfmin && vfrc(i,j,k-1) > vfmin){
            Real uz    = Real(0.5)*(q(i,j,k-1,QW) + q(i,j,k,QW));
            Real pint  = Real(0.5)*(q(i,j,k-1,QPRES) + q(i,j,k,QPRES));
            Real pwint = Real(0.5)*(q(i,j,k-1,QPRES)*q(i,j,k-1,QW) + q(i,j,k,QPRES)*q(i,j,k,QW));

            fzrho  = uz*Real(0.5)*(uo(i,j,k-1,URHO) + uo(i,j,k,URHO));
            fzmx   = uz*Real(0.5)*(uo(i,j,k-1,UMX)  + uo(i,j,k,UMX));
            fzmy   = uz*Real(0.5)*(uo(i,j,k-1,UMY)  + uo(i,j,k,UMY));
            fzmz   = pint + (uz*Real(0.5)*(uo(i,j,k-1,UMZ) + uo(i,j,k,UMZ)));
            fzeden = pwint + uz*Real(0.5)*(uo(i,j,k-1,UEDEN) + uo(i,j,k,UEDEN)); 
            
            fzrhoy_fuel = uz * Real(0.5) * (uo(i,j,k-1,URHOY_F) + uo(i,j,k,URHOY_F));
            fzrhoy_oxid = uz * Real(0.5) * (uo(i,j,k-1,URHOY_O) + uo(i,j,k,URHOY_O));
            fzrhoy_prod = uz * Real(0.5) * (uo(i,j,k-1,URHOY_P) + uo(i,j,k,URHOY_P));
        }
        fcz(i,j,k,URHO)  = fzrho;
        fcz(i,j,k,UMX)   = fzmx;
        fcz(i,j,k,UMY)   = fzmy;
        fcz(i,j,k,UMZ)   = fzmz;
        fcz(i,j,k,UEDEN) = fzeden;
        
        fcz(i,j,k,URHOY_F) = fzrhoy_fuel;
        fcz(i,j,k,URHOY_O) = fzrhoy_oxid;
        fcz(i,j,k,URHOY_P) = fzrhoy_prod;
    }
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_diff_flux_eb_x (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fdx,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Real dxinv, amrex::Real dt,
             const amrex::Real nudiff,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::Array4<amrex::Real const> const& vfrc,
             amrex::Array4<amrex::Real const> const& apx,
             const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i-1,j,k).isRegular()){
        Real ux = Real(0.5)*( q(i-1,j,k,QU) + q(i,j,k,QU) );
        Real epsx = dt*dxinv*ux;
        Real nuxx = nudiff + (Real(1.0/3.0)*epsx*epsx);
        for(int n = URHO; n < NEQNS; ++n){
            fdx(i,j,k,n) = nuxx*(uo(i,j,k,n) - uo(i-1,j,k,n));
        }
    }else{
        if(flag(i,j,k).isConnected(-1,0,0) && vfrc(i,j,k) > vfmin && vfrc(i-1,j,k) > vfmin){
            Real ux     = Real(0.5)*( q(i-1,j,k,QU) + q(i,j,k,QU) );
            Real vol    = Real(0.5)*( (1.0/vfrc(i-1,j,k)) + (1.0/vfrc(i,j,k)) );
            Real epsx   = dt*dxinv*vol*ux*apx(i,j,k);
            Real nuxx   = nudiff + (Real(1.0/3.0)*epsx*epsx);
            Real volint = 0.5*(vfrc(i,j,k) + vfrc(i-1,j,k));
            for(int n = URHO; n < NEQNS; ++n){
                fdx(i,j,k,n) = nuxx*( uo(i,j,k,n) - uo(i-1,j,k,n) )*volint;
            }
        } 
    }
        
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_diff_flux_eb_y (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fdy,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Real dyinv, amrex::Real dt,
             const amrex::Real nudiff,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::Array4<amrex::Real const> const& vfrc,
             amrex::Array4<amrex::Real const> const& apy,
             const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i,j-1,k).isRegular()){
        Real uy = Real(0.5)*(q(i,j-1,k,QV) + q(i,j,k,QV));
        Real epsy = dt*dyinv*uy;
        Real nuyy = nudiff + (Real(1.0/3.0)*epsy*epsy);
        for(int n = URHO; n < NEQNS; ++n){
            fdy(i,j,k,n) = nuyy*(uo(i,j,k,n) - uo(i,j-1,k,n));
        }
    }else{
        if(flag(i,j,k).isConnected(0,-1,0) && vfrc(i,j,k) > vfmin && vfrc(i,j-1,k) > vfmin){
            Real uy     = Real(0.5)*( q(i,j-1,k,QV) + q(i,j,k,QV) );
            Real vol    = Real(0.5)*( (1.0/vfrc(i,j-1,k)) + (1.0/vfrc(i,j,k)) );
            Real epsy   = dt*dyinv*vol*uy*apy(i,j,k);
            Real nuyy   = nudiff + (Real(1.0/3.0)*epsy*epsy);
            Real volint = 0.5*( vfrc(i,j,k) + vfrc(i,j-1,k) );
            for(int n = URHO; n < NEQNS; ++n){
                fdy(i,j,k,n) = nuyy*( uo(i,j,k,n) - uo(i,j-1,k,n) )*volint;
            }
        }
    }
}

#if AMREX_SPACEDIM==3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_diff_flux_eb_z (int i, int j, int k,
             amrex::Array4<amrex::Real> const& q,
             amrex::Array4<amrex::Real> const& fdz,
             amrex::Array4<amrex::Real const> const& uo,
             amrex::Real dzinv, amrex::Real dt,
             const amrex::Real nudiff,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::Array4<amrex::Real const> const& vfrc,
             amrex::Array4<amrex::Real const> const& apz,
             const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i,j,k-1).isRegular()){
        Real uz = Real(0.5)*(q(i,j,k-1,QW) + q(i,j,k,QW));
        Real epsz = dt*dzinv*uz;
        Real nuzz = nudiff + (Real(1.0/3.0)*epsz*epsz);
        for(int n = URHO; n < NEQNS; ++n){
            fdz(i,j,k,n) = nuzz*(uo(i,j,k,n) - uo(i,j,k-1,n));
        }
    }else{
        if(flag(i,j,k).isConnected(0,0,-1) && vfrc(i,j,k) > vfmin && vfrc(i,j,k-1) > vfmin){
            Real uz     = Real(0.5)*( q(i,j,k-1,QW) + q(i,j,k,QW) );
            Real vol    = Real(0.5)*( (1.0/vfrc(i,j,k-1)) + (1.0/vfrc(i,j,k)) );
            Real epsz   = dt*dzinv*vol*uz*apz(i,j,k);
            Real nuzz   = nudiff + (Real(1.0/3.0)*epsz*epsz);
            Real volint = 0.5*( vfrc(i,j,k) + vfrc(i,j,k-1) );
            for(int n = URHO; n < NEQNS; ++n){
                fdz(i,j,k,n) = nuzz*( uo(i,j,k,n) - uo(i,j,k-1,n) )*volint;
            }
        }
    }
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_ad_flux_eb_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& q,
                amrex::Array4<amrex::Real> const& fax,
                amrex::Array4<amrex::Real const> const& uo,
                amrex::Array4<amrex::Real> const& ucx,
                amrex::Real dxinv, amrex::Real dt, 
                amrex::Real diff1, amrex::Real diffcc,
                const amrex::Real mudiff,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                amrex::Array4<amrex::Real const> const& vfrc,
                amrex::Array4<amrex::Real const> const& apx,
                const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i-1,j,k).isRegular()){
        Real ux      = Real(0.5)*( q(i-1,j,k,QU) + q(i,j,k,QU) );
        Real epsx    = dt*dxinv*ux;
        Real muxx    = (Real(1.0/6.0) - (Real(1.0/6.0)*epsx*epsx)); 
        for(int n = URHO; n < NEQNS; ++n){
            Real resdiff  = mudiff * (uo(i,j,k,n) - uo(i-1,j,k,n));
            fax(i,j,k,n) = diff1*( (muxx*(ucx(i,j,k,n) -  ucx(i-1,j,k,n))) ) - resdiff;
        }
    }else{
        GpuArray<Real,NEQNS> fx;
        for(int n = URHO; n < NEQNS; ++n){
            fx[n] = 0.0;
        }

        if(flag(i,j,k).isConnected(-1,0,0) && vfrc(i,j,k) > vfmin && vfrc(i-1,j,k) > vfmin){
            Real ux      = Real(0.5)*( q(i-1,j,k,QU) + q(i,j,k,QU) );
            Real vol     = Real(0.5)*( (1.0/vfrc(i-1,j,k)) + (1.0/vfrc(i,j,k)) );
            Real epsx    = dt*dxinv*vol*ux*apx(i,j,k);
            Real muxx    = ( Real(1.0/6.0) - (Real(1.0/6.0)*epsx*epsx) ); 
            Real volint  = 0.5*( vfrc(i,j,k) + vfrc(i-1,j,k) );
            for(int n = URHO; n < NEQNS; ++n){
                Real resdiff  = mudiff * ( uo(i,j,k,n) - uo(i-1,j,k,n) )*volint;
                fx[n] = diffcc*volint*( ( muxx*(ucx(i,j,k,n) -  ucx(i-1,j,k,n)) ) ) - resdiff;
            }
        }

        for(int n = URHO; n < NEQNS; ++n){
            fax(i,j,k,n) = fx[n];
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_ad_flux_eb_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& q,
                amrex::Array4<amrex::Real> const& fay,
                amrex::Array4<amrex::Real const> const& uo,
                amrex::Array4<amrex::Real> const& ucy,
                amrex::Real dyinv, amrex::Real dt,
                amrex::Real diff1, amrex::Real diffcc,
                const amrex::Real mudiff,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                amrex::Array4<amrex::Real const> const& vfrc,
                amrex::Array4<amrex::Real const> const& apy,
                const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i,j-1,k).isRegular()){
        Real uy      = Real(0.5)*( q(i,j-1,k,QV) + q(i,j,k,QV) );
        Real epsy    = dt*dyinv*uy;
        Real muyy    = (Real(1.0/6.0) - (Real(1.0/6.0)*epsy*epsy));
        for(int n = URHO; n < NEQNS; ++n){
            Real resdiff = mudiff * (uo(i,j,k,n) - uo(i,j-1,k,n));
            fay(i,j,k,n) = diff1*( muyy*(ucy(i,j,k,n) -  ucy(i,j-1,k,n)) ) - resdiff ;
        }
    }else{
        GpuArray<Real,NEQNS> fy;
        for(int n = URHO; n < NEQNS; ++n){
            fy[n] = 0.0;
        }

        if(flag(i,j,k).isConnected(0,-1,0) && vfrc(i,j,k) > vfmin && vfrc(i,j-1,k) > vfmin){
            Real uy      = Real(0.5)*( q(i,j-1,k,QV) + q(i,j,k,QV) );
            Real vol     = Real(0.5)*( (1.0/vfrc(i,j-1,k)) + (1.0/vfrc(i,j,k)) );
            Real epsy    = dt*dyinv*vol*uy*apy(i,j,k);
            Real muyy    = Real(1.0/6.0) - (Real(1.0/6.0)*epsy*epsy);
            Real volint  = 0.5*( vfrc(i,j,k) + vfrc(i,j-1,k) );
            for(int n = URHO; n < NEQNS; ++n){
                Real resdiff = mudiff * (uo(i,j,k,n) - uo(i,j-1,k,n)) * volint;
                fy[n] = diffcc*volint*(muyy*(ucy(i,j,k,n) -  ucy(i,j-1,k,n))) - resdiff;
            }
        }
    
        for(int n = URHO; n < NEQNS; ++n){
            fay(i,j,k,n) = fy[n];
        }
    }
}

#if AMREX_SPACEDIM==3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_ad_flux_eb_z (int i, int j, int k,
                amrex::Array4<amrex::Real> const& q,
                amrex::Array4<amrex::Real> const& faz,
                amrex::Array4<amrex::Real const> const& uo,
                amrex::Array4<amrex::Real> const& ucz,
                amrex::Real dzinv, amrex::Real dt,
                amrex::Real diff1, amrex::Real diffcc,
                const amrex::Real mudiff,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                amrex::Array4<amrex::Real const> const& vfrc,
                amrex::Array4<amrex::Real const> const& apz,
                const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i,j,k-1).isRegular()){
        Real uz      = Real(0.5)*( q(i,j,k-1,QW) + q(i,j,k,QW) );
        Real epsz    = dt*dzinv*uz;
        Real muzz    = (Real(1.0/6.0) - (Real(1.0/6.0)*epsz*epsz));
        for(int n = URHO; n < NEQNS; ++n){
            Real resdiff = mudiff * (uo(i,j,k,n) - uo(i,j,k-1,n));
            faz(i,j,k,n) = diff1*( muzz*(ucz(i,j,k,n) -  ucz(i,j,k-1,n)) ) - resdiff ;
        }
    }else{
        GpuArray<Real,NEQNS> fz;
        for(int n = URHO; n < NEQNS; ++n){
            fz[n] = 0.0;
        }

        if(flag(i,j,k).isConnected(0,0,-1) && vfrc(i,j,k) > vfmin && vfrc(i,j,k-1) > vfmin){
            Real uz      = Real(0.5)*( q(i,j,k-1,QW) + q(i,j,k,QW) );
            Real vol     = Real(0.5)*( (1.0/vfrc(i,j,k-1)) + (1.0/vfrc(i,j,k)) );
            Real epsz    = dt*dzinv*vol*uz*apz(i,j,k);
            Real muzz    = Real(1.0/6.0) - (Real(1.0/6.0)*epsz*epsz);
            Real volint  = 0.5*( vfrc(i,j,k) + vfrc(i,j-1,k) );
            for(int n = URHO; n < NEQNS; ++n){
                Real resdiff = mudiff * (uo(i,j,k,n) - uo(i,j,k-1,n)) * volint;
                fz[n] = diffcc*volint*(muzz*(ucz(i,j,k,n) -  ucz(i,j,k-1,n))) - resdiff;
            }
        }
    
        for(int n = URHO; n < NEQNS; ++n){
            faz(i,j,k,n) = fz[n];
        }
    }
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_prelimit_ad_flux_eb_x (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& udx,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                amrex::Array4<amrex::Real const> const& vfrc,
                const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if( flag(i,j,k).isRegular()   && flag(i-1,j,k).isRegular()
     && flag(i-2,j,k).isRegular() && flag(i+1,j,k).isRegular() ){
        Real fltmp    = amrex::Math::abs(flax(i,j,k,n));
        Real sgn      = amrex::Math::copysign(1.0, udx(i,j,k,n) - udx(i-1,j,k,n));
        flax(i,j,k,n) = sgn*amrex::max( 0.0, amrex::min( fltmp,
            sgn*(udx(i-1,j,k,n) - udx(i-2,j,k,n)), sgn*(udx(i+1,j,k,n) - udx(i,j,k,n))  ) ); 
    
    }else{
       if(flag(i,j,k).isConnected(-1,0,0) && vfrc(i,j,k) > vfmin && vfrc(i-1,j,k) > vfmin){        
            Real fltmp    = amrex::Math::abs(flax(i,j,k,n));
            Real sgn      = amrex::Math::copysign(1.0, udx(i,j,k,n) - udx(i-1,j,k,n));
        
            if(flag(i-1,j,k).isConnected(-1,0,0) && vfrc(i-1,j,k) > vfmin && vfrc(i-2,j,k) > vfmin)
                fltmp = amrex::min( fltmp, sgn*( udx(i-1,j,k,n) - udx(i-2,j,k,n) )*vfrc(i-1,j,k) );
        
            if(flag(i,j,k).isConnected(1,0,0) && vfrc(i,j,k) > vfmin && vfrc(i+1,j,k) > vfmin)
                fltmp = amrex::min( fltmp, sgn*( udx(i+1,j,k,n) - udx(i,j,k,n) )*vfrc(i,j,k) );
        
            Real fx = sgn * amrex::max(0.0, fltmp);

            flax(i,j,k,n) = fx;
        } 
    }

}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_prelimit_ad_flux_eb_y (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& udy,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                amrex::Array4<amrex::Real const> const& vfrc,
                const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if( flag(i,j,k).isRegular()   && flag(i,j-1,k).isRegular()
    &&  flag(i,j-2,k).isRegular() && flag(i,j+1,k).isRegular() ){
        Real fltmp = amrex::Math::abs(flay(i,j,k,n));
        Real sgn = amrex::Math::copysign(1.0, udy(i,j,k,n) - udy(i,j-1,k,n));
        flay(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udy(i,j-1,k,n) - udy(i,j-2,k,n)),
                                   sgn*(udy(i,j+1,k,n) - udy(i,j,k,n))  ));
    }else{
       if(flag(i,j,k).isConnected(0,-1,0) && vfrc(i,j,k) > vfmin && vfrc(i,j-1,k) > vfmin){
            Real fltmp = amrex::Math::abs(flay(i,j,k,n));
            Real sgn = amrex::Math::copysign(1.0, udy(i,j,k,n) - udy(i,j-1,k,n));

            if(flag(i,j-1,k).isConnected(0,-1,0) && vfrc(i,j-1,k) > vfmin && vfrc(i,j-2,k) > vfmin)
                fltmp = amrex::min(fltmp, sgn*(udy(i,j-1,k,n)-udy(i,j-2,k,n))*vfrc(i,j-1,k));

            if(flag(i,j,k).isConnected(0,1,0) && vfrc(i,j,k) > vfmin && vfrc(i,j+1,k) > vfmin)
                fltmp = amrex::min( fltmp, sgn*(udy(i,j+1,k,n) - udy(i,j,k,n))*vfrc(i,j,k) );

            Real fy = sgn * amrex::max(0.0, fltmp);

            flay(i,j,k,n) = fy;
        } 
    }    
}

#if AMREX_SPACEDIM==3

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_prelimit_ad_flux_eb_z (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real> const& udz,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                amrex::Array4<amrex::Real const> const& vfrc,
                const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if( flag(i,j,k).isRegular()   && flag(i,j,k-1).isRegular()
    &&  flag(i,j,k-2).isRegular() && flag(i,j,k+1).isRegular() ){
        Real fltmp = amrex::Math::abs(flaz(i,j,k,n));
        Real sgn = amrex::Math::copysign(1.0, udz(i,j,k,n) - udz(i,j,k-1,n));
        flaz(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp, sgn*(udz(i,j,k-1,n) - udz(i,j,k-2,n)),
                                   sgn*(udz(i,j,k+1,n) - udz(i,j,k,n))  ));
    }else{
       if(flag(i,j,k).isConnected(0,0,-1) && vfrc(i,j,k) > vfmin && vfrc(i,j,k-1) > vfmin){
            Real fltmp = amrex::Math::abs(flaz(i,j,k,n));
            Real sgn = amrex::Math::copysign(1.0, udz(i,j,k,n) - udz(i,j,k-1,n));

            if(flag(i,j,k-1).isConnected(0,0,-1) && vfrc(i,j,k-1) > vfmin && vfrc(i,j,k-2) > vfmin)
                fltmp = amrex::min(fltmp, sgn*(udz(i,j,k-1,n)-udz(i,j,k-2,n))*vfrc(i,j,k-1));

            if(flag(i,j,k).isConnected(0,0,1) && vfrc(i,j,k) > vfmin && vfrc(i,j,k+1) > vfmin)
                fltmp = amrex::min( fltmp, sgn*(udz(i,j,k+1,n) - udz(i,j,k,n))*vfrc(i,j,k) );

            Real fz = sgn * amrex::max(0.0, fltmp);

            flaz(i,j,k,n) = fz;
        } 
    }    
}

#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_compute_frac_eb_fluxes (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& flay,
#if AMREX_SPACEDIM==3
                amrex::Array4<amrex::Real> const& flaz,
#endif
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout,
                amrex::Array4<amrex::Real> const& ud,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                amrex::Array4<amrex::Real const> const& vfrc,
                const amrex::Real vfmin) noexcept
{
    using amrex::Real;

    if(  flag(i,j,k).isRegular() && flag(i-1,j,k).isRegular() && flag(i+1,j,k).isRegular()
    && flag(i,j-1,k).isRegular() && flag(i,j+1,k).isRegular() 
#if AMREX_SPACEDIM==3
    && flag(i,j,k-1).isRegular() && flag(i,j,k+1).isRegular()
#endif
    ){

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
                        ud(i+1,j,k,n),ud(i,j+1,k,n)
#if AMREX_SPACEDIM==3
                        ,ud(i,j,k-1,n),ud(i,j,k+1,n)
#endif
                        );
        Real umax    = amrex::max(ud(i-1,j,k,n),ud(i,j-1,k,n),ud(i,j,k,n),
                        ud(i+1,j,k,n),ud(i,j+1,k,n)
#if AMREX_SPACEDIM==3
                        ,ud(i,j,k-1,n), ud(i,j,k+1,n)
#endif
                        ); 

        frin(i,j,k,n)  = (umax - ud(i,j,k,n))/(Real(1E-50) + fintmp);
        frout(i,j,k,n) = (ud(i,j,k,n) - umin)/(Real(1E-50) + fouttmp); 
    }else{
        if(!flag(i,j,k).isCovered() && vfrc(i,j,k) > vfmin){
            Real umax    = ud(i,j,k,n);
            Real umin    = ud(i,j,k,n);
            Real fintmp = 0.0;
            Real fouttmp = 0.0;

            if(flag(i,j,k).isConnected(-1,0,0) && vfrc(i-1,j,k) > vfmin){
                fintmp  = amrex::max(flax(i,j,k,n),Real(0.0));
                fouttmp = -amrex::min(flax(i,j,k,n),Real(0.0));

                umin = amrex::min(umin, ud(i-1,j,k,n));
                umax = amrex::max(umax, ud(i-1,j,k,n));
            }

            if(flag(i,j,k).isConnected(1,0,0) && vfrc(i+1,j,k) > vfmin){
                fintmp  = fintmp  - amrex::min(flax(i+1,j,k,n),Real(0.0));
                fouttmp = fouttmp + amrex::max(flax(i+1,j,k,n),Real(0.0));

                umin    = amrex::min(umin, ud(i+1,j,k,n));
                umax    = amrex::max(umax, ud(i+1,j,k,n));
            }

            if(flag(i,j,k).isConnected(0,-1,0) && vfrc(i,j-1,k) > vfmin){
                fintmp  = fintmp  + amrex::max(flay(i,j,k,n),Real(0.0));
                fouttmp = fouttmp - amrex::min(flay(i,j,k,n),Real(0.0));

                umin = amrex::min(umin, ud(i,j-1,k,n));
                umax = amrex::max(umax, ud(i,j-1,k,n));
            }

            if(flag(i,j,k).isConnected(0,1,0) && vfrc(i,j+1,k) > vfmin){
                fintmp  = fintmp  - amrex::min(flay(i,j+1,k,n),Real(0.0));
                fouttmp = fouttmp + amrex::max(flay(i,j+1,k,n),Real(0.0));

                umin    = amrex::min(umin, ud(i,j+1,k,n));
                umax    = amrex::max(umax, ud(i,j+1,k,n));
            }

#if AMREX_SPACEDIM==3
            if(flag(i,j,k).isConnected(0,0,-1) && vfrc(i,j,k-1) > vfmin){
                fintmp  = fintmp  + amrex::max(flaz(i,j,k,n),Real(0.0));
                fouttmp = fouttmp - amrex::min(flaz(i,j,k,n),Real(0.0));

                umin = amrex::min(umin, ud(i,j,k-1,n));
                umax = amrex::max(umax, ud(i,j,k-1,n));
            }

            if(flag(i,j,k).isConnected(0,0,1) && vfrc(i,j,k+1) > vfmin){
                fintmp  = fintmp  - amrex::min(flaz(i,j,k+1,n),Real(0.0));
                fouttmp = fouttmp + amrex::max(flaz(i,j,k+1,n),Real(0.0));

                umin    = amrex::min(umin, ud(i,j,k+1,n));
                umax    = amrex::max(umax, ud(i,j,k+1,n));
            }
#endif

            frin(i,j,k,n)  = (umax - ud(i,j,k,n))*vfrc(i,j,k)/(Real(1E-50) + fintmp);
            frout(i,j,k,n) = (ud(i,j,k,n) - umin)*vfrc(i,j,k)/(Real(1E-50) + fouttmp); 
        }else{
            frin(i,j,k,n)  = 0.0;
            frout(i,j,k,n) = 0.0;
        }
    }

}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_correct_fluxes_eb_x (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flax,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout,
                Array4<Real const> const& apx) noexcept
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
fct_correct_fluxes_eb_y (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flay,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout,
                Array4<Real const> const& apy) noexcept
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
fct_correct_fluxes_eb_z (int i, int j, int k, int n,
                amrex::Array4<amrex::Real> const& flaz,
                amrex::Array4<amrex::Real> const& frin,
                amrex::Array4<amrex::Real> const& frout,
                Array4<Real const> const& apz) noexcept
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
