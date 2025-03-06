#ifndef FCT_DIFFUSION_K_H_
#define FCT_DIFFUSION_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_CONSTANTS.H>
#include <cmath>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_diffcoef (int i, int j, int k,
              amrex::Array4<amrex::Real const> const& q,
              amrex::Array4<amrex::Real> const& coefs,
              Parm const& parm) noexcept
{
    using amrex::Real;

    if (q(i,j,k,QTEMP) < 0.)
    {
//        std::cout << "Computing diff coeffs at " << IntVect(i,j,k) <<
//             " with  temp = " << q(i,j,k,UTEMP)  << std::endl;
        amrex::Abort("Negative temperature sent to cns_diffcoef_eb");
    }

    coefs(i,j,k,0) = parm.kappa_0 * parm.Pr * std::pow(q(i,j,k,UTEMP),Real(0.7));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_constcoef (int i, int j, int k,
              amrex::Array4<amrex::Real> const& coefs,
              Parm const& parm) noexcept
{
    coefs(i,j,k,0) = parm.const_visc_mu;
    // coefs(i,j,k,CXI)  = parm.const_visc_ki;
    // coefs(i,j,k,CLAM) = parm.const_lambda;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_phys_diff_x (int i, int j, int k,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& uo,
               amrex::Array4<amrex::Real const> const& coeffs,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
               amrex::Array4<amrex::Real> const& fx,
               int do_reac, Parm const& parm) noexcept
{
    using amrex::Real;

    Real  dTdx = (q(i,j,k,QTEMP)-q(i-1,j,k,QTEMP))*dxinv[0];
    AMREX_D_TERM(Real  dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];,
                 Real  dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];,
                 Real  dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];);
    Real  dudy = (q(i,j+1,k,QU)+q(i-1,j+1,k,QU)-q(i,j-1,k,QU)-q(i-1,j-1,k,QU))*(0.25*dxinv[1]);
    Real  dvdy = (q(i,j+1,k,QV)+q(i-1,j+1,k,QV)-q(i,j-1,k,QV)-q(i-1,j-1,k,QV))*(0.25*dxinv[1]);
#if (AMREX_SPACEDIM == 2)
    Real  divu = dudx + dvdy;
#else
    Real  dudz = (q(i,j,k+1,QU)+q(i-1,j,k+1,QU)-q(i,j,k-1,QU)-q(i-1,j,k-1,QU))*(0.25*dxinv[2]);
    Real  dwdz = (q(i,j,k+1,QW)+q(i-1,j,k+1,QW)-q(i,j,k-1,QW)-q(i-1,j,k-1,QW))*(0.25*dxinv[2]);
    Real  divu = dudx + dvdy + dwdz;
#endif

    Real  muf = 0.5*(coeffs(i,j,k,0)+coeffs(i-1,j,k,0));
    // Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i-1,j,k,CXI));

    AMREX_D_TERM(Real  tauxx = muf*(2.0*dudx-(2.0/3.0)*divu);,
                 Real  tauxy = muf*(dudy+dvdx);,
                 Real  tauxz = muf*(dudz+dwdx););

    fx(i,j,k,URHO) = Real(0.0);
    AMREX_D_TERM(fx(i,j,k,UMX)   = tauxx;,
                 fx(i,j,k,UMY)   = tauxy;,
                 fx(i,j,k,UMZ)   = tauxz;);
    fx(i,j,k,UEDEN) = 0.5*( (q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx
                    +       (q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy
#if (AMREX_SPACEDIM == 3)
                    +       (q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz
#endif
                    +       ((coeffs(i,j,k,0) +coeffs(i-1,j,k,0))*parm.cp/parm.Pr)*dTdx );

    if(do_reac == 1){

        Real dYdx_fuel = dxinv[0]*( (uo(i,j,k,URHOY_F)/uo(i,j,k,URHO)) - (uo(i-1,j,k,URHOY_F)/uo(i-1,j,k,URHO)) );

        Real dYdx_oxid = dxinv[0]*( (uo(i,j,k,URHOY_O)/uo(i,j,k,URHO)) - (uo(i-1,j,k,URHOY_O)/uo(i-1,j,k,URHO)) );
        
        Real rhoD = 0.5*(coeffs(i-1,j,k,0) + coeffs(i,j,k,0)) / ( parm.Pr * parm.Le );

        fx(i, j, k, URHOY_F) = rhoD * dYdx_fuel;
        fx(i, j, k, URHOY_O) = rhoD * dYdx_oxid;
        
    }

}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_phys_diff_y (int i, int j, int k, 
            amrex::Array4<amrex::Real const> const& q,
            amrex::Array4<amrex::Real const> const& uo,
            amrex::Array4<amrex::Real const> const& coeffs,
            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
            amrex::Array4<amrex::Real> const& fy,
            int do_reac, Parm const& parm) noexcept
{
    using amrex::Real;

    Real  dTdy = (q(i,j,k,QTEMP)-q(i,j-1,k,QTEMP))*dxinv[1];
    AMREX_D_TERM(Real  dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];,
                 Real  dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];,
                 Real  dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];);

    Real  dudx = (q(i+1,j,k,QU)+q(i+1,j-1,k,QU)-q(i-1,j,k,QU)-q(i-1,j-1,k,QU))*(0.25*dxinv[0]);
    Real  dvdx = (q(i+1,j,k,QV)+q(i+1,j-1,k,QV)-q(i-1,j,k,QV)-q(i-1,j-1,k,QV))*(0.25*dxinv[0]);
#if (AMREX_SPACEDIM == 2)
    Real  divu = dudx + dvdy;
#else
    Real  dvdz = (q(i,j,k+1,QV)+q(i,j-1,k+1,QV)-q(i,j,k-1,QV)-q(i,j-1,k-1,QV))*(0.25*dxinv[2]);
    Real  dwdz = (q(i,j,k+1,QW)+q(i,j-1,k+1,QW)-q(i,j,k-1,QW)-q(i,j-1,k-1,QW))*(0.25*dxinv[2]);
    Real  divu = dudx + dvdy + dwdz;
#endif

    Real  muf  = 0.5*(coeffs(i,j,k,0)+coeffs(i,j-1,k,0));
    // Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i,j-1,k,CXI));
    AMREX_D_TERM(Real  tauyy = muf*(2.0*dvdy-(2.0/3.0)*divu);,
                 Real  tauxy = muf*(dudy+dvdx);,
                 Real  tauyz = muf*(dwdy+dvdz););

    fy(i,j,k,URHO) = Real(0.0);
    AMREX_D_TERM(fy(i,j,k,UMX)   = tauxy;,
                 fy(i,j,k,UMY)   = tauyy;,
                 fy(i,j,k,UMZ)   = tauyz;);
    fy(i,j,k,UEDEN) = 0.5*( (q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy
                    +       (q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy
#if (AMREX_SPACEDIM == 3)
                    +       (q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz
#endif
                    +       ((coeffs(i,j,k,0) +coeffs(i,j-1,k,0))*parm.cp/parm.Pr)*dTdy );

    if(do_reac == 1){

        Real dYdy_fuel = dxinv[1]*( (uo(i,j,k,URHOY_F)/uo(i,j,k,URHO)) - (uo(i,j-1,k,URHOY_F)/uo(i,j-1,k,URHO)) );

        Real dYdy_oxid = dxinv[1]*( (uo(i,j,k,URHOY_O)/uo(i,j,k,URHO)) - (uo(i,j-1,k,URHOY_O)/uo(i,j-1,k,URHO)) );

        Real rhoD = 0.5*( coeffs(i,j,k,0) + coeffs(i,j-1,k,0) ) / (parm.Pr * parm.Le);

        fy(i,j,k,URHOY_F) = rhoD * dYdy_fuel;
        fy(i,j,k,URHOY_O) = rhoD * dYdy_oxid;

    }

}

#if (AMREX_SPACEDIM == 3)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_phys_diff_z (int i, int j, int k,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& uo,
               amrex::Array4<amrex::Real const> const& coeffs,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
               amrex::Array4<amrex::Real> const& fz,
               int do_reac, Parm const& parm) noexcept
{
    using amrex::Real;

    Real  dTdz = (q(i,j,k,QTEMP)-q(i,j,k-1,QTEMP))*dxinv[2];
    
    Real  dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    Real  dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    Real  dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];

    Real  dudx = (q(i+1,j,k,QU)+q(i+1,j,k-1,QU)-q(i-1,j,k,QU)-q(i-1,j,k-1,QU))*(0.25*dxinv[0]);
    Real  dvdy = (q(i,j+1,k,QV)+q(i,j+1,k-1,QV)-q(i,j-1,k,QV)-q(i,j-1,k-1,QV))*(0.25*dxinv[1]);

    Real  dwdx = (q(i+1,j,k,QW)+q(i+1,j,k-1,QW)-q(i-1,j,k,QW)-q(i-1,j,k-1,QW))*(0.25*dxinv[0]); 
    Real  dwdy = (q(i,j+1,k,QW)+q(i,j+1,k-1,QW)-q(i,j-1,k,QW)-q(i,j-1,k-1,QW))*(0.25*dxinv[1]);

    Real  divu = dudx + dvdy + dwdz;

    Real  muf  = 0.5*(coeffs(i,j,k,0)+coeffs(i,j,k-1,0));
    
    // Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i,j,k-1,CXI));
    Real  tauxz = muf*(dudz+dwdx);
    Real  tauyz = muf*(dvdz+dwdy);
    Real  tauzz = muf*(2.0*dwdz - (2.0/3.0)*divu);

    fz(i,j,k,URHO) = Real(0.0);
    
    fz(i,j,k,UMX)   = tauxz;
    fz(i,j,k,UMY)   = tauyz;
    fz(i,j,k,UMZ)   = tauzz;

    fz(i,j,k,UEDEN) = 0.5*( (q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz
                    +       (q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz
#if (AMREX_SPACEDIM == 3)
                    +       (q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz
#endif
                    +       ((coeffs(i,j,k,0) +coeffs(i,j,k-1,0))*parm.cp/parm.Pr)*dTdz );

    if(do_reac == 1){

        Real dYdz_fuel = dxinv[2]*( (uo(i,j,k,URHOY_F)/uo(i,j,k,URHO)) - (uo(i,j,k-1,URHOY_F)/uo(i,j,k-1,URHO)) );

        Real dYdz_oxid = dxinv[2]*( (uo(i,j,k,URHOY_O)/uo(i,j,k,URHO)) - (uo(i,j,k-1,URHOY_O)/uo(i,j,k-1,URHO)) );

        Real rhoD = 0.5*( coeffs(i,j,k,0) + coeffs(i,j,k-1,0) ) / (parm.Pr * parm.Le);

        fz(i,j,k,URHOY_F) = rhoD * dYdz_fuel;
        fz(i,j,k,URHOY_O) = rhoD * dYdz_oxid;
    }
}

#endif

#endif
