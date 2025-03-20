#ifndef FCT_NSCBC_H_
#define FCT_NSCBC_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>

#include <AMReX_TagBox.H>
#include <AMReX_EBFArrayBox.H>

#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>

#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "cns_prob_parm.H"

#include <cmath>
#include <algorithm>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
nscbc_outflow_x(
  const amrex::Real s_int[NGROW][NUM_STATE],
  amrex::Real s_ext[NUM_STATE],
  const int i,
  const int j,
#if AMREX_SPACEDIM==3
  const int k,
#endif
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  Parm const& parm,
  amrex::Array4<amrex::Real> const& dat)
{
    using amrex::Real;

    const int* domlo = geomdata.Domain().loVect();
    const int* domhi = geomdata.Domain().hiVect();
    const amrex::Real* dx = geomdata.CellSize();

    // Subsonic outflow BC
    // Obtain primitive variables
    // Get the local Mach number
    Real ux = s_int[0][UMX] / s_int[0][URHO];
    Real ss = std::sqrt( parm.eos_gamma * parm.Rsp * s_int[0][UTEMP] );
    Real mach = (1.0/ss)*std::sqrt( ( ux*ux + ((s_int[0][UMY]/s_int[0][URHO])*(s_int[0][UMY]/s_int[0][URHO])) ) );

    if(mach >= 1.0){
        for(int nc = URHO; nc < NUM_STATE; ++nc){
            s_ext[nc] = s_int[0][nc];
        } //for(nc)
    }
    else{
        // Define the relaxation coefficient parameter K (Eq 38)
        Real krel = parm.sigma * ss * (1. - mach*mach) 
                    / (geomdata.ProbHi(0) - geomdata.ProbLo(0)); 
        Real tau = 0.0;

//  -------------------START OF TRANSVERSE PART----------------------

        if(std::fabs(Real(1.) - parm.beta) > Real(1.e-3)){
            // Compute the transverse terms
            int idx = domhi[0];
            if(sgn == -1) idx = domlo[0];

            Real dudy = 0.0, dvdy = 0.0, dpdy = 0.0;
            if(j > domlo[1] || j < domhi[1]){
                amrex::IntVect lp1(AMREX_D_DECL(idx, j+1, k));
                amrex::IntVect lm1(AMREX_D_DECL(idx, j-1, k));
                dudy = ( (dat(lp1,UMX)/dat(lp1,URHO)) - (dat(lm1,UMX)/dat(lm1,URHO)) );
                dvdy = ( (dat(lp1,UMY)/dat(lp1,URHO)) - (dat(lm1,UMY)/dat(lm1,URHO)) );
                dpdy = ( dat(lp1,UEINT) - dat(lm1,UEINT) ) * (parm.eos_gamma-1.);
            }//if(j > domlo[1] || j < domhi[1])
            // Use one-sided difference if j == domlo.y or j == domhi.y
            if(j == domlo[1]){
                amrex::IntVect lp0(AMREX_D_DECL(idx, j, k));
                amrex::IntVect lp1(AMREX_D_DECL(idx, j+1, k));
                amrex::IntVect lp2(AMREX_D_DECL(idx, j+2, k));
                dudy = ( -3.0*(dat(lp0,UMX)/dat(lp0,URHO)) 
                        + 4.0*(dat(lp1,UMX)/dat(lp1,URHO)) 
                        - (dat(lp2,UMX)/dat(lp2,URHO)) );
                dvdy = ( -3.0*(dat(lp0,UMY)/dat(lp0,URHO)) 
                         + 4.0*(dat(lp1,UMY)/dat(lp1,URHO)) 
                         - (dat(lp2,UMY)/dat(lp2,URHO)) );
                dpdy = ( parm.eos_gamma - 1. ) 
                     * ( -3.0*dat(lp0,UEINT) + 4.0*dat(lp1,UEINT) - dat(lp2,UEINT) );
            }else if(j == domhi[1]){
                amrex::IntVect lm0(AMREX_D_DECL(idx, j, k));
                amrex::IntVect lm1(AMREX_D_DECL(idx, j-1, k));
                amrex::IntVect lm2(AMREX_D_DECL(idx, j-2, k));
                dudy = ( (dat(lm2,UMX)/dat(lm2,URHO)) - 4.0*(dat(lm1,UMX)/dat(lm1,URHO)) 
                     + 3.0*(dat(lm0,UMX)/dat(lm0,URHO)) );
                dvdy = ( (dat(lm2,UMY)/dat(lm2,URHO)) - 4.0*(dat(lm1,UMY)/dat(lm1,URHO)) 
                     + 3.0*(dat(lm0,UMY)/dat(lm0,URHO)) );
                dpdy = ( parm.eos_gamma - 1.0 ) * 
                       ( dat(lm2,UEINT) - 4.0*dat(lm1,UEINT) + 3.0*dat(lm0,UEINT) );
            } //if(j == domlo[1])

            // Compute the transverse terms if needed
            if(sgn == 1){
                // Outflow boundary at i == domhi[0]
                // The other Li's are obtained from one-sided 2nd order differences
                // Computing tau_1 here
                tau = 0.5*( (s_int[0][UMY]/s_int[0][URHO])*(dpdy - ss*s_int[0][URHO]*dudy) 
                    + (parm.eos_gamma*(parm.eos_gamma-1.)*s_int[0][UEINT]*dvdy) ) / dx[1];            
            }else if(sgn == -1){
                // Outflow boundary at i == domlo[0]
                // COmputing tau_4 here
                tau = 0.5*( (s_int[0][UMY]/s_int[0][URHO])*(dpdy + ss*s_int[0][URHO]*dudy) 
                    + parm.eos_gamma*(parm.eos_gamma-1.)*s_int[0][UEINT]*dvdy ) / dx[1];
            } //if(sgn==1)
        } //if(std::fabs(Real(1.) - parm.beta) > Real(1.e-3))

//  -------------------END OF TRANSVERSE PART----------------------
        // sgc stores the primitive variables of the cell with index (iv[0], iv[1], iv[2])
#if AMREX_SPACEDIM==2
        Real sgc[NPRIMBC] = {0.0, 0.0, 0.0, 0.0};
        // The order of variables is rho, u, v, p
        Real dqdx[4] = {0.0, 0.0, 0.0, 0.0};
        Real Li[4] = {0.0, 0.0, 0.0, 0.0};
#endif
        if(sgn == 1){
            // Boundary at x == Lx
            // Since the boundary is at x = xhi, L1 is modeled
            // The other Li's are obtained from one-sided 2nd order differences 
#if AMREX_SPACEDIM==2
            Li[0] = (krel * ( ((parm.eos_gamma-1.)*s_int[0][UEINT]) - parm.ptarg )) 
                  - ((1. - parm.beta) * tau);
            // Use 2nd order backward difference to compute other Li's
            Li[1] = ux * (0.5 / dx[0])
                  * ( ss*ss*( s_int[2][SRHO] - 4.0*s_int[1][SRHO] + 3.*s_int[0][SRHO] ) 
                  - (parm.eos_gamma-1.)*( s_int[2][UEINT] - 4.*s_int[1][UEINT] + 3.*s_int[0][UEINT] ) );
            Li[2] = ux * (0.5/dx[0]) 
                  * ( (s_int[2][UMY]/s_int[2][URHO]) - 4.*(s_int[1][UMY]/s_int[1][URHO]) + 3.*(s_int[0][UMY]/s_int[0][URHO]) );
            Li[3] = (ux + ss) * (0.5 / dx[0]) 
                  * ( (parm.eos_gamma-1.)*( s_int[2][UEINT] - 4.*s_int[1][UEINT] + 3.*s_int[0][UEINT] ) 
                  + s_int[0][URHO] * ss 
                  *( (s_int[2][UMX]/s_int[2][URHO]) - 4.*(s_int[1][UMX]/s_int[1][URHO]) + 3.*(s_int[0][UMX]/s_int[0][URHO]) )  );

            if(std::fabs(ux) > 0.0){
                dqdx[0] = (1.0/(ss*ss))*( (0.5*Li[0]/(ux-ss)) + (Li[1]/ux) + (0.5*Li[3]/(ss+ux)) );
                dqdx[2] = Li[2] / ux;
            } // if(std::fabs(ux) > 0.0)
            dqdx[1] = (0.5 / (ss*s_int[0][URHO])) * ( (-Li[0]/((ux-ss)))  + (Li[3]/(ss+ux)) );
            dqdx[3] = 0.5 * ( (Li[0] / (ux-ss)) + (Li[3] / (ux+ss)) );

            int del_i = i - domhi[0];
            for(int nc = 0; nc < NPRIMBC; ++nc){
                // qgc : array to store values of 4 ghost cells (domhi[0]+1 to domhi[0]+4)
                Real qgc[4] = {0.0, 0.0, 0.0, 0.0};
                Real sm1 = 0.0, sm0 = 0.0;
                if(nc==0){
                    sm1 = s_int[1][URHO]; sm0 = s_int[0][URHO];
                }else if(nc==1){
                    sm1 = s_int[1][UMX]/s_int[1][URHO]; sm0 = s_int[0][UMX]/s_int[0][URHO];
                }else if(nc==2){
                    sm1 = s_int[1][UMY]/s_int[1][URHO]; sm0 = s_int[0][UMY]/s_int[0][URHO];
                }else{
                    sm1 = (parm.eos_gamma-1.)*s_int[1][UEINT]; sm0 = (parm.eos_gamma-1.)*s_int[0][UEINT];
                }   
                qgc[0] = sm1 + 2.*dx[0]*dqdx[nc];
                qgc[1] = -2.*sm1 - 3.*sm0  + 6.*qgc[0] - 6.*dx[0]*dqdx[nc];
                qgc[2] =  3.*sm1 + 10.*sm0 - 18.*qgc[0] + 6.*qgc[1] + 12.*dx[0]*dqdx[nc];
                qgc[3] = -2.*sm1 - 13.*sm0 + 24.*qgc[0] -12.*qgc[1] + 4.*qgc[2] - 12.*dx[0]*dqdx[nc];
                if(del_i > 4) sgc[nc] = qgc[3];
                else sgc[nc] = qgc[del_i-1];
            } // for(nc)
            // amrex::Print() << "entered nscbc_outflow_x, nscbc_lo = " << parm.do_nscbc_lo[0] 
            //     << ", nscbc_hi = " << parm.do_nscbc_hi[0] << "\n";
#endif
        } // if(sgn == 1)
        else if(sgn == -1){
            // Boundary at x == 0
            // Since the boundary is at x = xlo, L4 is modeled
            // The other Li's are obtained from one-sided 2nd order differences 
#if AMREX_SPACEDIM==2
          Li[3] = (( parm.sigma * ss * (1. - mach*mach) / (geomdata.ProbHi(0) - geomdata.ProbLo(0)) ) * ( s_int[0][SPRES] - parm.ptarg )) 
                - ((1. - parm.beta) * tau);
         // Use 2nd order backward difference to compute other Li's
          Li[1] = ux * (0.5 / dx[0] )
                * ( ss*ss*( -3.*s_int[0][URHO] + 4.0*s_int[1][URHO] - s_int[2][URHO] ) 
                - (parm.eos_gamma-1.)*( -3.*s_int[0][UEINT] + 4.*s_int[1][UEINT] - s_int[2][UEINT] ) );
          Li[2] = ux * (0.5/dx[0]) * ( -3.*(s_int[0][UMY]/s_int[0][URHO]) 
                + 4.*(s_int[1][UMY]/s_int[0][URHO]) 
                - (s_int[2][UMY]/s_int[2][URHO]) );
          Li[0] = (ux - ss) * (0.5 / dx[0]) 
                * ( (parm.eos_gamma-1.)*( -3.*s_int[0][UEINT] + 4.*s_int[1][UEINT] - s_int[2][UEINT] ) 
                - s_int[0][URHO] * ss *( -3.*(s_int[0][UMX]/s_int[0][URHO]) 
                + 4.*(s_int[1][UMX]/s_int[1][URHO]) - (s_int[2][UMX]/s_int[2][URHO]) )  );

          if(std::fabs(ux) > 0.0){
            dqdx[0] = (1.0/(ss*ss))*( (0.5*Li[0]/(ux-ss)) + (Li[1]/ux) + (0.5*Li[3]/(ss+ux)) );
            dqdx[2] = Li[2] / ux;
          } // if(std::fabs(ux) > 0.0)
          dqdx[1] = (0.5 / (ss*s_int[0][SRHO])) * ( (-Li[0]/((ux-ss)))  + (Li[3]/(ss+ux)) );
          dqdx[3] = 0.5 * ( (Li[0] / (ux-ss)) + (Li[3] / (ux+ss)) );

          int del_i = domlo[0] - i;
          for(int nc = 0; nc < NPRIMBC; ++nc){
            // qgc : array to store values of 4 ghost cells (domhi[0]+1 to domhi[0]+4)
            Real qgc[4] = {0.0, 0.0, 0.0, 0.0};
            Real sp1 = 0.0, sp0 = 0.0;
            if(nc==0){
                sp1 = s_int[1][URHO]; sp0 = s_int[0][URHO];
            }else if(nc==1){
                sp1 = s_int[1][UMX]/s_int[1][URHO]; sp0 = s_int[0][UMX]/s_int[0][URHO];
            }else if(nc==2){
                sp1 = s_int[1][UMY]/s_int[1][URHO]; sp0 = s_int[0][UMY]/s_int[0][URHO];
            }else{
                sp1 = (parm.eos_gamma-1.)*s_int[1][UEINT]; sp0 = (parm.eos_gamma-1.)*s_int[0][UEINT];
            } 
            qgc[0] = sp1 - 2.*dx[0]*dqdx[nc];
            qgc[1] = -2.*sp1 - 3.*sp0  + 6.*qgc[0]  + 6.*dx[0]*dqdx[nc];
            qgc[2] =  3.*sp1 + 10.*sp0 - 18.*qgc[0] + 6.*qgc[1] - 12.*dx[0]*dqdx[nc];
            qgc[3] = -2.*sp1 - 13.*sp0 + 24.*qgc[0] -12.*qgc[1] + 4.*qgc[2] + 12.*dx[0]*dqdx[nc];
            if(del_i > 4) sgc[nc] = qgc[3];
            else sgc[nc] = qgc[del_i-1];
          } // for(nc)  
#endif
        } //if(sgn == -1)

#if AMREX_SPACEDIM==2
        s_ext[URHO]  = sgc[0];
        s_ext[UMX]   = s_ext[URHO] * sgc[1];
        s_ext[UMY]   = s_ext[URHO] * sgc[2];
        s_ext[UEDEN] = (sgc[3]/(parm.eos_gamma-1.)) + 0.5*sgc[0]*(sgc[1]*sgc[1] + sgc[2]*sgc[2]);
        s_ext[UEINT] = sgc[3] / (parm.eos_gamma-1.);
        s_ext[URHOY_F] = sgc[0] * s_int[0][URHOY_F] / s_int[0][URHO];
        s_ext[URHOY_O] = sgc[0] * s_int[0][URHOY_O] / s_int[0][URHO];
        s_ext[URHOY_P] = sgc[0] * s_int[0][URHOY_P] / s_int[0][URHO];
        s_ext[UTEMP] = sgc[3] / (parm.Rsp * sgc[0]);
        s_ext[SFOIL] = s_int[0][SFOIL];
#endif
        
    } //if(mach >= 1.0)

}


AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
nscbc_inflow_x(
  const amrex::Real s_int[NGROW][NUM_STATE],
  amrex::Real s_ext[NUM_STATE],
  const int i,
  const int j,
#if AMREX_SPACEDIM==3
  const int k,
#endif
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  Parm const& parm,
  amrex::Array4<amrex::Real> const& dat)
{
    using amrex::Real;

    const int* domlo = geomdata.Domain().loVect();
    const int* domhi = geomdata.Domain().hiVect();
    const amrex::Real* dx = geomdata.CellSize();

    // Subsonic inflow BC in plane normal to x-direction
    // Obtain primitive variables
    // Get the local Mach number
    Real ux = s_int[0][UMX] / s_int[0][URHO];
    Real ss = std::sqrt( parm.eos_gamma * parm.Rsp * s_int[0][UTEMP] );
    Real mach = (1.0/ss)*std::sqrt( ( ux*ux + ((s_int[0][UMY]/s_int[0][URHO])*(s_int[0][UMY]/s_int[0][URHO])) ) );

    Real tau[3] = {0.0, 0.0, 0.0};

//  -------------------START OF TRANSVERSE PART----------------------
    if(std::fabs(Real(1.) - parm.beta) > Real(1.e-3)){
          // Compute the transverse terms
          int idx = domhi[0];
          if(sgn == -1) idx = domlo[0];

          Real dudy = 0.0, dvdy = 0.0, dpdy = 0.0, drdy = 0.0;
          if(j > domlo[1] || j < domhi[1]){
            amrex::IntVect lp1(AMREX_D_DECL(idx, j+1, k));
            amrex::IntVect lm1(AMREX_D_DECL(idx, j-1, k));
            dudy = ( dat(lp1,UMX)/dat(lp1,URHO) - dat(lm1,UMX)/dat(lm1,URHO) );
            dvdy = ( dat(lp1,UMY)/dat(lp1,URHO) - dat(lm1,UMY)/dat(lm1,URHO) );
            dpdy = ( dat(lp1,UEINT) - dat(lm1,UEINT) ) * (parm.eos_gamma-1.);
            drdy = ( dat(lp1,URHO)  - dat(lm1,URHO) );
          }//if(j > domlo[1] || j < domhi[1])
          // Use one-sided difference if j == domlo.y or j == domhi.y
          if(j == domlo[1]){
            amrex::IntVect lp0(AMREX_D_DECL(idx, j, k));
            amrex::IntVect lp1(AMREX_D_DECL(idx, j+1, k));
            amrex::IntVect lp2(AMREX_D_DECL(idx, j+2, k));
            dudy = ( -3.0*(dat(lp0,UMX)/dat(lp0,URHO)) + 4.0*(dat(lp1,UMX)/dat(lp1,URHO)) 
                 - (dat(lp2,UMX)/dat(lp2,URHO)) );
            dvdy = ( -3.0*(dat(lp0,UMY)/dat(lp0,URHO)) + 4.0*(dat(lp1,UMY)/dat(lp1,URHO)) 
                 - (dat(lp2,UMY)/dat(lp2,URHO)) );
            dpdy = ( -3.0*dat(lp0,UEINT) + 4.0*dat(lp1,UEINT) - dat(lp2,UEINT) )* (parm.eos_gamma-1.);
            drdy = ( -3.0*dat(lp0,URHO) + 4.0*dat(lp1,URHO) - dat(lp2,URHO) );
          }else if(j == domhi[1]){
            amrex::IntVect lm0(AMREX_D_DECL(idx, j, k));
            amrex::IntVect lm1(AMREX_D_DECL(idx, j-1, k));
            amrex::IntVect lm2(AMREX_D_DECL(idx, j-2, k));
            dudy = ( (dat(lm2,UMX)/dat(lm2,URHO)) - 4.0*(dat(lm1,UMX)/dat(lm1,URHO)) 
                 + 3.0*(dat(lm0,UMX)/dat(lm0,URHO)) );
            dvdy = ( (dat(lm2,UMY)/dat(lm2,URHO)) - 4.0*(dat(lm1,UMY)/dat(lm1,URHO)) 
                 + 3.0*(dat(lm0,UMY)/dat(lm0,URHO)) );
            dpdy = (parm.eos_gamma-1.0) * ( dat(lm2,UEINT) - 4.0*dat(lm1,UEINT) + 3.0*dat(lm0,UEINT) );
            drdy =  ( dat(lm2,URHO) - 4.0*dat(lm1,URHO) + 3.0*dat(lm0,URHO) );
          }

          // Compute the transverse terms if needed
          if(sgn == 1){
            // Inflow boundary at i == domhi[0]
            // The other Li's are obtained from one-sided 2nd order differences
            // tau[0] is tau1
            tau[0] = (0.5/dx[1])*( (s_int[0][UMY]/s_int[0][URHO])*(dpdy - ss*s_int[0][URHO]*dudy) 
                   + (parm.eos_gamma*(parm.eos_gamma-1.)*s_int[0][UEINT]*dvdy) );           
          }else if(sgn == -1){
            // Inflow boundary at i == domlo[0]
            // tau[0] is tau4
            tau[0] = (0.5/dx[1])*( (s_int[0][UMY]/s_int[0][URHO])*(dpdy + ss*s_int[0][URHO]*dudy) 
                + parm.eos_gamma*(parm.eos_gamma-1.)*s_int[0][UEINT]*dvdy );
          } //if(sgn==1)
          tau[1] = (0.5/dx[1]) * ( ( (s_int[0][UMY]/s_int[0][URHO])*( ss*ss*drdy - dpdy ) )
                   + ( ss*ss*s_int[0][URHO]*dvdy ) 
                   - (parm.eos_gamma * (parm.eos_gamma-1.) * s_int[0][UEINT]*dvdy) ); 
          tau[2] = (0.5/dx[1]) * ( (s_int[0][UMY]/s_int[0][URHO])*dvdy + (dpdy/s_int[0][URHO]) );
        } //if(std::fabs(Real(1.) - parm.beta) > Real(1.e-3)) 

//  -------------------END OF TRANSVERSE PART----------------------

        // sgc stores the primitive variables of the cell with index (i, j, k)
#if AMREX_SPACEDIM==2
        Real sgc[NPRIMBC] = {0.0, 0.0, 0.0, 0.0};
        // The order of variables is rho, u, v, p
        Real dqdx[4] = {0.0, 0.0, 0.0, 0.0};
        Real Li[4]   = {0.0, 0.0, 0.0, 0.0};
#elif AMREX_SPACEDIM==3
        Real sgc[NPRIMBC] = {0.0, 0.0, 0.0, 0.0, 0.0};
        // The order of variables is rho, u, v, p
        Real dqdx[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        Real Li[5]   = {0.0, 0.0, 0.0, 0.0, 0.0};
#endif

        if(sgn == 1){
          // Inflow boundary at x == Lx
          // Since the boundary is at x = xhi, L1, L2, L3 are modeled
          // The other Li's are obtained from one-sided 2nd order differences 
#if AMREX_SPACEDIM==2
          Li[0] = ( parm.eta[0] * s_int[0][URHO] * ss * ss *(1. - mach*mach) 
                * (ux - parm.utarg) / (geomdata.ProbHi(0) - geomdata.ProbLo(0)) )
                - tau[0];
         // Use 2nd order backward difference to compute other Li's
          Real Tloc = (parm.eos_gamma-1.) * s_int[0][UEINT] / (s_int[0][URHO] * parm.Rsp);
          Li[1] = ( parm.eta[1] * s_int[0][URHO] * ss * parm.Rsp 
                * (Tloc - parm.Ttarg) / (geomdata.ProbHi(0) - geomdata.ProbLo(0)) )
                - tau[1];
          Li[2] = (parm.eta[2] * ss * ((s_int[0][UMY]/s_int[0][URHO]) - parm.vtarg) / (geomdata.ProbHi(0) - geomdata.ProbLo(0)))
                - tau[2];
          Li[3] = (ux + ss) * (0.5 / dx[0]) 
                * ( (parm.eos_gamma-1.)*( s_int[2][UEINT] - 4.*s_int[1][UEINT] + 3.*s_int[0][UEINT] ) 
                + s_int[0][URHO] * ss *( (s_int[2][UMX]/s_int[2][URHO]) 
                - 4.*(s_int[1][UMX]/s_int[1][URHO]) + 3.*(s_int[0][UMX]/s_int[0][URHO]) )  );

          if(std::fabs(ux) > 0.0){
            dqdx[0] = (1.0/(ss*ss))*( (0.5*Li[0]/(ux-ss)) + (Li[1]/ux) + (0.5*Li[3]/(ss+ux)) );
            dqdx[2] = Li[2] / ux;
          } // if(std::fabs(ux) > 0.0)
          dqdx[1] = (0.5 / (ss*s_int[0][URHO])) * ( (-Li[0]/(ux-ss))  + (Li[3]/(ss+ux)) );
          dqdx[3] = 0.5 * ( (Li[0] / (ux-ss)) + (Li[3] / (ux+ss)) );
#endif         
          int del_i = i - domhi[0];
          for(int nc = 0; nc < NPRIMBC; ++nc){
            // qgc : array to store values of 4 ghost cells (domhi[0]+1 to domhi[0]+4)
            Real qgc[4] = {0.0, 0.0, 0.0, 0.0};
            Real sm1 = 0.0, sm0 = 0.0;
            if(nc==0){
                sm1 = s_int[1][URHO]; sm0 = s_int[0][URHO];
            }else if(nc==1){
                sm1 = s_int[1][UMX]/s_int[1][URHO]; sm0 = s_int[0][UMX]/s_int[0][URHO];
            }else if(nc==2){
                sm1 = s_int[1][UMY]/s_int[1][URHO]; sm0 = s_int[0][UMY]/s_int[0][URHO];
            }else{
                sm1 = (parm.eos_gamma-1.)*s_int[1][UEINT]; sm0 = (parm.eos_gamma-1.)*s_int[0][UEINT];
            }   
            qgc[0] = sm1 + 2.*dx[0]*dqdx[nc];
            qgc[1] = -2.*sm1 - 3.*sm0  + 6.*qgc[0] - 6.*dx[0]*dqdx[nc];
            qgc[2] =  3.*sm1 + 10.*sm0 - 18.*qgc[0] + 6.*qgc[1] + 12.*dx[0]*dqdx[nc];
            qgc[3] = -2.*sm1 - 13.*sm0 + 24.*qgc[0] -12.*qgc[1] + 4.*qgc[2] - 12.*dx[0]*dqdx[nc];
            if(del_i > 4) sgc[nc] = qgc[3];
            else sgc[nc] = qgc[del_i-1];
          } // for(nc)

        }
        else if(sgn == -1){
          // Subsonic inflow boundary at x == 0
          // Since the boundary is at x = xlo, L2, L3, L4 are modeled
          // The other Li's are obtained from one-sided 2nd order differences 
#if AMREX_SPACEDIM==2
          Li[3] = ( parm.eta[3] * s_int[0][URHO] * ss * ss * (1. - mach*mach) 
                * (ux - parm.utarg) / (geomdata.ProbHi(0) - geomdata.ProbLo(0)) )
                - tau[0];
          Real Tloc = s_int[0][UTEMP];
          Li[1] = ( parm.eta[1] * s_int[0][URHO] * ss * parm.Rsp 
                * (Tloc - parm.Ttarg) / (geomdata.ProbHi(0) - geomdata.ProbLo(0)) )
                - tau[1];
          Li[2] = (parm.eta[2] * ss * ((s_int[0][UMY]/s_int[0][URHO]) - parm.vtarg) / (geomdata.ProbHi(0) - geomdata.ProbLo(0)))
                - tau[2];
          Li[0] = (ux - ss) * (0.5 / dx[0]) 
                * ( (parm.eos_gamma-1.)*( -3.*s_int[0][UEINT] + 4.*s_int[1][UEINT] - s_int[2][UEINT] ) 
                - s_int[0][URHO] * ss *( -3.*(s_int[0][UMX]/s_int[0][URHO]) 
                + 4.*(s_int[1][UMX]/s_int[1][URHO]) - (s_int[2][UMX]/s_int[2][URHO]) )  );

          if(std::fabs(ux) > 0.0){
            dqdx[0] = (1.0/(ss*ss))*( (0.5*Li[0]/(ux-ss)) + (Li[1]/ux) + (0.5*Li[3]/(ss+ux)) );
            dqdx[2] = Li[2] / ux;
          } // if(std::fabs(ux) > 0.0)
          dqdx[1] = (0.5 / (ss*s_int[0][SRHO])) * ( (-Li[0]/(ux-ss))  + (Li[3]/(ss+ux)) );
          dqdx[3] = 0.5 * ( (Li[0] / (ux-ss)) + (Li[3] / (ux+ss)) );
        
          int del_i = domlo[0] - i;
          for(int nc = SRHO; nc < NPRIMBC; ++nc){
            // qgc : array to store values of 4 ghost cells (domhi[0]+1 to domhi[0]+4)
            Real qgc[4] = {0.0, 0.0, 0.0, 0.0};
            Real sp1 = 0.0, sp0 = 0.0;
            if(nc==0){
                sp1 = s_int[1][URHO]; sp0 = s_int[0][URHO];
            }else if(nc==1){
                sp1 = s_int[1][UMX]/s_int[1][URHO]; sp0 = s_int[0][UMX]/s_int[0][URHO];
            }else if(nc==2){
                sp1 = s_int[1][UMY]/s_int[1][URHO]; sp0 = s_int[0][UMY]/s_int[0][URHO];
            }else{
                sp1 = (parm.eos_gamma-1.)*s_int[1][UEINT]; sp0 = (parm.eos_gamma-1.)*s_int[0][UEINT];
            } 
            qgc[0] = sp1 - 2.*dx[0]*dqdx[nc];
            qgc[1] = -2.*sp1 - 3.*sp0  + 6.*qgc[0]  + 6.*dx[0]*dqdx[nc];
            qgc[2] =  3.*sp1 + 10.*sp0 - 18.*qgc[0] + 6.*qgc[1] - 12.*dx[0]*dqdx[nc];
            qgc[3] = -2.*sp1 - 13.*sp0 + 24.*qgc[0] -12.*qgc[1] + 4.*qgc[2] + 12.*dx[0]*dqdx[nc];
            if(del_i > 4) sgc[nc] = qgc[3];
            else sgc[nc] = qgc[del_i-1];
          } // for(nc)

        s_ext[URHO]  = sgc[0];
        s_ext[UMX]   = s_ext[URHO] * sgc[1];
        s_ext[UMY]   = s_ext[URHO] * sgc[2];
        s_ext[UEDEN] = (sgc[3]/(parm.eos_gamma-1.)) + 0.5*sgc[0]*(sgc[1]*sgc[1] + sgc[2]*sgc[2]);
        s_ext[UEINT] = sgc[3] / (parm.eos_gamma-1.);
        s_ext[URHOY_F] = sgc[0] * s_int[0][URHOY_F] / s_int[0][URHO];
        s_ext[URHOY_O] = sgc[0] * s_int[0][URHOY_O] / s_int[0][URHO];
        s_ext[URHOY_P] = sgc[0] * s_int[0][URHOY_P] / s_int[0][URHO];
        s_ext[UTEMP] = sgc[3] / (parm.Rsp * sgc[0]);
        s_ext[SFOIL] = s_int[0][SFOIL];
#endif 
    } // if(sgn == -1)
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
nscbc_outflow_riemann_x(
  const amrex::Real s_int[NGROW][NUM_STATE],
  amrex::Real s_ext[NUM_STATE],
  const int i,
  const int j,
#if AMREX_SPACEDIM==3
  const int k,
#endif
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  Parm const& parm,
  amrex::Array4<amrex::Real> const& dat)
{
    using amrex::Real;

    // Subsonic Outflow BC based on Riemann invariants
    Real pout = parm.ptarg;
    Real vout = s_int[0][UMY] / s_int[0][URHO];
    Real vmodin = std::sqrt( (s_int[0][UMX]*s_int[0][UMX]/(s_int[0][URHO]*s_int[0][URHO]))
                + (s_int[0][UMY]*s_int[0][UMY]/(s_int[0][URHO]*s_int[0][URHO])) );
    Real ssin = std::sqrt(parm.eos_gamma * parm.Rsp * s_int[0][UTEMP]);
    Real rm1in = vmodin + (2.*ssin/(parm.eos_gamma-1.));
    Real entin = (parm.eos_gamma-1.)*s_int[0][UEINT] 
                / std::pow(s_int[0][URHO], parm.eos_gamma);

    s_ext[URHO]  = std::pow(pout / entin, 1.0/parm.eos_gamma);
    Real ssout   = std::sqrt(parm.eos_gamma*pout/s_ext[URHO]);
    Real qout    = rm1in - (2.0 * ssout / (parm.eos_gamma-1.));
    s_ext[UMX]   = s_ext[URHO] * std::sqrt(qout*qout - vout*vout);
    s_ext[UMY]   = s_ext[URHO] * vout;
    s_ext[UEINT] = pout / (parm.eos_gamma-1.);
    s_ext[UEDEN] = s_ext[UEINT] + ((0.5 / s_ext[URHO]) * ( s_ext[UMX]*s_ext[UMX] + s_ext[UMY]*s_ext[UMY] ));
    s_ext[UTEMP] = pout / (parm.Rsp * s_ext[URHO]);
    s_ext[URHOY_F] = s_ext[URHO] * s_int[0][URHOY_F] / s_int[0][URHO];
    s_ext[URHOY_O] = s_ext[URHO] * s_int[0][URHOY_O] / s_int[0][URHO];
    s_ext[URHOY_P] = s_ext[URHO] * s_int[0][URHOY_P] / s_int[0][URHO];
    s_ext[SFOIL] = 0.0;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
nscbc_inflow_riemann_x(
  const amrex::Real s_int[NGROW][NUM_STATE],
  amrex::Real s_ext[NUM_STATE],
  const int i,
  const int j,
#if AMREX_SPACEDIM==3
  const int k,
#endif
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  Parm const& parm,
  amrex::Array4<amrex::Real> const& dat)
{
    using amrex::Real;

    // Subsonic inflow BC using Riemann invariants
    Real ux = s_int[0][UMX] / s_int[0][URHO];
    Real ss = std::sqrt( parm.eos_gamma * parm.Rsp * s_int[0][UTEMP] );
    Real vmod0 = std::sqrt( ux*ux + (s_int[0][UMY]*s_int[0][UMY]/(s_int[0][URHO]*s_int[0][URHO])) );
    Real vmodinf = std::sqrt( parm.utarg*parm.utarg + parm.vtarg*parm.vtarg );
    Real ssinf = std::sqrt(parm.eos_gamma * parm.Rsp * parm.Ttarg);
    Real mach = (1.0/ss)*vmod0;
        
    Real Mainf = vmodinf / ssinf;
    Real pstinf = parm.ptarg * std::pow(1. + (0.5*(parm.eos_gamma-1.)*Mainf*Mainf), 
                  parm.eos_gamma/(parm.eos_gamma-1.) );
    Real alfa = 0.0;

    Real rm1inf = vmodinf + (2.0*ssinf / (parm.eos_gamma-1.));
    // rm2in : Riemann invariant obtianed from interior points
    Real rm2in  = vmod0 - (2.0*ss / (parm.eos_gamma-1.));
    Real q0  = 0.5 * (rm1inf + rm2in);
    Real ss0 = 0.25 * (parm.eos_gamma-1.) * (rm1inf - rm2in);
    Real Ma0 = q0 / ss0;
    Real pres0 = pstinf / std::pow(1. + (0.5*(parm.eos_gamma-1.)*Ma0*Ma0), parm.eos_gamma / (parm.eos_gamma-1.));

    s_ext[URHO]   = parm.eos_gamma * pres0 / (ss0 * ss0);
    s_ext[UMX]    = s_ext[URHO] * q0 * cos(alfa);
    s_ext[UMY]    = s_ext[URHO] * q0 * sin(alfa);
    Real kineng   = 0.5 * (s_ext[UMX]*s_ext[UMX] + s_ext[UMY]*s_ext[UMY]) / s_ext[URHO];
    s_ext[UEDEN]  = (pres0 / (parm.eos_gamma-1.)) + kineng;
    s_ext[UEINT]  = pres0 / (parm.eos_gamma-1.);
    s_ext[UTEMP]  = pres0 / (parm.Rsp * s_ext[URHO]);
    s_ext[SFOIL]  = 0.0;
    s_ext[URHOY_F]  = s_ext[URHO] * s_int[0][URHOY_F] / s_int[0][URHO];
    s_ext[URHOY_O]  = s_ext[URHO] * s_int[0][URHOY_O] / s_int[0][URHO];
    s_ext[URHOY_P]  = s_ext[URHO] * s_int[0][URHOY_P] / s_int[0][URHO];
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
nscbc_inflow_lele_x(
  const amrex::Real s_int[NGROW][NUM_STATE],
  amrex::Real s_ext[NUM_STATE],
  const int i,
  const int j,
#if AMREX_SPACEDIM==3
  const int k,
#endif
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  Parm const& parm,
  amrex::Array4<amrex::Real> const& dat)
  {
    using amrex::Real;

    const int* domlo = geomdata.Domain().loVect();
    const int* domhi = geomdata.Domain().hiVect();
    const amrex::Real* dx = geomdata.CellSize();

    // Subsonic inflow BC in plane normal to x-direction
    // Obtain primitive variables
    // Get the local Mach number
    Real ux = s_int[0][UMX] / s_int[0][URHO];
    Real ss = std::sqrt( parm.eos_gamma * parm.Rsp * s_int[0][UTEMP] );
    Real mach = (1.0/ss)*std::sqrt( ( ux*ux + ((s_int[0][UMY]/s_int[0][URHO])*(s_int[0][UMY]/s_int[0][URHO])) ) );

    // The order of variables is rho, u, v, p
    Real Li[4]   = {0.0, 0.0, 0.0, 0.0};

    // Subsonic inflow boundary at x == 0
    // Since the boundary is at x = xlo, L2, L3, L4 are modeled
    // The other Li's are obtained from one-sided 2nd order differences 
#if AMREX_SPACEDIM==2
    Li[0] = (ux - ss) * (0.5 / dx[0]) 
          * ( ( (parm.eos_gamma-1.)*( -3.*s_int[0][UEINT] + 4.*s_int[1][UEINT] - s_int[2][UEINT] ) )
          - s_int[0][URHO] * ss *( -3.*(s_int[0][UMX]/s_int[0][URHO]) 
          + 4.*(s_int[1][UMX]/s_int[1][URHO]) - (s_int[2][UMX]/s_int[2][URHO]) )  );
    Li[3] = Li[0];
    Li[1] = 0.5 * (parm.eos_gamma - 1.) * ( Li[0] + Li[3] );

    Real drodx = (1.0/(ss*ss))*( (0.5*Li[0]/(ux-ss)) + (Li[1]/ux) + (0.5*Li[3]/(ss+ux)) );
        
    int del_i = domlo[0] - i;
    Real sp1 = s_int[1][URHO]; Real sp0 = sp0 = s_int[0][URHO];
    Real qgc[4] = {0.0, 0.0, 0.0, 0.0};
    qgc[0] = sp1 - 2.*dx[0]*drodx;
    qgc[1] = -2.*sp1 - 3.*sp0  + 6.*qgc[0]  + 6.*dx[0]*drodx;
    qgc[2] =  3.*sp1 + 10.*sp0 - 18.*qgc[0] + 6.*qgc[1] - 12.*dx[0]*drodx;
    qgc[3] = -2.*sp1 - 13.*sp0 + 24.*qgc[0] -12.*qgc[1] + 4.*qgc[2] + 12.*dx[0]*drodx;

    if(del_i > 4) s_ext[URHO] = qgc[3];
    else s_ext[URHO] = qgc[del_i-1];

    s_ext[UMX]   = s_ext[URHO] * parm.utarg;
    s_ext[UMY]   = s_ext[URHO] * parm.vtarg;
    Real pres    = s_ext[URHO] * parm.Rsp * parm.Ttarg;
    Real kineng  = (0.5 / s_ext[URHO]) * ( s_ext[UMX]*s_ext[UMX] + s_ext[UMY]*s_ext[UMY] );
    s_ext[UEDEN] = (pres/(parm.eos_gamma-1.)) + kineng;
    s_ext[UEINT] = pres / (parm.eos_gamma-1.);
    s_ext[URHOY_F] = s_ext[URHO] * s_int[0][URHOY_F] / s_int[0][URHO];
    s_ext[URHOY_O] = s_ext[URHO] * s_int[0][URHOY_O] / s_int[0][URHO];
    s_ext[URHOY_P] = s_ext[URHO] * s_int[0][URHOY_P] / s_int[0][URHO];
    s_ext[UTEMP] = parm.Ttarg;
    s_ext[SFOIL] = s_int[0][SFOIL];

#endif 

  }

#endif
