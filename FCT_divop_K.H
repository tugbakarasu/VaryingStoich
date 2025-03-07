#include <AMReX_Config.H>
#include <AMReX_REAL.H>
#include <CNS_parm.H>
#include <FCT_diffusion_eb_K.H>
#include <cns_prob_parm.H>
#include "CNS.H"

// Calculate equivalence ratio from mass fractions
// For this first compute unburnt mass fraction of fuel and oxidizer
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
Real phiFromYoYfYp(const Real yO, const Real yF, const Real yP, const ProbParm* prob_parm) 
{
    Real yO0 = yO + (prob_parm->OF_st / (1 + prob_parm->OF_st)) * yP;
    Real yF0 = yF + (1 / (1 + prob_parm->OF_st)) * yP;
    
    return prob_parm->OF_st*yF0/yO0;
}

// Calculate local equivalence ratio
// Get it using the local mass fractions of fuel and oxidizer
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
Real phiLocalFromYoYfYp(const Real yO, const Real yF, const Real yP, const ProbParm* prob_parm)
{
    return prob_parm->OF_st*yF/(yO + 1.E-20);
}

// Convert from phi and progress variable to individual mass fractions
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void phi_yProg_to_Yi(const Real phi, const Real yProg, 
                     Real& yO, Real& yF, Real& yP, const ProbParm* prob_parm)
{
    const Real OFst = prob_parm->OF_st;
    
    if (phi < 1.0) {
        yP = (1.0 - yProg)*phi*(OFst + 1.0)/(OFst + phi);
        yF = yProg*phi/(OFst + phi);
        yO = (yProg*phi + (1.0 - phi))*OFst/(OFst + phi);
    } else {
        yP = (OFst + 1.0)/(OFst + phi)*(1.0 - yProg);
        yF = (yProg + (phi - 1.0))/(OFst + phi);
        yO = yProg*OFst/(OFst + phi);
    }
}

// Function to calculate phi and reaction progress variable
// For this compute the unburnt fuel and oxidizer mass fraction 
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void phiAndYReact(Real yO, Real yF, Real yP, Real& phi, Real& Y, const ProbParm* prob_parm)
{
    // Calculate reconstructed initial mass fractions
    Real yO0 = yO + (prob_parm->OF_st / (1 + prob_parm->OF_st))*yP + 1.E-20;
    Real yF0 = yF + (1 / (1 + prob_parm->OF_st))*yP + 1.E-20;
    
    // Calculate phi
    phi = prob_parm->OF_st*yF0/yO0;
    
    // Calculate reaction progress variable
    // Use ternary operator for MERGE functionality
    Y = (phi < 1.0) ? yF/yF0 : yO/yO0;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void fct_eb_compute_div_loworder (int i, int j, int k, int n, IntVect const& blo, IntVect const& bhi,
                     Array4<Real> const& q, Array4<Real const> const& uo,
                     // Partially convected quantities (ucx, ucy, ucz)
                     AMREX_D_DECL(Array4<Real> const& ucx,
                                  Array4<Real> const& ucy,
                                  Array4<Real> const& ucz),
                     // Partially diffused quantities (udx, udy, udz)
                     AMREX_D_DECL(Array4<Real> const& udx,
                                  Array4<Real> const& udy,
                                  Array4<Real> const& udz),
                     // Low-order solution (udfab)
                     Array4<Real> const& udiff, Array4<Real> const& divc,
                     // Convective fluxes (fltx) at face centres
                     AMREX_D_DECL(Array4<Real const> const& u,
                                  Array4<Real const> const& v,
                                  Array4<Real const> const& w),
                     // Diffusion fluxes (fldx) at face centres (in the form volume*(conserved qty))
                     AMREX_D_DECL(Array4<Real const> const& ud,
                                  Array4<Real const> const& vd,
                                  Array4<Real const> const& wd),
                     // Physical diffusion fluxes (flpdx) at face centres
                     AMREX_D_DECL(Array4<Real const> const& upd,
                                  Array4<Real const> const& vpd,
                                  Array4<Real const> const& wpd),
                     AMREX_D_DECL(Array4<Real> const& fx,
                                  Array4<Real> const& fy,
                                  Array4<Real> const& fz),
                     Array4<EBCellFlag const> const& flag,
                     Array4<Real const> const& vfrc,
                     Array4<Real const> const& bcent,
                     Array4<Real const> const& coefs,
                     Array4<Real      > const& redistwgt,
                     AMREX_D_DECL(Array4<Real const> const& apx,
                                  Array4<Real const> const& apy,
                                  Array4<Real const> const& apz),
                     AMREX_D_DECL(Array4<Real const> const& fcx,
                                  Array4<Real const> const& fcy,
                                  Array4<Real const> const& fcz),
                     GpuArray<Real,AMREX_SPACEDIM> const& dxinv,
                     GpuArray<Real,AMREX_SPACEDIM> const& dx, Real dtinv, Real dt,
                     Parm const& parm, int eb_weights_type, bool do_visc, bool do_reac,
                     int rk, const amrex::Real vfmin)
{
    AMREX_D_TERM(bool x_high = (i == bhi[0]);,
                 bool y_high = (j == bhi[1]);,
                 bool z_high = (k == bhi[2]));
    bool valid_cell = AMREX_D_TERM( (blo[0] <= i) && (i <= bhi[0]),
                                 && (blo[1] <= j) && (j <= bhi[1]),
                                 && (blo[2] <= k) && (k <= bhi[2]) );

#if (AMREX_SPACEDIM == 2)
    if (flag(i,j,k).isCovered() || vfrc(i,j,k) <= vfmin)
    {
        ucx(i,j,k,n)   = 0.0;
        ucy(i,j,k,n)   = 0.0;
        udx(i,j,k,n)   = 0.0;
        udy(i,j,k,n)   = 0.0;
        udiff(i,j,k,n) = 0.0;
        divc(i,j,k,n)  = 0.0;
        if (valid_cell) {
            fx(i,j,k,n) = 0.;
            fy(i,j,k,n) = 0.;
            if (x_high) {
                fx(i+1,j,k,n) = 0.;
            }
            if (y_high) {
                fy(i,j+1,k,n) = 0.;
            }
        }
    }
    else if (flag(i,j,k).isRegular())
    {
        ucx(i,j,k,n)   = dxinv[0]*( u(i+1,j,k,n) - u(i,j,k,n) );
        ucy(i,j,k,n)   = dxinv[1]*( v(i,j+1,k,n) - v(i,j,k,n) );
        udx(i,j,k,n)   = dxinv[0]*( u(i+1,j,k,n) - u(i,j,k,n) ) 
                       - dtinv*( ud(i+1,j,k,n) - ud(i,j,k,n) );
        udy(i,j,k,n)   = dxinv[1]*( v(i,j+1,k,n) - v(i,j,k,n) )
                       - dtinv*( vd(i,j+1,k,n) - vd(i,j,k,n) );
        udiff(i,j,k,n) = dxinv[0]*( u(i+1,j,k,n) - u(i,j,k,n) ) 
                       - dtinv*( ud(i+1,j,k,n) - ud(i,j,k,n) )
                       + dxinv[1]*( v(i,j+1,k,n) - v(i,j,k,n) )
                       - dtinv*( vd(i,j+1,k,n) - vd(i,j,k,n) );
        divc(i,j,k,n) = dxinv[0]*( u(i+1,j,k,n) - u(i,j,k,n) ) 
                       - dtinv*( ud(i+1,j,k,n) - ud(i,j,k,n) )
                       + dxinv[1]*( v(i,j+1,k,n) - v(i,j,k,n) )
                       - dtinv*( vd(i,j+1,k,n) - vd(i,j,k,n) );

        if(do_visc){
            udx(i,j,k,n)   = udx(i,j,k,n) - dxinv[0]*(upd(i+1,j,k,n) - upd(i,j,k,n));
            udy(i,j,k,n)   = udy(i,j,k,n) - dxinv[1]*(vpd(i,j+1,k,n) - vpd(i,j,k,n));
            udiff(i,j,k,n) = udiff(i,j,k,n) - dxinv[0]*(upd(i+1,j,k,n) - upd(i,j,k,n))
                           - dxinv[1]*(vpd(i,j+1,k,n) - vpd(i,j,k,n));
            divc(i,j,k,n)  = divc(i,j,k,n) - dxinv[0]*(upd(i+1,j,k,n) - upd(i,j,k,n))
                           - dxinv[1]*(vpd(i,j+1,k,n) - vpd(i,j,k,n));
        }
	
	const ProbParm* prob_parm = CNS::d_prob_parm;
	
        if(do_reac){
        
            Real yF = uo(i,j,k,URHOY_F) / uo(i,j,k,URHO);
            Real yO = uo(i,j,k,URHOY_O) / uo(i,j,k,URHO);
            Real yP = uo(i,j,k,URHOY_P) / uo(i,j,k,URHO);
            
            // Calculate unburnt mixture equivalence ratio
	    const ProbParm* prob_parm = CNS::d_prob_parm;
	    Real phi_unburnt = phiFromYoYfYp(yO, yF, yP, prob_parm);
	
	    // Calculate local equivalence ratio (actual current ratio)
	    Real phi_local = phiLocalFromYoYfYp(yO, yF, yP, prob_parm);
	
	    // Calculate reconstructed initial mass fractions
	    Real yO0 = yO + (prob_parm->OF_st / (1 + prob_parm->OF_st)) * yP;
	    Real yF0 = yF + (1 / (1 + prob_parm->OF_st)) * yP;
	    
	    // Calculate reaction progress variable
	    // This represents how much of the original reactants have been consumed
	    Real Y_react = (phi_unburnt < 1.0) ? yF/yF0 : yO/yO0;
            
            Real omegarho = -uo(i,j,k,URHO) * parm.pre_exp * Y_react
                          * std::exp(-parm.Ea_dim / (parm.Rsp * parm.eos_mu * uo(i,j,k,UTEMP)));
            
            // Calculate species-specific reaction rates based on phi_unburnt (the original mixture)
		Real omega_F, omega_Ox, omega_Pr;
		Real s = prob_parm->OF_st; // Stoichiometric O/F ratio

		if (phi_unburnt < 1.0) {
		    // F-lean condition
		    omega_F =  (phi_unburnt / (s + phi_unburnt)) * omegarho;
		    omega_Ox =  (s * phi_unburnt / (s + phi_unburnt)) * omegarho;
		    omega_Pr = -((s + 1) * phi_unburnt / (s + phi_unburnt)) * omegarho;
		} 
		else if (phi_unburnt > 1.0) {
		    // F-rich condition
		    omega_F = (1.0 / (s + phi_unburnt)) * omegarho;
		    omega_Ox = (s / (s + phi_unburnt)) * omegarho;
		    omega_Pr = -((s + 1.0) / (s + phi_unburnt)) * omegarho;
		}
		else {
		    // Stoichiometric condition
		    omega_F = (1.0 / (s + 1.0)) * omegarho;
		    omega_Ox = (s / (s + 1.0)) * omegarho;
		    omega_Pr = -omegarho;
		}
                          
            if(n == UEDEN){
                udx(i,j,k,n)   = udx(i,j,k,n)   + omegarho*parm.q_dim;
                udy(i,j,k,n)   = udy(i,j,k,n)   + omegarho*parm.q_dim;
                udiff(i,j,k,n) = udiff(i,j,k,n) + omegarho*parm.q_dim;
                divc(i,j,k,n)  = divc(i,j,k,n)  + omegarho*parm.q_dim;
            }else if(n == URHOY_F){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_F;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_F;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_F;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_F;
            }else if(n == URHOY_O){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_Ox;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_Ox;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_Ox;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_Ox;
            }else if(n == URHOY_P){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_Pr;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_Pr;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_Pr;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_Pr;
            }
            
        }
        if (valid_cell) {
            fx(i,j,k,n) = u(i,j,k,n) - (dx[0]*dtinv*ud(i,j,k,n));
            fy(i,j,k,n) = v(i,j,k,n) - (dx[1]*dtinv*vd(i,j,k,n));
            if(do_visc){
                fx(i,j,k,n) = fx(i,j,k,n) - upd(i,j,k,n);
                fy(i,j,k,n) = fy(i,j,k,n) - vpd(i,j,k,n);
            }
            if (x_high) {
                fx(i+1,j,k,n) = u(i+1,j,k,n) - (dx[0]*dtinv*ud(i+1,j,k,n));
                if(do_visc)
                    fx(i+1,j,k,n) = fx(i+1,j,k,n) - upd(i+1,j,k,n);
            }
            if (y_high) {
                fy(i,j+1,k,n) = v(i,j+1,k,n) - (dx[1]*dtinv*vd(i,j+1,k,n));
                if(do_visc)
                    fy(i,j+1,k,n) = fy(i,j+1,k,n) - vpd(i,j+1,k,n);
            }
        }
    }
    else
    {
        Real fxmpd = 0.0;
        if(do_visc) fxmpd = upd(i,j,k,n);

        Real fxmc = u(i,j,k,n);
        Real fxmd = ud(i,j,k,n);
        Real ar = 1.0, arp1 = 1.0;
        if (apx(i,j,k) != 0.0 && apx(i,j,k) != 1.0) {
            ar = 1.0/apx(i,j,k);
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt, fcx(i,j,k,0)));
            Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0_rt;
            fxmc = (1.0-fracy)*fxmc + fracy * u(i,jj,k ,n);
            fxmd = (1.0-fracy)*fxmd + fracy *ud(i,jj,k ,n);
            if(do_visc) fxmpd = (1.-fracy)*fxmpd + fracy * upd(i,jj,k,n);
        }
        if (valid_cell) {
            fx(i,j,k,n) = fxmc - (dx[0]*dtinv*ar*fxmd) - fxmpd;
        }

        Real fxppd = 0.0;
        if(do_visc) fxppd = upd(i+1,j,k,n);

        Real fxpc = u(i+1,j,k,n);
        Real fxpd = ud(i+1,j,k,n);
        if (apx(i+1,j,k) != 0.0 && apx(i+1,j,k) != 1.0) {
            arp1 = 1.0/apx(i+1,j,k);
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt,fcx(i+1,j,k,0)));
            Real fracy = flag(i+1,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0_rt;
            fxpc = (1.0-fracy)*fxpc + fracy * u(i+1,jj,k,n);
            fxpd = (1.0-fracy)*fxpd + fracy *ud(i+1,jj,k,n);

            if(do_visc) fxppd = (1.0-fracy)*fxppd + fracy *upd(i+1,jj,k,n);
        }
        if (valid_cell && x_high) {
            fx(i+1,j,k,n) = fxpc - (dx[0]*dtinv*arp1*fxpd) - fxppd;
        }

        Real fympd = 0.0;
        if(do_visc) fympd = vpd(i,j,k,n);

        Real fymc = v(i,j,k,n);
        Real fymd = vd(i,j,k,n);
        ar = 1.0; arp1 = 1.0;
        if (apy(i,j,k) != 0.0 && apy(i,j,k) != 1.0) {
            ar = 1.0/apy(i,j,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j,k,0)));
            Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0_rt;
            fymc = (1.0-fracx)*fymc +  fracx *v(ii,j,k,n);
            fymd = (1.0-fracx)*fymd +  fracx *vd(ii,j,k,n);

            if(do_visc) fympd = (1.0-fracx)*fympd +  fracx *vpd(ii,j,k,n);
        }
        if (valid_cell) {
            fy(i,j,k,n) = fymc - (dx[1]*dtinv*ar*fymd) - fympd;
        }

        Real fyppd = 0.0;
        if(do_visc) fyppd = vpd(i,j+1,k,n);

        Real fypc = v(i,j+1,k,n);
        Real fypd = vd(i,j+1,k,n);
        if (apy(i,j+1,k) != 0.0 && apy(i,j+1,k) != 1.0) {
            arp1 = 1.0/apy(i,j+1,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j+1,k,0)));
            Real fracx = flag(i,j+1,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0_rt;
            fypc = (1.0-fracx)*fypc + fracx *v(ii,j+1,k,n);
            fypd = (1.0-fracx)*fypd + fracx *vd(ii,j+1,k,n);

            if(do_visc) fyppd = (1.0-fracx)*fyppd + fracx *vpd(ii,j+1,k,n);
        }
        if (valid_cell && y_high) {
            fy(i,j+1,k,n) = fypc - (dx[1]*dtinv*arp1*fypd) - fyppd;
        }

        // Compute quantities of cut-cells
        ucx(i,j,k,n)   = (1.0/vfrc(i,j,k)) * dxinv[0] * ( apx(i+1,j,k)*fxpc - apx(i,j,k)*fxmc );
        ucy(i,j,k,n)   = (1.0/vfrc(i,j,k)) * dxinv[1] * ( apy(i,j+1,k)*fypc - apy(i,j,k)*fymc );
        udx(i,j,k,n)   = (1.0/vfrc(i,j,k)) * ( (dxinv[0] * (apx(i+1,j,k)*fxpc - apx(i,j,k)*fxmc))
                       - dtinv*( fxpd - fxmd ) 
                       - (dxinv[0] * (apx(i+1,j,k)*fxppd - apx(i,j,k)*fxmpd)) ); 
        udy(i,j,k,n)   = (1.0/vfrc(i,j,k)) * ( (dxinv[1] * (apy(i,j+1,k)*fypc - apy(i,j,k)*fymc))
                       - dtinv*( fypd - fymd )
                       - (dxinv[1] * (apy(i,j+1,k)*fyppd - apy(i,j,k)*fympd)) );
        udiff(i,j,k,n) = (1.0/vfrc(i,j,k)) * ( 
                         (dxinv[0] * (apx(i+1,j,k)*fxpc - apx(i,j,k)*fxmc))
                       + (dxinv[1] * (apy(i,j+1,k)*fypc - apy(i,j,k)*fymc))
                       - dtinv*( fxpd - fxmd )
                       - dtinv*( fypd - fymd ) 
                       - (dxinv[0] * (apx(i+1,j,k)*fxppd - apx(i,j,k)*fxmpd))
                       - (dxinv[1] * (apy(i,j+1,k)*fyppd - apy(i,j,k)*fympd)) );
        divc(i,j,k,n) = (1.0/vfrc(i,j,k)) * ( 
                         (dxinv[0] * (apx(i+1,j,k)*fxpc - apx(i,j,k)*fxmc))
                       + (dxinv[1] * (apy(i,j+1,k)*fypc - apy(i,j,k)*fymc))
                       - dtinv*( fxpd - fxmd )
                       - dtinv*( fypd - fymd ) 
                       - (dxinv[0] * (apx(i+1,j,k)*fxppd - apx(i,j,k)*fxmpd))
                       - (dxinv[1] * (apy(i,j+1,k)*fyppd - apy(i,j,k)*fympd)) );

        GpuArray<Real,NEQNS> fc_wall;

        compute_hyp_wallflux(q(i,j,k,QRHO),AMREX_D_DECL(q(i,j,k,QU),q(i,j,k,QV),q(i,j,k,QW)),
                             q(i,j,k,QPRES),apx(i,j,k),apx(i+1,j,k),apy(i,j,k),apy(i,j+1,k),
                             fc_wall,parm);

        fc_wall[URHOY_F] = fc_wall[URHO] 
                       * amrex::max(Real(0.0), amrex::min(1.0,uo(i,j,k,URHOY_F) / uo(i,j,k,URHO)));
         fc_wall[URHOY_O] = fc_wall[URHO] 
                       * amrex::max(Real(0.0), amrex::min(1.0,uo(i,j,k,URHOY_P) / uo(i,j,k,URHO)));
         fc_wall[URHOY_P] = fc_wall[URHO] 
                       * amrex::max(Real(0.0), amrex::min(1.0,uo(i,j,k,URHOY_O) / uo(i,j,k,URHO)));              

        // Here we assume dx == dy == dz
        divc(i,j,k,n)  = divc(i,j,k,n) + (fc_wall[n]*dxinv[1]/vfrc(i,j,k));
        udiff(i,j,k,n) = udiff(i,j,k,n) + (fc_wall[n]*dxinv[1]/vfrc(i,j,k));

        if (do_visc)
        {
            GpuArray<Real,NEQNS> flux_diff_wall;
            fct_compute_diff_wallflux(i,j,k,q,coefs,bcent,
                                  apx(i,j,k),apx(i+1,j,k), apy(i,j,k),apy(i,j+1,k),
                                  flux_diff_wall,parm);
            udiff(i,j,k,n) +=  flux_diff_wall[n]*dxinv[1]/vfrc(i,j,k);
            divc(i,j,k,n)  +=  flux_diff_wall[n]*dxinv[1]/vfrc(i,j,k);
        }

        if(do_reac){
        
            Real yF = uo(i,j,k,URHOY_F) / uo(i,j,k,URHO);
            Real yO = uo(i,j,k,URHOY_O) / uo(i,j,k,URHO);
            Real yP = uo(i,j,k,URHOY_P) / uo(i,j,k,URHO);
            
            // Calculate unburnt mixture equivalence ratio
	    const ProbParm* prob_parm = CNS::d_prob_parm;
            Real phi_unburnt = phiFromYoYfYp(yO, yF, yP, prob_parm);
	
	    // Calculate local equivalence ratio (actual current ratio)
	    Real phi_local = phiLocalFromYoYfYp(yO, yF, yP, prob_parm);
	
	    // Calculate reconstructed initial mass fractions
	    Real yO0 = yO + (prob_parm->OF_st / (1 + prob_parm->OF_st)) * yP;
	    Real yF0 = yF + (1 / (1 + prob_parm->OF_st)) * yP;
	    
	    // Calculate reaction progress variable
	    // This represents how much of the original reactants have been consumed
	    Real Y_react = (phi_unburnt < 1.0) ? yF/yF0 : yO/yO0;
            
            Real omegarho = -uo(i,j,k,URHO) * parm.pre_exp * Y_react
                          * std::exp(-parm.Ea_dim / (parm.Rsp * parm.eos_mu * uo(i,j,k,UTEMP)));
            
            // Calculate species-specific reaction rates based on phi_unburnt (the original mixture)
		Real omega_F, omega_Ox, omega_Pr;
		Real s = prob_parm->OF_st; // Stoichiometric O/F ratio

		if (phi_unburnt < 1.0) {
		    // F-lean condition
		    omega_F =  (phi_unburnt / (s + phi_unburnt)) * omegarho;
		    omega_Ox =  (s * phi_unburnt / (s + phi_unburnt)) * omegarho;
		    omega_Pr = -((s + 1) * phi_unburnt / (s + phi_unburnt)) * omegarho;
		} 
		else if (phi_unburnt > 1.0) {
		    // F-rich condition
		    omega_F = (1.0 / (s + phi_unburnt)) * omegarho;
		    omega_Ox = (s / (s + phi_unburnt)) * omegarho;
		    omega_Pr = -((s + 1.0) / (s + phi_unburnt)) * omegarho;
		}
		else {
		    // Stoichiometric condition
		    omega_F = (1.0 / (s + 1.0)) * omegarho;
		    omega_Ox = (s / (s + 1.0)) * omegarho;
		    omega_Pr = -omegarho;
		}
                          
            if(n == UEDEN){
                udx(i,j,k,n)   = udx(i,j,k,n)   + omegarho*parm.q_dim;
                udy(i,j,k,n)   = udy(i,j,k,n)   + omegarho*parm.q_dim;
                udiff(i,j,k,n) = udiff(i,j,k,n) + omegarho*parm.q_dim;
                divc(i,j,k,n)  = divc(i,j,k,n)  + omegarho*parm.q_dim;
            }else if(n == URHOY_F){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_F;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_F;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_F;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_F;
            }else if(n == URHOY_O){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_Ox;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_Ox;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_Ox;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_Ox;
            }else if(n == URHOY_P){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_Pr;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_Pr;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_Pr;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_Pr;
            }
            
        }
    }

#else // 3-d starts here
    if(flag(i,j,k).isCovered() || vfrc(i,j,k) <= vfmin)
    {
        ucx(i,j,k,n)   = 0.0; ucy(i,j,k,n)   = 0.0; ucz(i,j,k,n) = 0.0;
        udx(i,j,k,n)   = 0.0; udy(i,j,k,n)   = 0.0; udz(i,j,k,n) = 0.0;
        udiff(i,j,k,n) = 0.0;
        divc(i,j,k,n)  = 0.0;
        if (valid_cell) {
            fx(i,j,k,n) = 0.; fy(i,j,k,n) = 0.; fz(i,j,k,n) = 0.;
            if (x_high) {
                fx(i+1,j,k,n) = 0.;
            }
            if (y_high) {
                fy(i,j+1,k,n) = 0.;
            }
            if (z_high) {
                fz(i,j,k+1,n) = 0.;
            }
        }
    } // if(flag(i,j,k).isCovered() || vfrc(i,j,k) <= vfmin)
    else if (flag(i,j,k).isRegular())
    {
        ucx(i,j,k,n)   = dxinv[0]*( u(i+1,j,k,n) - u(i,j,k,n) );
        ucy(i,j,k,n)   = dxinv[1]*( v(i,j+1,k,n) - v(i,j,k,n) );
        ucz(i,j,k,n)   = dxinv[2]*( w(i,j,k+1,n) - w(i,j,k,n) );

        udx(i,j,k,n)   = dxinv[0]*( u(i+1,j,k,n) - u(i,j,k,n) ) 
                       - dtinv*( ud(i+1,j,k,n) - ud(i,j,k,n) );
        udy(i,j,k,n)   = dxinv[1]*( v(i,j+1,k,n) - v(i,j,k,n) )
                       - dtinv*( vd(i,j+1,k,n) - vd(i,j,k,n) );
        udz(i,j,k,n)   = dxinv[2]*( w(i,j,k+1,n) - w(i,j,k,n) )
                       - dtinv*( wd(i,j,k+1,n) - wd(i,j,k,n) );

        udiff(i,j,k,n) = dxinv[0]*( u(i+1,j,k,n) - u(i,j,k,n) ) 
                       - dtinv*( ud(i+1,j,k,n) - ud(i,j,k,n) )
                       + dxinv[1]*( v(i,j+1,k,n) - v(i,j,k,n) )
                       - dtinv*( vd(i,j+1,k,n) - vd(i,j,k,n) )
                       + dxinv[2]*( w(i,j,k+1,n) - w(i,j,k,n) )
                       - dtinv*( wd(i,j,k+1,n) - wd(i,j,k,n) );

        divc(i,j,k,n) = dxinv[0]*( u(i+1,j,k,n) - u(i,j,k,n) ) 
                       - dtinv*( ud(i+1,j,k,n) - ud(i,j,k,n) )
                       + dxinv[1]*( v(i,j+1,k,n) - v(i,j,k,n) )
                       - dtinv*( vd(i,j+1,k,n) - vd(i,j,k,n) )
                       + dxinv[2]*( w(i,j,k+1,n) - w(i,j,k,n) )
                       - dtinv*( wd(i,j,k+1,n) - wd(i,j,k,n) );

        if(do_visc){
            udx(i,j,k,n)   = udx(i,j,k,n) - dxinv[0]*(upd(i+1,j,k,n) - upd(i,j,k,n));
            udy(i,j,k,n)   = udy(i,j,k,n) - dxinv[1]*(vpd(i,j+1,k,n) - vpd(i,j,k,n));
            udz(i,j,k,n)   = udz(i,j,k,n) - dxinv[2]*(wpd(i,j,k+1,n) - wpd(i,j,k,n));
            udiff(i,j,k,n) = udiff(i,j,k,n) - dxinv[0]*(upd(i+1,j,k,n) - upd(i,j,k,n))
                           - dxinv[1]*(vpd(i,j+1,k,n) - vpd(i,j,k,n))
                           - dxinv[2]*(wpd(i,j,k+1,n) - wpd(i,j,k,n));
            divc(i,j,k,n)  = divc(i,j,k,n) - dxinv[0]*(upd(i+1,j,k,n) - upd(i,j,k,n))
                           - dxinv[1]*(vpd(i,j+1,k,n) - vpd(i,j,k,n))
                           - dxinv[2]*(wpd(i,j,k+1,n) - wpd(i,j,k,n));
        }

        if(do_reac){
        
            Real yF = uo(i,j,k,URHOY_F) / uo(i,j,k,URHO);
            Real yO = uo(i,j,k,URHOY_O) / uo(i,j,k,URHO);
            Real yP = uo(i,j,k,URHOY_P) / uo(i,j,k,URHO);
            
            // Calculate unburnt mixture equivalence ratio
	    Real phi_unburnt = phiFromYoYfYp(yO, yF, yP);
	
	    // Calculate local equivalence ratio (actual current ratio)
	    Real phi_local = phiLocalFromYoYfYp(yO, yF, yP);
	
	    // Calculate reconstructed initial mass fractions
	    Real yO0 = yO + (prob_parm->OF_st / (1 + prob_parm->OF_st)) * yP;
	    Real yF0 = yF + (1 / (1 + prob_parm->OF_st)) * yP;
	    
	    // Calculate reaction progress variable
	    // This represents how much of the original reactants have been consumed
	    Real Y_react = (phi_unburnt < 1.0) ? yF/yF0 : yO/yO0;
            
            Real omegarho = -uo(i,j,k,URHO) * parm.pre_exp * Y_react
                          * std::exp(-parm.Ea_dim / (parm.Rsp * parm.eos_mu * uo(i,j,k,UTEMP)));
            
            // Calculate species-specific reaction rates based on phi_unburnt (the original mixture)
		Real omega_F, omega_Ox, omega_Pr;
		Real s = prob_parm->OF_st; // Stoichiometric O/F ratio

		if (phi_unburnt < 1.0) {
		    // F-lean condition
		    omega_F =  (phi_unburnt / (s + phi_unburnt)) * omegarho;
		    omega_Ox =  (s * phi_unburnt / (s + phi_unburnt)) * omegarho;
		    omega_Pr = -((s + 1) * phi_unburnt / (s + phi_unburnt)) * omegarho;
		} 
		else if (phi_unburnt > 1.0) {
		    // F-rich condition
		    omega_F = (1.0 / (s + phi_unburnt)) * omegarho;
		    omega_Ox = (s / (s + phi_unburnt)) * omegarho;
		    omega_Pr = -((s + 1.0) / (s + phi_unburnt)) * omegarho;
		}
		else {
		    // Stoichiometric condition
		    omega_F = (1.0 / (s + 1.0)) * omegarho;
		    omega_Ox = (s / (s + 1.0)) * omegarho;
		    omega_Pr = -omegarho;
		}
                          
            if(n == UEDEN){
                udx(i,j,k,n)   = udx(i,j,k,n)   + omegarho*parm.q_dim;
                udy(i,j,k,n)   = udy(i,j,k,n)   + omegarho*parm.q_dim;
                udz(i,j,k,n)   = udz(i,j,k,n)   + omegarho*parm.q_dim;
                udiff(i,j,k,n) = udiff(i,j,k,n) + omegarho*parm.q_dim;
                divc(i,j,k,n)  = divc(i,j,k,n)  + omegarho*parm.q_dim;
            }else if(n == URHOY_F){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_F;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_F;
                udz(i,j,k,n)   = udz(i,j,k,n)   - omega_F;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_F;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_F;
            }else if(n == URHOY_O){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_Ox;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_Ox;
                udz(i,j,k,n)   = udz(i,j,k,n)   - omega_Ox;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_Ox;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_Ox;
            }else if(n == URHOY_P){
                udx(i,j,k,n)   = udx(i,j,k,n)   - omega_Pr;
                udy(i,j,k,n)   = udy(i,j,k,n)   - omega_Pr;
                udz(i,j,k,n)   = udz(i,j,k,n)   - omega_Pr;
                udiff(i,j,k,n) = udiff(i,j,k,n) - omega_Pr;
                divc(i,j,k,n)  = divc(i,j,k,n)  - omega_Pr;
            }
            
        }

        if (valid_cell) {
            fx(i,j,k,n) = u(i,j,k,n) - (dx[0]*dtinv*ud(i,j,k,n));
            fy(i,j,k,n) = v(i,j,k,n) - (dx[1]*dtinv*vd(i,j,k,n));
            fz(i,j,k,n) = w(i,j,k,n) - (dx[2]*dtinv*wd(i,j,k,n));
            if(do_visc){
                fx(i,j,k,n) = fx(i,j,k,n) - upd(i,j,k,n);
                fy(i,j,k,n) = fy(i,j,k,n) - vpd(i,j,k,n);
                fz(i,j,k,n) = fz(i,j,k,n) - wpd(i,j,k,n);
            }
            if (x_high) {
                fx(i+1,j,k,n) = u(i+1,j,k,n) - (dx[0]*dtinv*ud(i+1,j,k,n));
                if(do_visc)
                    fx(i+1,j,k,n) = fx(i+1,j,k,n) - upd(i+1,j,k,n);
            }
            if (y_high) {
                fy(i,j+1,k,n) = v(i,j+1,k,n) - (dx[1]*dtinv*vd(i,j+1,k,n));
                if(do_visc)
                    fy(i,j+1,k,n) = fy(i,j+1,k,n) - vpd(i,j+1,k,n);
            }
            if (z_high) {
                fz(i,j,k+1,n) = w(i,j,k+1,n) - (dx[2]*dtinv*wd(i,j,k+1,n));
                if(do_visc)
                    fz(i,j,k+1,n) = fz(i,j,k+1,n) - wpd(i,j,k+1,n);
            }
        }

    } // else if (flag(i,j,k).isRegular())
    else
    {
        Real fxmpd = 0.0;
        if(do_visc) fxmpd = upd(i,j,k,n);

        Real fxmc = u(i,j,k,n);
        Real fxmd = ud(i,j,k,n);
        Real ar = 1.0, arp1 = 1.0;
        if (apx(i,j,k) != 0.0 && apx(i,j,k) != 1.0) {
            ar = 1.0/apx(i,j,k);
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt, fcx(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0_rt, fcx(i,j,k,1)));
            Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0_rt;
            Real fracz = flag(i,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0_rt;
            fxmc = (1.0-fracy)*(1.0-fracz)*fxmc 
                 +      fracy *(1.0-fracz)*u(i,jj,k ,n)
                 +      fracz *(1.0-fracy)*u(i,j ,kk,n)
                 +      fracy *     fracz *u(i,jj,kk,n);
            fxmd = (1.0-fracy)*(1.0-fracz)*fxmd 
                 +      fracy *(1.0-fracz)*ud(i,jj,k ,n)
                 +      fracz *(1.0-fracy)*ud(i,j ,kk,n)
                 +      fracy *     fracz *ud(i,jj,kk,n);
            if(do_visc) {
                fxmpd = (1.0-fracy)*(1.0-fracz)*fxmpd 
                 +      fracy *(1.0-fracz)*upd(i,jj,k ,n)
                 +      fracz *(1.0-fracy)*upd(i,j ,kk,n)
                 +      fracy *     fracz *upd(i,jj,kk,n);
            }
        }
        if (valid_cell) {
            fx(i,j,k,n) = fxmc - (dx[0]*dtinv*ar*fxmd) - fxmpd;
        }

        Real fxppd = 0.0;
        if(do_visc) fxppd = upd(i+1,j,k,n);

        Real fxpc = u(i+1,j,k,n);
        Real fxpd = ud(i+1,j,k,n);
        if (apx(i+1,j,k) != 0.0 && apx(i+1,j,k) != 1.0) {
            arp1 = 1.0/apx(i+1,j,k);
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt,fcx(i+1,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0_rt,fcx(i+1,j,k,1)));
            Real fracy = flag(i+1,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0_rt;
            Real fracz = flag(i+1,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcx(i+1,j,k,1)) : 0.0_rt;
            fxpc = (1.0-fracy)*(1.0-fracz)*fxpc
                 +      fracy *(1.0-fracz)*u(i+1,jj,k ,n)
                 +      fracz *(1.0-fracy)*u(i+1,j ,kk,n)
                 +      fracy *     fracz *u(i+1,jj,kk,n);
            fxpd = (1.0-fracy)*(1.0-fracz)*fxpd
                 +      fracy *(1.0-fracz)*ud(i+1,jj,k ,n)
                 +      fracz *(1.0-fracy)*ud(i+1,j ,kk,n)
                 +      fracy *     fracz *ud(i+1,jj,kk,n);
            if(do_visc){
                fxppd = (1.0-fracy)*(1.0-fracz)*fxppd
                      +      fracy *(1.0-fracz)*upd(i+1,jj,k ,n)
                      +      fracz *(1.0-fracy)*upd(i+1,j ,kk,n)
                      +      fracy *     fracz *upd(i+1,jj,kk,n);
            }
            
        }
        if (valid_cell && x_high) {
            fx(i+1,j,k,n) = fxpc - (dx[0]*dtinv*arp1*fxpd) - fxppd;
        }

        Real fympd = 0.0;
        if(do_visc) fympd = vpd(i,j,k,n);

        Real fymc = v(i,j,k,n);
        Real fymd = vd(i,j,k,n);
        ar = 1.0; arp1 = 1.0;
        if (apy(i,j,k) != 0.0 && apy(i,j,k) != 1.0) {
            ar = 1.0/apy(i,j,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j,k,1)));
            Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0_rt;
            Real fracz = flag(i,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0_rt;
            fymc = (1.0-fracx)*(1.0-fracz)*fymc
                 +      fracx *(1.0-fracz)*v(ii,j,k ,n)
                 +      fracz *(1.0-fracx)*v(i ,j,kk,n)
                 +      fracx *     fracz *v(ii,j,kk,n);
            fymd = (1.0-fracx)*(1.0-fracz)*fymd
                 +      fracx *(1.0-fracz)*vd(ii,j,k ,n)
                 +      fracz *(1.0-fracx)*vd(i ,j,kk,n)
                 +      fracx *     fracz *vd(ii,j,kk,n);

            if(do_visc){
                fympd = (1.0-fracx)*(1.0-fracz)*fympd
                 +      fracx *(1.0-fracz)*vpd(ii,j,k ,n)
                 +      fracz *(1.0-fracx)*vpd(i ,j,kk,n)
                 +      fracx *     fracz *vpd(ii,j,kk,n);
            } 
        }
        if (valid_cell) {
            fy(i,j,k,n) = fymc - (dx[1]*dtinv*ar*fymd) - fympd;
        }

        Real fyppd = 0.0;
        if(do_visc) fyppd = vpd(i,j+1,k,n);

        Real fypc = v(i,j+1,k,n);
        Real fypd = vd(i,j+1,k,n);
        if (apy(i,j+1,k) != 0.0 && apy(i,j+1,k) != 1.0) {
            arp1 = 1.0/apy(i,j+1,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j+1,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j+1,k,1)));
            Real fracx = flag(i,j+1,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0_rt;
            Real fracz = flag(i,j+1,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcy(i,j+1,k,1)) : 0.0_rt;
            fypc = (1.0-fracx)*(1.0-fracz)*fypc
                 +      fracx *(1.0-fracz)*v(ii,j+1,k ,n)
                 +      fracz *(1.0-fracx)*v(i ,j+1,kk,n)
                 +      fracx *     fracz *v(ii,j+1,kk,n);
            fypd = (1.0-fracx)*(1.0-fracz)*fypd
                 +      fracx *(1.0-fracz)*vd(ii,j+1,k ,n)
                 +      fracz *(1.0-fracx)*vd(i ,j+1,kk,n)
                 +      fracx *     fracz *vd(ii,j+1,kk,n);

            if(do_visc){
                fyppd = (1.0-fracx)*(1.0-fracz)*fyppd
                 +      fracx *(1.0-fracz)*vpd(ii,j+1,k ,n)
                 +      fracz *(1.0-fracx)*vpd(i ,j+1,kk,n)
                 +      fracx *     fracz *vpd(ii,j+1,kk,n);
            } 
        }
        if (valid_cell && y_high) {
            fy(i,j+1,k,n) = fypc - (dx[1]*dtinv*arp1*fypd) - fyppd;
        }

        Real fzmpd = 0.0;
        if(do_visc) fzmpd = wpd(i,j,k,n);

        Real fzmc = w(i,j,k,n);
        Real fzmd = wd(i,j,k,n);
        ar = 1.0, arp1 = 1.0;

        if (apz(i,j,k) != 0.0 && apz(i,j,k) != 1.0) {
            ar = 1.0/apz(i,j,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcz(i,j,k,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt,fcz(i,j,k,1)));
            Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0_rt;
            Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0_rt;

            fzmc = (1.0-fracx)*(1.0-fracy)*fzmc
                 +      fracx *(1.0-fracy)*w(ii,j ,k,n)
                 +      fracy *(1.0-fracx)*w(i ,jj,k,n)
                 +      fracx *     fracy *w(ii,jj,k,n);

            fzmd = (1.0-fracx)*(1.0-fracy)*fzmd
                 +      fracx *(1.0-fracy)*wd(ii,j ,k,n)
                 +      fracy *(1.0-fracx)*wd(i ,jj,k,n)
                 +      fracx *     fracy *wd(ii,jj,k,n);

            if(do_visc){
                fzmpd = (1.0-fracx)*(1.0-fracy)*fzmpd
                 +      fracx *(1.0-fracy)*wpd(ii,j ,k,n)
                 +      fracy *(1.0-fracx)*wpd(i ,jj,k,n)
                 +      fracx *     fracy *wpd(ii,jj,k,n);
            }
        }

        if (valid_cell) {
            fz(i,j,k,n) = fzmc - (dx[2]*dtinv*ar*fzmd) - fzmpd;
        }

        Real fzppd = 0.0;
        if(do_visc) fzppd = wpd(i,j,k+1,n);

        Real fzpc = w(i,j,k+1,n);
        Real fzpd = wd(i,j,k+1,n);

        if (apz(i,j,k+1) != 0.0 && apz(i,j,k+1) != 1.0) {
            arp1 = 1.0/apz(i,j,k+1);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcz(i,j,k+1,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt,fcz(i,j,k+1,1)));
            Real fracx = flag(i,j,k+1).isConnected(ii-i,0,0) ? amrex::Math::abs(fcz(i,j,k+1,0)) : 0.0_rt;
            Real fracy = flag(i,j,k+1).isConnected(0,jj-j,0) ? amrex::Math::abs(fcz(i,j,k+1,1)) : 0.0_rt;
            
            fzpc = (1.0-fracx)*(1.0-fracy)*fzpc
                +      fracx *(1.0-fracy)*w(ii,j ,k+1,n)
                +      fracy *(1.0-fracx)*w(i ,jj,k+1,n)
                +      fracx *     fracy *w(ii,jj,k+1,n);

            fzpd = (1.0-fracx)*(1.0-fracy)*fzpd
                +      fracx *(1.0-fracy)*wd(ii,j ,k+1,n)
                +      fracy *(1.0-fracx)*wd(i ,jj,k+1,n)
                +      fracx *     fracy *wd(ii,jj,k+1,n);

            if(do_visc){
                fzppd = (1.0-fracx)*(1.0-fracy)*fzppd
                +      fracx *(1.0-fracy)*wpd(ii,j ,k+1,n)
                +      fracy *(1.0-fracx)*wpd(i ,jj,k+1,n)
                +      fracx *     fracy *wpd(ii,jj,k+1,n);
            }

        }

        if (valid_cell && z_high) {
            fz(i,j,k+1,n) = fzpc - (dx[2]*dtinv*arp1*fzpd) - fzppd;
        }

        // Compute quantities of cut-cells
        ucx(i,j,k,n)   = (1.0/vfrc(i,j,k)) * dxinv[0] * ( apx(i+1,j,k)*fxpc - apx(i,j,k)*fxmc );
        ucy(i,j,k,n)   = (1.0/vfrc(i,j,k)) * dxinv[1] * ( apy(i,j+1,k)*fypc - apy(i,j,k)*fymc );
        ucz(i,j,k,n)   = (1.0/vfrc(i,j,k)) * dxinv[2] * ( apz(i,j,k+1)*fzpc - apz(i,j,k)*fzmc );

        udx(i,j,k,n)   = (1.0/vfrc(i,j,k)) * ( (dxinv[0] * (apx(i+1,j,k)*fxpc - apx(i,j,k)*fxmc))
                       - dtinv*( fxpd - fxmd ) 
                       - (dxinv[0] * (apx(i+1,j,k)*fxppd - apx(i,j,k)*fxmpd)) ); 
        udy(i,j,k,n)   = (1.0/vfrc(i,j,k)) * ( (dxinv[1] * (apy(i,j+1,k)*fypc - apy(i,j,k)*fymc))
                       - dtinv*( fypd - fymd )
                       - (dxinv[1] * (apy(i,j+1,k)*fyppd - apy(i,j,k)*fympd)) );
        udz(i,j,k,n)   = (1.0/vfrc(i,j,k)) * ( (dxinv[2] * (apz(i,j,k+1)*fzpc - apz(i,j,k)*fzmc))
                       - dtinv*( fzpd - fzmd )
                       - (dxinv[2] * (apz(i,j,k+1)*fzppd - apz(i,j,k)*fzmpd)) );

        udiff(i,j,k,n) = (1.0/vfrc(i,j,k)) * ( 
                         (dxinv[0] * (apx(i+1,j,k)*fxpc - apx(i,j,k)*fxmc))
                       + (dxinv[1] * (apy(i,j+1,k)*fypc - apy(i,j,k)*fymc))
                       + (dxinv[2] * (apz(i,j,k+1)*fzpc - apz(i,j,k)*fzmc))
                       - dtinv*( fxpd - fxmd )
                       - dtinv*( fypd - fymd )
                       - dtinv*( fzpd - fzmd ) 
                       - (dxinv[0] * (apx(i+1,j,k)*fxppd - apx(i,j,k)*fxmpd))
                       - (dxinv[1] * (apy(i,j+1,k)*fyppd - apy(i,j,k)*fympd)) 
                       - (dxinv[2] * (apz(i,j,k+1)*fzppd - apz(i,j,k)*fzmpd)) );

        divc(i,j,k,n) = (1.0/vfrc(i,j,k)) * ( 
                         (dxinv[0] * (apx(i+1,j,k)*fxpc - apx(i,j,k)*fxmc))
                       + (dxinv[1] * (apy(i,j+1,k)*fypc - apy(i,j,k)*fymc))
                       + (dxinv[2] * (apz(i,j,k+1)*fzpc - apz(i,j,k)*fzmc))
                       - dtinv*( fxpd - fxmd )
                       - dtinv*( fypd - fymd ) 
                       - dtinv*( fzpd - fzmd ) 
                       - (dxinv[0] * (apx(i+1,j,k)*fxppd - apx(i,j,k)*fxmpd))
                       - (dxinv[1] * (apy(i,j+1,k)*fyppd - apy(i,j,k)*fympd))
                       - (dxinv[2] * (apz(i,j,k+1)*fzppd - apz(i,j,k)*fzmpd)) );

        GpuArray<Real,NEQNS> fc_wall;
        compute_hyp_wallflux(q(i,j,k,QRHO),AMREX_D_DECL(q(i,j,k,QU),q(i,j,k,QV),q(i,j,k,QW)),
                             q(i,j,k,QPRES),apx(i,j,k),apx(i+1,j,k),apy(i,j,k),apy(i,j+1,k),
                             apz(i,j,k),apz(i,j,k+1),fc_wall,parm);

        // Here we assume dx == dy == dz
        divc(i,j,k,n)  = divc(i,j,k,n) + (fc_wall[n]*dxinv[0]/vfrc(i,j,k));
        udiff(i,j,k,n) = udiff(i,j,k,n) + (fc_wall[n]*dxinv[0]/vfrc(i,j,k));

        if (do_visc)
        {
            GpuArray<Real,NEQNS> flux_diff_wall;
            fct_compute_diff_wallflux(i,j,k,q,coefs,bcent,
                                  apx(i,j,k),apx(i+1,j,k), 
                                  apy(i,j,k),apy(i,j+1,k),
                                  apz(i,j,k),apz(i,j,k+1),
                                  flux_diff_wall,parm);
            udiff(i,j,k,n) +=  flux_diff_wall[n]*dxinv[0]*dxinv[0]/vfrc(i,j,k);
            divc(i,j,k,n)  +=  flux_diff_wall[n]*dxinv[0]*dxinv[0]/vfrc(i,j,k);
        }

        udy(i,j,k,n)   = udy(i,j,k,n)   - omega_P;

    }

#endif

    // The operations following this assume we have returned the negative of the divergence of fluxes.
    udiff(i,j,k,n) *= -1.0;
    ucx(i,j,k,n) *= -1.0;
    ucy(i,j,k,n) *= -1.0;
    udx(i,j,k,n) *= -1.0;
    udy(i,j,k,n) *= -1.0;

#if AMREX_SPACEDIM==3
    ucz(i,j,k,n) *= -1.0;
    udz(i,j,k,n) *= -1.0;
#endif

    // Go ahead and make the redistwgt array here since we'll need it in flux_redistribute
    if (eb_weights_type == 0)
        { redistwgt(i,j,k) = 1.0; }
    else if (eb_weights_type == 1)
        { redistwgt(i,j,k) = q(i,j,k,QRHO)*( q(i,j,k,QEINT) +
#if (AMREX_SPACEDIM == 2)
                        0.5*(q(i,j,k,QU)*q(i,j,k,QU) + q(i,j,k,QV)*q(i,j,k,QV)) ); }
#else
                    0.5*(q(i,j,k,QU)*q(i,j,k,QU) + q(i,j,k,QV)*q(i,j,k,QV) + q(i,j,k,QW)*q(i,j,k,QW)) );
      }
#endif
    else if (eb_weights_type == 2)
        { redistwgt(i,j,k) = q(i,j,k,QRHO); }
    else if (eb_weights_type == 3 && vfrc(i,j,k) > vfmin)
        {  redistwgt(i,j,k) = vfrc(i,j,k); }

}

// ----------------------------------------------------------------------------
// Computing the divergence for corrected fluxes

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void fct_eb_compute_div_corrected (int i, int j, int k, int n, IntVect const& blo, IntVect const& bhi,
                     Array4<Real> const& q, Array4<Real const> const& uo,
                     // Conservative divergence (divc)
                     Array4<Real> const& divc,
                     // Corrected fluxes (flax) at face centres
                     AMREX_D_DECL(Array4<Real const> const& ua,
                                  Array4<Real const> const& va,
                                  Array4<Real const> const& wa),
                     // Fluxes at face centroids (for refluxing)
                     AMREX_D_DECL(Array4<Real> const& fx,
                                  Array4<Real> const& fy,
                                  Array4<Real> const& fz),
                     Array4<EBCellFlag const> const& flag,
                     Array4<Real const> const& vfrc,
                     Array4<Real const> const& bcent,
                     Array4<Real const> const& coefs,
                     Array4<Real      > const& redistwgt,
                     AMREX_D_DECL(Array4<Real const> const& apx,
                                  Array4<Real const> const& apy,
                                  Array4<Real const> const& apz),
                     AMREX_D_DECL(Array4<Real const> const& fcx,
                                  Array4<Real const> const& fcy,
                                  Array4<Real const> const& fcz),
                     GpuArray<Real,AMREX_SPACEDIM> const& dxinv,
                     GpuArray<Real,AMREX_SPACEDIM> const& dx, Real dtinv,
                     Parm const& /*parm*/, int eb_weights_type, bool do_visc, int rk, const amrex::Real vfmin)
{
    AMREX_D_TERM(bool x_high = (i == bhi[0]);,
                 bool y_high = (j == bhi[1]);,
                 bool z_high = (k == bhi[2]));
    bool valid_cell = AMREX_D_TERM( (blo[0] <= i) && (i <= bhi[0]),
                                 && (blo[1] <= j) && (j <= bhi[1]),
                                 && (blo[2] <= k) && (k <= bhi[2]) );

#if (AMREX_SPACEDIM == 2)
    if (flag(i,j,k).isCovered() || vfrc(i,j,k) <= vfmin)
    {
        divc(i,j,k,n) = 0.0;
        if (valid_cell) {
            fx(i,j,k,n) = 0.;
            fy(i,j,k,n) = 0.;
            if (x_high) {
                fx(i+1,j,k,n) = 0.;
            }
            if (y_high) {
                fy(i,j+1,k,n) = 0.;
            }
        }
    }
    else if (flag(i,j,k).isRegular())
    {
        Real temp = divc(i,j,k,n);
        divc(i,j,k,n) = temp + dtinv*( ua(i+1,j,k,n) - ua(i,j,k,n) )
                      + dtinv*( va(i,j+1,k,n) - va(i,j,k,n) );
        if (valid_cell) {
            fx(i,j,k,n) += (dx[0]*dtinv*ua(i,j,k,n));
            fy(i,j,k,n) += (dx[1]*dtinv*va(i,j,k,n));
            if (x_high) {
                fx(i+1,j,k,n) += dx[0]*dtinv*ua(i+1,j,k,n);
            }
            if (y_high) {
                fy(i,j+1,k,n) += dx[1]*dtinv*va(i,j+1,k,n);
            }
        }
    }
    else
    {
        Real ar = 1.0, arp1 = 1.0;
        Real fxma = ua(i,j,k,n);
        if (apx(i,j,k) != 0.0 && apx(i,j,k) != 1.0) {
            ar = 1.0/apx(i,j,k);
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt, fcx(i,j,k,0)));
            Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0_rt;
            fxma = (1.0-fracy)*fxma + fracy *ua(i,jj,k ,n);
        }
        if (valid_cell) {
            fx(i,j,k,n) += (dx[0]*dtinv*ar*fxma);
        }

        Real fxpa = ua(i+1,j,k,n);
        if (apx(i+1,j,k) != 0.0 && apx(i+1,j,k) != 1.0) {
            arp1 = 1.0/apx(i+1,j,k);
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt,fcx(i+1,j,k,0)));
            Real fracy = flag(i+1,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0_rt;
            fxpa = (1.0-fracy)*fxpa + fracy *ua(i+1,jj,k,n);
        }
        if (valid_cell && x_high) {
            fx(i+1,j,k,n) += (dx[0]*dtinv*arp1*fxpa);
        }

        Real fyma = va(i,j,k,n);
        ar = 1.0, arp1 = 1.0;
        if (apy(i,j,k) != 0.0 && apy(i,j,k) != 1.0) {
            ar = 1.0/apy(i,j,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j,k,0)));
            Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0_rt;
            fyma = (1.0-fracx)*fyma +  fracx *va(ii,j,k,n);
        }
        if (valid_cell) {
            fy(i,j,k,n) += (dx[1]*dtinv*ar*fyma);
        }

        Real fypa = va(i,j+1,k,n);
        if (apy(i,j+1,k) != 0.0 && apy(i,j+1,k) != 1.0) {
            arp1 = 1.0/apy(i,j+1,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j+1,k,0)));
            Real fracx = flag(i,j+1,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0_rt;
            fypa = (1.0-fracx)*fypa + fracx *va(ii,j+1,k,n);
        }
        if (valid_cell && y_high && rk == 2) {
            fy(i,j+1,k,n) += (dx[1]*dtinv*arp1*fypa);
        }

        // Compute quantities of cut-cells
        Real temp = divc(i,j,k,n);
        divc(i,j,k,n) = temp
                      + (1.0/vfrc(i,j,k))*( dtinv*( fxpa - fxma )
                      + dtinv*( fypa - fyma ) );
    }

#else // 3-d starts here
    if (flag(i,j,k).isCovered() || vfrc(i,j,k) <= vfmin)
    {
        divc(i,j,k,n) = 0.0;
        if (valid_cell) {
            fx(i,j,k,n) = 0.; fy(i,j,k,n) = 0.; fz(i,j,k,n) = 0.;
            if (x_high) {
                fx(i+1,j,k,n) = 0.;
            }
            if (y_high) {
                fy(i,j+1,k,n) = 0.;
            }
            if(z_high){
                fz(i,j,k+1,n) = 0.;
            }
        }
    }
    else if (flag(i,j,k).isRegular())
    {
        Real temp = divc(i,j,k,n);
        divc(i,j,k,n) = temp + dtinv*( ua(i+1,j,k,n) - ua(i,j,k,n) )
                      + dtinv*( va(i,j+1,k,n) - va(i,j,k,n) )
                      + dtinv*( wa(i,j,k+1,n) - wa(i,j,k,n) );
        if (valid_cell) {
            fx(i,j,k,n) += (dx[0]*dtinv*ua(i,j,k,n));
            fy(i,j,k,n) += (dx[1]*dtinv*va(i,j,k,n));
            fz(i,j,k,n) += (dx[2]*dtinv*wa(i,j,k,n));
            if (x_high) {
                fx(i+1,j,k,n) += dx[0]*dtinv*ua(i+1,j,k,n);
            }
            if (y_high) {
                fy(i,j+1,k,n) += dx[1]*dtinv*va(i,j+1,k,n);
            }
            if (z_high) {
                fz(i,j,k+1,n) += dx[2]*dtinv*wa(i,j,k+1,n);
            }
        }
    }
    else
    {
        Real ar = 1.0, arp1 = 1.0;
        Real fxma = ua(i,j,k,n);
        if (apx(i,j,k) != 0.0 && apx(i,j,k) != 1.0) {
            ar = 1.0/apx(i,j,k);
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt, fcx(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0_rt, fcx(i,j,k,1)));
            Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0_rt;
            Real fracz = flag(i,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0_rt;

            fxma = (1.0-fracy)*(1.0-fracz)*fxma
                +      fracy *(1.0-fracz)*ua(i,jj,k ,n)
                +      fracz *(1.0-fracy)*ua(i,j ,kk,n)
                +      fracy *     fracz *ua(i,jj,kk,n);
        }
        if (valid_cell) {
            fx(i,j,k,n) += (dx[0]*dtinv*ar*fxma);
        }

        Real fxpa = ua(i+1,j,k,n);
        if (apx(i+1,j,k) != 0.0 && apx(i+1,j,k) != 1.0) {
            arp1 = 1.0/apx(i+1,j,k);
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt,fcx(i+1,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0_rt,fcx(i+1,j,k,1)));
            Real fracy = flag(i+1,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0_rt;
            Real fracz = flag(i+1,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcx(i+1,j,k,1)) : 0.0_rt;

            fxpa = (1.0-fracy)*(1.0-fracz)*fxpa
                +      fracy *(1.0-fracz)*ua(i+1,jj,k ,n)
                +      fracz *(1.0-fracy)*ua(i+1,j ,kk,n)
                +      fracy *     fracz *ua(i+1,jj,kk,n);

        }
        if (valid_cell && x_high) {
            fx(i+1,j,k,n) += (dx[0]*dtinv*arp1*fxpa);
        }

        Real fyma = va(i,j,k,n);
        ar = 1.0, arp1 = 1.0;
        if (apy(i,j,k) != 0.0 && apy(i,j,k) != 1.0) {
            ar = 1.0/apy(i,j,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j,k,1)));
            Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0_rt;
            Real fracz = flag(i,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0_rt;
            fyma = (1.0-fracx)*(1.0-fracz)*fyma
                +      fracx *(1.0-fracz)*va(ii,j,k ,n)
                +      fracz *(1.0-fracx)*va(i ,j,kk,n)
                +      fracx *     fracz *va(ii,j,kk,n);
        }
        if (valid_cell) {
            fy(i,j,k,n) += (dx[1]*dtinv*ar*fyma);
        }

        Real fypa = va(i,j+1,k,n);
        if (apy(i,j+1,k) != 0.0 && apy(i,j+1,k) != 1.0) {
            arp1 = 1.0/apy(i,j+1,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j+1,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0_rt,fcy(i,j+1,k,1)));
            Real fracx = flag(i,j+1,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0_rt;
            Real fracz = flag(i,j+1,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcy(i,j+1,k,1)) : 0.0_rt;

            fypa = (1.0-fracx)*(1.0-fracz)*fypa
                +      fracx *(1.0-fracz)*va(ii,j+1,k ,n)
                +      fracz *(1.0-fracx)*va(i ,j+1,kk,n)
                +      fracx *     fracz *va(ii,j+1,kk,n);
        }
        if (valid_cell && y_high) {
            fy(i,j+1,k,n) += (dx[1]*dtinv*arp1*fypa);
        }

        Real fzma = wa(i,j,k,n);
        ar = 1.0, arp1 = 1.0;
        if (apz(i,j,k) != 0.0 && apz(i,j,k) != 1.0) {
            ar = 1.0/apz(i,j,k);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcz(i,j,k,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt,fcz(i,j,k,1)));
            Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0_rt;
            Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0_rt;
            fzma = (1.0-fracx)*(1.0-fracy)*fzma
                 +      fracx *(1.0-fracy)*wa(ii,j ,k,n)
                 +      fracy *(1.0-fracx)*wa(i ,jj,k,n)
                 +      fracx *     fracy *wa(ii,jj,k,n);
        }
        if (valid_cell) {
            fz(i,j,k,n) += (dx[2]*dtinv*ar*fzma);
        }

        Real fzpa = wa(i,j,k+1,n);
        if (apz(i,j,k+1) != 0.0 && apz(i,j,k+1) != 1.0) {
            arp1 = 1.0/apz(i,j,k+1);
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0_rt,fcz(i,j,k+1,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0_rt,fcz(i,j,k+1,1)));
            Real fracx = flag(i,j,k+1).isConnected(ii-i,0,0) ? amrex::Math::abs(fcz(i,j,k+1,0)) : 0.0_rt;
            Real fracy = flag(i,j,k+1).isConnected(0,jj-j,0) ? amrex::Math::abs(fcz(i,j,k+1,1)) : 0.0_rt;

            fzpa = (1.0-fracx)*(1.0-fracy)*fzpa
                 +      fracx *(1.0-fracy)*wa(ii,j ,k+1,n)
                 +      fracy *(1.0-fracx)*wa(i ,jj,k+1,n)
                 +      fracx *     fracy *wa(ii,jj,k+1,n);
        }
        if (valid_cell && z_high) {
            fz(i,j,k+1,n) += (dx[2]*dtinv*arp1*fzpa);
        }

        // Compute quantities of cut-cells
        Real temp = divc(i,j,k,n);
        divc(i,j,k,n) = temp
                      + (1.0/vfrc(i,j,k))*( dtinv*( fxpa - fxma )
                      + dtinv*( fypa - fyma ) 
                      + dtinv*( fzpa - fzma ));
    }

#endif
    divc(i,j,k,n) *= -1.0;

    // Go ahead and make the redistwgt array here since we'll need it in flux_redistribute
    if (eb_weights_type == 0)
        { redistwgt(i,j,k) = 1.0; }
    else if (eb_weights_type == 1)
        { redistwgt(i,j,k) = q(i,j,k,QRHO)*( q(i,j,k,QEINT) +
#if (AMREX_SPACEDIM == 2)
                        0.5*(q(i,j,k,QU)*q(i,j,k,QU) + q(i,j,k,QV)*q(i,j,k,QV)) ); }
#else
                    0.5*(q(i,j,k,QU)*q(i,j,k,QU) + q(i,j,k,QV)*q(i,j,k,QV) + q(i,j,k,QW)*q(i,j,k,QW)) );
      }
#endif
    else if (eb_weights_type == 2)
        { redistwgt(i,j,k) = q(i,j,k,QRHO); }
    else if (eb_weights_type == 3 && vfrc(i,j,k) > vfmin)
        { redistwgt(i,j,k) = vfrc(i,j,k); }

}
