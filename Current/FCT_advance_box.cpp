
#include <CNS.H>
#include <CNS_hydro_K.H>
#include <FCT_diffusion_K.H>
#include <FCT_hydro_K.H>
#include <CNS_parm.H>
#include <cns_prob_parm.H>

#include <FCT_advance_box.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>

using namespace amrex;

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
    const Real OF_st = prob_parm->OF_st;

    if (phi < 1.0) {
        yP = (1.0 - yProg)*phi*(OF_st + 1.0)/(OF_st + phi);
        yF = yProg*phi/(OF_st + phi);
        yO = (yProg*phi + (1.0 - phi))*OF_st/(OF_st + phi);
    } else {
        yP = (OF_st + 1.0)/(OF_st + phi)*(1.0 - yProg);
        yF = (yProg + (phi - 1.0))/(OF_st + phi);
        yO = yProg*OF_st/(OF_st + phi);
    }
}

// Function to calculate phi and reaction progress variable
// For this compute the unburnt fuel and oxidizer mass fraction
AMREX_GPU_DEVICE
//AMREX_FORCE_INLINE
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

void
CNS::compute_dSdt_box_fct (const Box& bx,
                       Array4<Real const>& sofab,
                       Array4<Real      >& sfab,
                       Array4<Real      >& dsdtfab,
                       const std::array<FArrayBox*, AMREX_SPACEDIM>& flux,
                       Real dt, int rk)
{
    BL_PROFILE("CNS::compute_dSdt_box_fct()");

    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();

    const Real diff = diff1;
    const int react_do = do_react;
    Real dtinv = 1.0/dt;

    FArrayBox qtmp, diff_coeff;

    FArrayBox flt[BL_SPACEDIM], fld[BL_SPACEDIM], ut[BL_SPACEDIM],
              ud[BL_SPACEDIM], utmp, fracin, fracou;
    // FArrayBox flpd[BL_SPACEDIM];

    Parm const* lparm = d_parm;

    AMREX_D_TERM(auto const& fxfab = flux[0]->array();,
                 auto const& fyfab = flux[1]->array();,
                 auto const& fzfab = flux[2]->array(););

    // const Box& bxg2 = amrex::grow(bx,2);

    const Box& bxg4 = amrex::grow(bx,4);
    qtmp.resize(bxg4, NPRIM);
    auto const& q = qtmp.array();

    const Box& bxg3 = amrex::grow(bx,3);
    utmp.resize(bxg3,NEQNS);
    auto const& udfab = utmp.array();

    for (int dir = 0; dir < AMREX_SPACEDIM ; dir++) {
            const Box& bxtmp = amrex::surroundingNodes(bx,dir);
            flt[dir].resize(amrex::grow(bxtmp,3),NEQNS);
            fld[dir].resize(amrex::grow(bxtmp,3),NEQNS);
            ut[dir].resize(bxg3,NEQNS);
            ud[dir].resize(bxg3,NEQNS);

            // if(do_visc == 1){
                // flpd[dir].resize(amrex::grow(bxtmp,3),NEQNS);
                // flpd[dir].setVal<RunOn::Device>(0.0);
            // }

            flt[dir].setVal<RunOn::Device>(0.0);
            fld[dir].setVal<RunOn::Device>(0.0);
            ut[dir].setVal<RunOn::Device>(0.0);
            ud[dir].setVal<RunOn::Device>(0.0);
    }

    GpuArray<Array4<Real>, AMREX_SPACEDIM> fltx{ AMREX_D_DECL(flt[0].array(),
                                                flt[1].array(), flt[2].array())};
    GpuArray<Array4<Real>, AMREX_SPACEDIM> fldx{ AMREX_D_DECL(fld[0].array(),
                                                fld[1].array(), fld[2].array())};
    GpuArray<Array4<Real>, AMREX_SPACEDIM> utr{ AMREX_D_DECL(ut[0].array(),
                                                ut[1].array(), ut[2].array())};
    GpuArray<Array4<Real>, AMREX_SPACEDIM> udi{ AMREX_D_DECL(ud[0].array(),
                                                ud[1].array(), ud[2].array())};

    // GpuArray<Array4<Real>, AMREX_SPACEDIM> flpdx{ AMREX_D_DECL(flpd[0].array(),
    //                                             flpd[1].array(), flpd[2].array())};


    if (do_visc == 1)
    {
       diff_coeff.resize(bxg4, 1);
    }
    if(rk == 1){
        amrex::ParallelFor(bxg4,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ctoprim(i, j, k, sofab, sofab, q, *lparm);  });
    }else{
        amrex::ParallelFor(bxg4,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ctoprim(i, j, k, sofab, sfab, q, *lparm);  });
    }

    const Box& bxg1 = amrex::grow(bx,1);

    // --------------Computing the convective fluxes-----------------------
    // x-direction
    int cdir = 0;
    const Box& bxx = amrex::grow(amrex::surroundingNodes(bx,cdir),3);
        amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_x(i, j, k, q, fltx[0], sfab, sofab);   });

    // y-direction
    cdir = 1;
    const Box& byy = amrex::grow(amrex::surroundingNodes(bx,1),3);
        amrex::ParallelFor(byy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_y(i, j, k, q, fltx[1], sfab, sofab);   });

#if AMREX_SPACEDIM==3
    // z-direction
    cdir = 2;
    const Box& bzz = amrex::grow(amrex::surroundingNodes(bx,2),3);
        amrex::ParallelFor(bzz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_z(i, j, k, q, fltx[2], sfab, sofab);   });
#endif

    Gpu::streamSynchronize();
    Gpu::synchronize();

    // --------------Computing the diffusion fluxes--------------------
    Real nudiff = 1.0/6.0;
#if AMREX_SPACEDIM==3
    nudiff = 1.0/12.0;
#endif

    // x-direction
    amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_x(i, j, k, q, fldx[0], sofab, dxinv[0], dt, nudiff);   });
    // y-direction
    amrex::ParallelFor(byy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_y(i, j, k, q, fldx[1], sofab, dxinv[1], dt, nudiff);   });

#if AMREX_SPACEDIM==3
    // z-direction
    amrex::ParallelFor(bzz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_z(i, j, k, q, fldx[2], sofab, dxinv[2], dt, nudiff);   });

#endif

    Gpu::streamSynchronize();
    Gpu::synchronize();

    // Obtain low order solution, transported quantities, diffused quantities
    // Obtain the contribution of low-order solution to RHS (dSdt)
    amrex::ParallelFor(bxg3, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            AMREX_D_TERM(
            utr[0](i,j,k,n) = sofab(i,j,k,n) - dt*dxinv[0]*(fltx[0](i+1,j,k,n)-fltx[0](i,j,k,n));,
            utr[1](i,j,k,n) = sofab(i,j,k,n) - dt*dxinv[1]*(fltx[1](i,j+1,k,n)-fltx[1](i,j,k,n));,
            utr[2](i,j,k,n) = sofab(i,j,k,n) - dt*dxinv[2]*(fltx[2](i,j,k+1,n)-fltx[2](i,j,k,n));
            );

            AMREX_D_TERM(
            udi[0](i,j,k,n) = utr[0](i,j,k,n) + (fldx[0](i+1,j,k,n)-fldx[0](i,j,k,n));,
            udi[1](i,j,k,n) = utr[1](i,j,k,n) + (fldx[1](i,j+1,k,n)-fldx[1](i,j,k,n));,
            udi[2](i,j,k,n) = utr[2](i,j,k,n) + (fldx[2](i,j,k+1,n)-fldx[2](i,j,k,n));
            );

            udfab(i,j,k,n)  = sofab(i,j,k,n) - dt*dxinv[0]*(fltx[0](i+1,j,k,n) - fltx[0](i,j,k,n))
                                             + (fldx[0](i+1,j,k,n) - fldx[0](i,j,k,n))
                                             - dt*dxinv[1]*(fltx[1](i,j+1,k,n) - fltx[1](i,j,k,n))
                                             + (fldx[1](i,j+1,k,n) - fldx[1](i,j,k,n))
#if AMREX_SPACEDIM==3
                                             - dt*dxinv[2]*(fltx[2](i,j,k+1,n) - fltx[2](i,j,k,n))
                                             + (fldx[2](i,j,k+1,n) - fldx[2](i,j,k,n))
#endif
                                             ;
        });

    // Add the reaction source terms to the energy and F mass fraction equations
    if(do_react == 1){

    	    const ProbParm* prob_parm = d_prob_parm; // Pointer to device-side problem parameters

	    amrex::ParallelFor(bxg3,
	    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
	    {
		// Get local mass fractions
		Real yO = sofab(i, j, k, URHOY_O) / sofab(i, j, k, URHO);
		Real yF = sofab(i, j, k, URHOY_F) / sofab(i, j, k, URHO);
		Real yP = sofab(i, j, k, URHOY_P) / sofab(i, j, k, URHO);

		// Calculate unburnt mixture equivalence ratio
		Real phi_unburnt = phiFromYoYfYp(yO, yF, yP, prob_parm);

		// Calculate local equivalence ratio (actual current ratio)
		Real phi_local = phiLocalFromYoYfYp(yO, yF, yP, prob_parm);

		// Calculate reconstructed initial mass fractions
		Real yO0 = yO + (prob_parm->OF_st / (1 + prob_parm->OF_st)) * yP;
		Real yF0 = yF + (1 / (1 + prob_parm->OF_st)) * yP;

		// Calculate reaction progress variable
		// This represents how much of the original reactants have been consumed
		Real Y_react = (phi_unburnt < 1.0) ? yF/yF0 : yO/yO0;

		if (yF0 == 0 || yO0 == 0) {
		    amrex::Print() << "Zero denominator detected at i=" << i << " j=" << j << " k=" << k
		                   << " yF0: " << yF0 << " yO0: " << yO0 << std::endl;
		}


		// Update CDM parameters based on unburnt mixture equivalence ratio
		Real pre_exp_tmp, Ea_nd_tmp, q_nd_tmp, kappa_0_tmp;
		Parm::Calculate_CDM_Parameters(phi_unburnt, pre_exp_tmp, Ea_nd_tmp, q_nd_tmp, kappa_0_tmp);


		// Calculate reaction rate using reaction progress variable
		Real omegarho = -sofab(i, j, k, URHO) * pre_exp_tmp * Y_react * std::exp(-Ea_nd_tmp / (lparm->Ru * sofab(i, j, k, UTEMP)));

		// Calculate species-specific reaction rates based on phi_unburnt (the original mixture)
		Real omega_F, omega_Ox, omega_Pr;
		Real s = prob_parm->OF_st; // Stoichiometric O/F ratio

                //amrex::Print() << "Reaction Rate (omegarho): " << omegarho 
                  //             << " Y_react: " << Y_react
               	//	       << " exp(-Ea/RT): " << std::exp(-Ea_nd_tmp / (lparm->Ru * sofab(i, j, k, UTEMP))) 
               	//	       << " Temperature: " << sofab(i, j, k, UTEMP) << std::endl;

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

		// Update energy in all dimensions
		AMREX_D_TERM(
		    udi[0](i,j,k,UEDEN) = udi[0](i,j,k,UEDEN) - dt*omegarho*lparm->q_dim;,
		    udi[1](i,j,k,UEDEN) = udi[1](i,j,k,UEDEN) - dt*omegarho*lparm->q_dim;,
		    udi[2](i,j,k,UEDEN) = udi[2](i,j,k,UEDEN) - dt*omegarho*lparm->q_dim;
		);

                //amrex::Print() << "Energy Source: " << dt * omegarho * lparm->q_dim << std::endl;
		//if (std::isnan(omegarho) || std::isnan(dt * omegarho * lparm->q_dim)) {
		  //  amrex::Print() << "NaN Energy Source at i=" << i << " j=" << j << " k=" << k << std::endl;
		//}

		// Update Fuel mass fraction in all dimensions
		AMREX_D_TERM(
		    udi[0](i, j, k, URHOY_F) = amrex::max(0.0, udi[0](i, j, k, URHOY_F) + dt * omega_F);,
		    udi[1](i, j, k, URHOY_F) = amrex::max(0.0, udi[1](i, j, k, URHOY_F) + dt * omega_F);,
		    udi[2](i, j, k, URHOY_F) = amrex::max(0.0, udi[2](i, j, k, URHOY_F) + dt * omega_F);
		);

		// Update Oxidizer mass fraction in all dimensions
		AMREX_D_TERM(
		    udi[0](i, j, k, URHOY_O) = amrex::max(0.0, udi[0](i, j, k, URHOY_O) + dt * omega_Ox);,
		    udi[1](i, j, k, URHOY_O) = amrex::max(0.0, udi[1](i, j, k, URHOY_O) + dt * omega_Ox);,
		    udi[2](i, j, k, URHOY_O) = amrex::max(0.0, udi[2](i, j, k, URHOY_O) + dt * omega_Ox);
		);

		// Update Product mass fraction in all dimensions
		AMREX_D_TERM(
		    udi[0](i, j, k, URHOY_P) = amrex::max(0.0, udi[0](i, j, k, URHOY_P) + dt * omega_Pr);,
		    udi[1](i, j, k, URHOY_P) = amrex::max(0.0, udi[1](i, j, k, URHOY_P) + dt * omega_Pr);,
		    udi[2](i, j, k, URHOY_P) = amrex::max(0.0, udi[2](i, j, k, URHOY_P) + dt * omega_Pr);
		);

		// Update the udfab array for all species
		udfab(i,j,k,UEDEN) = udfab(i,j,k,UEDEN) - dt*omegarho*lparm->q_dim;

		//amrex::Print() << "Heat Release (q): " << lparm->q_dim << std::endl;
		//amrex::Print() << "Energy Source: " << dt * omegarho * lparm->q_dim << std::endl;

		// Update Fuel
		udfab(i, j, k, URHOY_F) = amrex::min(udfab(i, j, k, URHO), udfab(i, j, k, URHOY_F) + dt * omega_F);
		udfab(i, j, k, URHOY_F) = amrex::max(0.0, udfab(i, j, k, URHOY_F));

		// Update Oxidizer
		udfab(i, j, k, URHOY_O) = amrex::min(udfab(i, j, k, URHO), udfab(i, j, k, URHOY_O) + dt * omega_Ox);
		udfab(i, j, k, URHOY_O) = amrex::max(0.0, udfab(i, j, k, URHOY_O));

		// Update Product
		udfab(i, j, k, URHOY_P) = amrex::min(udfab(i, j, k, URHO), udfab(i, j, k, URHOY_P) + dt * omega_Pr);
		udfab(i, j, k, URHOY_P) = amrex::max(0.0, udfab(i, j, k, URHOY_P));

		// Update low-order transported quantities
		AMREX_D_TERM(
		    utr[0](i,j,k,URHOY_F) = amrex::max(0.0,amrex::min(utr[0](i,j,k,URHO),utr[0](i,j,k,URHOY_F)));,
		    utr[1](i,j,k,URHOY_F) = amrex::max(0.0,amrex::min(utr[1](i,j,k,URHO),utr[1](i,j,k,URHOY_F)));,
		    utr[2](i,j,k,URHOY_F) = amrex::max(0.0,amrex::min(utr[2](i,j,k,URHO),utr[2](i,j,k,URHOY_F)));
		);

		AMREX_D_TERM(
		    utr[0](i,j,k,URHOY_O) = amrex::max(0.0,amrex::min(utr[0](i,j,k,URHO),utr[0](i,j,k,URHOY_O)));,
		    utr[1](i,j,k,URHOY_O) = amrex::max(0.0,amrex::min(utr[1](i,j,k,URHO),utr[1](i,j,k,URHOY_O)));,
		    utr[2](i,j,k,URHOY_O) = amrex::max(0.0,amrex::min(utr[2](i,j,k,URHO),utr[2](i,j,k,URHOY_O)));
		);

		AMREX_D_TERM(
		    utr[0](i,j,k,URHOY_P) = amrex::max(0.0,amrex::min(utr[0](i,j,k,URHO),utr[0](i,j,k,URHOY_P)));,
		    utr[1](i,j,k,URHOY_P) = amrex::max(0.0,amrex::min(utr[1](i,j,k,URHO),utr[1](i,j,k,URHOY_P)));,
		    utr[2](i,j,k,URHOY_P) = amrex::max(0.0,amrex::min(utr[2](i,j,k,URHO),utr[2](i,j,k,URHOY_P)));
		);
	    });
	}

    Gpu::streamSynchronize();
    Gpu::synchronize();

    const Box& bxd = amrex::surroundingNodes(bx,0);
    amrex::ParallelFor(bxd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
            fxfab(i,j,k,n) = fltx[0](i,j,k,n) - dx[0]*dtinv*fldx[0](i,j,k,n);
        });

    const Box& byd = amrex::surroundingNodes(bx,1);
    amrex::ParallelFor(byd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
           fyfab(i,j,k,n) = fltx[1](i,j,k,n) - dx[1]*dtinv*fldx[1](i,j,k,n);
        });

#if AMREX_SPACEDIM==3
     const Box& bzd = amrex::surroundingNodes(bx,2);
        amrex::ParallelFor(bzd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
            fzfab(i,j,k,n) = fltx[2](i,j,k,n) - dx[2]*dtinv*fldx[2](i,j,k,n);
        });
#endif

    // Compute the physical diffusion terms

    if (do_visc == 1)
    {
       auto const& coefs = diff_coeff.array();
       if(use_const_visc == 1 ) {
          amrex::ParallelFor(bxg4,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              fct_constcoef(i, j, k, coefs, *lparm);
          });
       } else {
        amrex::ParallelFor(bxg4,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              fct_diffcoef(i, j, k, q, coefs, *lparm);
          });
       }

       // -------------x-direction-----------------
       const Box& bxx = amrex::grow(amrex::surroundingNodes(bx,0),3);
       amrex::ParallelFor(bxx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
           fct_phys_diff_x(i, j, k, q, sofab, coefs, dxinv, fltx[0], react_do, *lparm);
       });

       // ------------y-direction-----------------
       const Box& byy = amrex::grow(amrex::surroundingNodes(bx,1),3);
       amrex::ParallelFor(byy,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
           fct_phys_diff_y(i, j, k, q, sofab, coefs, dxinv, fltx[1], react_do, *lparm);
       });

#if AMREX_SPACEDIM==3
       // ------------z-direction-----------------
       const Box& bzz = amrex::grow(amrex::surroundingNodes(bx,2),3);
       amrex::ParallelFor(bzz,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
           fct_phys_diff_z(i, j, k, q, sofab, coefs, dxinv, fltx[2], react_do, *lparm);
       });
#endif

       // --------------Update the low-order quantities by including physical diffusion terms-------------------
       amrex::ParallelFor(bxg3, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {

            AMREX_D_TERM(
            udi[0](i,j,k,n) = udi[0](i,j,k,n) + dt*dxinv[0]*(fltx[0](i+1,j,k,n)-fltx[0](i,j,k,n));,
            udi[1](i,j,k,n) = udi[1](i,j,k,n) + dt*dxinv[1]*(fltx[1](i,j+1,k,n)-fltx[1](i,j,k,n));,
            udi[2](i,j,k,n) = udi[2](i,j,k,n) + dt*dxinv[2]*(fltx[2](i,j,k+1,n)-fltx[2](i,j,k,n));
            );

            udfab(i,j,k,n)  = udfab(i,j,k,n) + dt*dxinv[0]*(fltx[0](i+1,j,k,n) - fltx[0](i,j,k,n))
                                             + dt*dxinv[1]*(fltx[1](i,j+1,k,n) - fltx[1](i,j,k,n))
#if AMREX_SPACEDIM==3
                                             + dt*dxinv[2]*(fltx[2](i,j,k+1,n) - fltx[2](i,j,k,n))
#endif
                                             ;
        });

       amrex::ParallelFor(bxg3,
	    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
	    {
		// Fuel mass fraction bounds
		AMREX_D_TERM(
		udi[0](i,j,k,URHOY_F) = amrex::max(0.0,amrex::min(udi[0](i,j,k,URHO),udi[0](i,j,k,URHOY_F)));,
		udi[1](i,j,k,URHOY_F) = amrex::max(0.0,amrex::min(udi[1](i,j,k,URHO),udi[1](i,j,k,URHOY_F)));,
		udi[2](i,j,k,URHOY_F) = amrex::max(0.0,amrex::min(udi[2](i,j,k,URHO),udi[2](i,j,k,URHOY_F)));
		);

		// Oxidizer mass fraction bounds
		AMREX_D_TERM(
		udi[0](i,j,k,URHOY_O) = amrex::max(0.0,amrex::min(udi[0](i,j,k,URHO),udi[0](i,j,k,URHOY_O)));,
		udi[1](i,j,k,URHOY_O) = amrex::max(0.0,amrex::min(udi[1](i,j,k,URHO),udi[1](i,j,k,URHOY_O)));,
		udi[2](i,j,k,URHOY_O) = amrex::max(0.0,amrex::min(udi[2](i,j,k,URHO),udi[2](i,j,k,URHOY_O)));
		);

		// Product mass fraction bounds
		AMREX_D_TERM(
		udi[0](i,j,k,URHOY_P) = amrex::max(0.0,amrex::min(udi[0](i,j,k,URHO),udi[0](i,j,k,URHOY_P)));,
		udi[1](i,j,k,URHOY_P) = amrex::max(0.0,amrex::min(udi[1](i,j,k,URHO),udi[1](i,j,k,URHOY_P)));,
		udi[2](i,j,k,URHOY_P) = amrex::max(0.0,amrex::min(udi[2](i,j,k,URHO),udi[2](i,j,k,URHOY_P)));
		);

		// Apply bounds to unified solution array
		udfab(i,j,k,URHOY_F) = amrex::max(0.0,amrex::min(udfab(i,j,k,URHO),udfab(i,j,k,URHOY_F)));
		udfab(i,j,k,URHOY_O) = amrex::max(0.0,amrex::min(udfab(i,j,k,URHO),udfab(i,j,k,URHOY_O)));
		udfab(i,j,k,URHOY_P) = amrex::max(0.0,amrex::min(udfab(i,j,k,URHO),udfab(i,j,k,URHOY_P)));
	    });

        const Box& bxd = amrex::surroundingNodes(bx,0);
        amrex::ParallelFor(bxd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
            fxfab(i,j,k,n) -= fltx[0](i,j,k,n);
        });

        const Box& byd = amrex::surroundingNodes(bx,1);
        amrex::ParallelFor(byd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
           fyfab(i,j,k,n) -= fltx[1](i,j,k,n);
        });

#if AMREX_SPACEDIM==3
        const Box& bzd = amrex::surroundingNodes(bx,2);
        amrex::ParallelFor(bzd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
            fzfab(i,j,k,n) -= fltx[2](i,j,k,n);
        });
#endif
    }

    // Compute the anti-diffusion fluxes and store these in fldx
        Real mudiff = 0.0;
#if AMREX_SPACEDIM==3
        mudiff = 1.0/12.0;
#endif
        // x-direction
        const Box& bxxg2 = amrex::grow(amrex::surroundingNodes(bx,0),2);
        amrex::ParallelFor(bxxg2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_x(i, j, k, q, fldx[0], sofab, utr[0], dxinv[0], dt, diff, mudiff);   });

        // y-direction
        const Box& byyg2 = amrex::grow(amrex::surroundingNodes(bx,1),2);
        amrex::ParallelFor(byyg2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_y(i, j, k, q, fldx[1], sofab, utr[1], dxinv[1], dt, diff, mudiff);   });

#if AMREX_SPACEDIM==3
        // z-direction
        const Box& bzzg2 = amrex::grow(amrex::surroundingNodes(bx,1),2);
        amrex::ParallelFor(bzzg2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_z(i, j, k, q, fldx[2], sofab, utr[2], dxinv[2], dt, diff, mudiff);   });
#endif

        Gpu::streamSynchronize();
        Gpu::synchronize();

//     // Prelimit anti-diffusion fluxes
    const Box& bxnd1 = amrex::grow(amrex::surroundingNodes(bx,0),1);
        amrex::ParallelFor(bxnd1, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_x(i, j, k, n, fldx[0], udi[0]); });

    const Box& bynd1 = amrex::grow(amrex::surroundingNodes(bx,1),1);
        amrex::ParallelFor(bynd1, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_y(i, j, k, n, fldx[1], udi[1]); });

#if AMREX_SPACEDIM==3
    const Box& bznd1 = amrex::grow(amrex::surroundingNodes(bx,2),1);
        amrex::ParallelFor(bznd1, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_z(i, j, k, n, fldx[2], udi[2]); });
#endif

    Gpu::streamSynchronize();
    Gpu::synchronize();

    // Calculate the total incoming and outgoing antidiffusive fluxes in each cell
    fracin.resize(amrex::grow(bx,1),NEQNS);
    fracin.setVal<RunOn::Device>(1.0);
    auto const& finfab = fracin.array();

    fracou.resize(amrex::grow(bx,1),NEQNS);
    fracou.setVal<RunOn::Device>(1.0);
    auto const& foufab = fracou.array();

    amrex::ParallelFor(bxg1, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_compute_frac_fluxes(i, j, k, n, AMREX_D_DECL(fldx[0], fldx[1], fldx[2]),
                                        finfab, foufab, udfab);  });

    Gpu::streamSynchronize();
    Gpu::synchronize();

// //     // Compute the corrected fluxes (no ghost cells)
    const Box& bxnd = amrex::surroundingNodes(bx,0);
    amrex::ParallelFor(bxnd, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {   fct_correct_fluxes_x(i, j, k, n, fldx[0], finfab, foufab); });

    const Box& bynd = amrex::surroundingNodes(bx,1);
    amrex::ParallelFor(bynd, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {   fct_correct_fluxes_y(i, j, k, n, fldx[1], finfab, foufab); });

    Gpu::streamSynchronize();
    Gpu::synchronize();
//     // --------------- Compute the corrected z-fluxes (no ghost cells) --------------------
#if AMREX_SPACEDIM==3
    const Box& bznd = amrex::surroundingNodes(bx,2);
    amrex::ParallelFor(bznd, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {   fct_correct_fluxes_z(i, j, k, n, fldx[2], finfab, foufab); });
#endif

    // Store the corrected fluxes in flux MultiFab for dSdt update
    amrex::ParallelFor(bxd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
            fxfab(i,j,k,n) += dx[0]*dtinv*fldx[0](i,j,k,n);
        });

    amrex::ParallelFor(byd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
           fyfab(i,j,k,n) += dx[1]*dtinv*fldx[1](i,j,k,n);
       });

#if AMREX_SPACEDIM==3
        amrex::ParallelFor(bzd, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {   fzfab(i,j,k,n) += dx[2]*dtinv*fldx[2](i,j,k,n); });
#endif

    amrex::ParallelFor(bx, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        fct_flux_to_dudt(i, j, k, n, dsdtfab, AMREX_D_DECL(fxfab,fyfab,fzfab), dxinv);
    });

    if(do_react == 1){
	    const ProbParm* prob_parm = d_prob_parm;

	    amrex::ParallelFor(bx,
		    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
		    {
			// Get local mass fractions
			Real yO = sofab(i, j, k, URHOY_O) / sofab(i, j, k, URHO);
			Real yF = sofab(i, j, k, URHOY_F) / sofab(i, j, k, URHO);
			Real yP = sofab(i, j, k, URHOY_P) / sofab(i, j, k, URHO);

			// Calculate phi and reaction progress variable
			Real phi, Y_react;
			phiAndYReact(yO, yF, yP, phi, Y_react, prob_parm);

			// Calculate reaction rate
			Real omegarho = -sofab(i, j, k, URHO) * lparm->pre_exp * Y_react
				      * std::exp(-lparm->Ea_dim / (lparm->Ru * sofab(i, j, k, UTEMP)));

			// Calculate species-specific reaction rates based on phi
			Real omega_F, omega_O, omega_P;
			Real s = prob_parm->OF_st; // Stoichiometric O/F ratio

			if (phi < 1.0) {
			    // F-lean condition
			    omega_F = (phi / (s + phi)) * omegarho;
			    omega_O = (s * phi / (s + phi)) * omegarho;
			    omega_P = -((s + 1) * phi / (s + phi)) * omegarho;
			}
			else {
			    // F-rich condition (including stoichiometric)
			    omega_F = (1.0 / (s + phi)) * omegarho;
			    omega_O = (s / (s + phi)) * omegarho;
			    omega_P = -((s + 1.0) / (s + phi)) * omegarho;
			}

			// Update the conserved variables with reaction source terms
			dsdtfab(i, j, k, UEDEN) = dsdtfab(i, j, k, UEDEN) - omegarho * lparm->q_dim;
			dsdtfab(i, j, k, URHOY_F) = dsdtfab(i, j, k, URHOY_F) + omega_F;
			dsdtfab(i, j, k, URHOY_O) = dsdtfab(i, j, k, URHOY_O) + omega_O;
			dsdtfab(i, j, k, URHOY_P) = dsdtfab(i, j, k, URHOY_P) + omega_P;
		    });
	}

    Gpu::synchronize();
    Gpu::streamSynchronize();
}
