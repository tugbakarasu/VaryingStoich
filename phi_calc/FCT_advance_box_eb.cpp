
#include <CNS.H>
#include <CNS_hydro_K.H>
#include <CNS_hydro_eb_K.H>
#include <CNS_divop_K.H>
// #include <CNS_diffusion_eb_K.H>
#include <FCT_diffusion_eb_K.H>
#include <FCT_hydro_K.H>
#include <FCT_hydro_eb_K.H>
#include <FCT_divop_K.H>

#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>

#if (AMREX_SPACEDIM == 2)
#include <AMReX_EBMultiFabUtil_2D_C.H>
#elif (AMREX_SPACEDIM == 3)
#include <AMReX_EBMultiFabUtil_3D_C.H>
#endif

using namespace amrex;

void
CNS::compute_dSdt_box_eb_fct (const Box& bx,
                          Array4<Real const> const& sofab,
                          Array4<Real      > const& sfab,
                          Array4<Real      > const& dsdtfab,
                          std::array<FArrayBox*, AMREX_SPACEDIM> const& flux,
                          Array4<EBCellFlag const> const& flag,
                          Array4<Real       const> const& vfrac,
                          AMREX_D_DECL(
                          Array4<Real       const> const& apx,
                          Array4<Real       const> const& apy,
                          Array4<Real       const> const& apz),
                          AMREX_D_DECL(
                          Array4<Real       const> const& fcx,
                          Array4<Real       const> const& fcy,
                          Array4<Real       const> const& fcz),
                          Array4<Real       const> const& bcent,
                          int as_crse,
                          Array4<Real            > const& drho_as_crse,
                          Array4<int        const> const& rrflag_as_crse,
                          int as_fine,
                          Array4<Real            > const& dm_as_fine,
                          Array4<int        const> const& lev_mask,
                          Real dt, int rk)
{
    BL_PROFILE("CNS::compute_dSdt_box_eb_fct()");

    const Box& bxg1 = amrex::grow(bx,1);
    const Box& bxg2 = amrex::grow(bx,2);
    const Box& bxg3 = amrex::grow(bx,3);
    const Box& bxg4 = amrex::grow(bx,4);
    const Box& bxg5 = amrex::grow(bx,5);
    const Box& bxg6 = amrex::grow(bx,6);
    const Box& bxg7 = amrex::grow(bx,7);
    const Box& bxg8 = amrex::grow(bx,8);

    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();
    const int react_do = do_react;

    const Real diff = diff1;
    const Real diff_cc = diffcc;
    const Real vfmin = vfc_threshold;

    // // Quantities for redistribution
    FArrayBox divc,optmp,redistwgt,delta_m;
    divc.resize(bxg7,NEQNS);
    optmp.resize(bxg7,NEQNS);
    delta_m.resize(bxg7,NEQNS);
    redistwgt.resize(bxg7,1);

    // Set to zero just in case
    divc.setVal<RunOn::Device>(0.0);
    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);
    redistwgt.setVal<RunOn::Device>(0.0);

    // Primitive variables
    FArrayBox qtmp;
    qtmp.resize(bxg8, NPRIM);

    // // Variables for FCT
    FArrayBox flt[BL_SPACEDIM], fld[BL_SPACEDIM], flpd[BL_SPACEDIM],
            ut[BL_SPACEDIM], ud[BL_SPACEDIM], utmp, fracin, fracou, divc_tmp;

    utmp.resize(bxg7,NEQNS);
    utmp.setVal<RunOn::Device>(0.0);
    auto const& udfab = utmp.array();

    divc_tmp.resize(bxg7,NEQNS);
    divc_tmp.setVal<RunOn::Device>(0.0);
    auto const& dcfab = divc_tmp.array();  

    for (int dir = 0; dir < AMREX_SPACEDIM ; dir++) {
            const Box& bxtmp = amrex::surroundingNodes(bx,dir);
            flt[dir].resize(amrex::grow(bxtmp,NUM_GROW-1),NEQNS);  
            fld[dir].resize(amrex::grow(bxtmp,NUM_GROW-1),NEQNS); 
            ut[dir].resize(bxg7,NEQNS);  
            ud[dir].resize(bxg7,NEQNS);

            if(do_visc == 1){
                flpd[dir].resize(amrex::grow(bxtmp,NUM_GROW-1),NEQNS); 
                flpd[dir].setVal<RunOn::Device>(0.0); 
            }

            flt[dir].setVal<RunOn::Device>(0.0); 
            fld[dir].setVal<RunOn::Device>(0.0); 
            ut[dir].setVal<RunOn::Device>(0.0); 
            ud[dir].setVal<RunOn::Device>(0.0); 
    }

    Gpu::streamSynchronize();
    Gpu::synchronize();

    GpuArray<Array4<Real>, AMREX_SPACEDIM> fltx{ AMREX_D_DECL(flt[0].array(), 
                                                flt[1].array(), flt[2].array())}; 
    GpuArray<Array4<Real>, AMREX_SPACEDIM> fldx{ AMREX_D_DECL(fld[0].array(), 
                                                fld[1].array(), fld[2].array())}; 
    GpuArray<Array4<Real>, AMREX_SPACEDIM> utr{ AMREX_D_DECL(ut[0].array(), 
                                                ut[1].array(), ut[2].array())}; 
    GpuArray<Array4<Real>, AMREX_SPACEDIM> udi{ AMREX_D_DECL(ud[0].array(), 
                                                ud[1].array(), ud[2].array())}; 
    GpuArray<Array4<Real>, AMREX_SPACEDIM> flpdx{ AMREX_D_DECL(flpd[0].array(), 
                                                flpd[1].array(), flpd[2].array())}; 

    FArrayBox diff_coeff;

    if (do_visc == 1)
    {
       diff_coeff.resize(bxg8, 1);
    }

    FArrayBox flux_tmp[AMREX_SPACEDIM];
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        flux_tmp[idim].resize(amrex::surroundingNodes(bxg3,idim),NEQNS);
        flux_tmp[idim].setVal<RunOn::Device>(0.);
    }

    Parm const* lparm = d_parm;

    AMREX_D_TERM(auto const& fxfab = flux_tmp[0].array();,
                 auto const& fyfab = flux_tmp[1].array();,
                 auto const& fzfab = flux_tmp[2].array(););

    auto const& q = qtmp.array();

    GpuArray<Real,3> weights;
    weights[0] = 0.;
    weights[1] = 1.;
    weights[2] = 0.5;

    // Initialize dm_as_fine to 0
    if (as_fine)
    {
        amrex::ParallelFor(bxg1, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
           dm_as_fine(i,j,k,n) = 0.;
        });
    }

    if(rk == 1){
        amrex::ParallelFor(bxg8,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ctoprim(i, j, k, sofab, sofab, q, *lparm); });
    }else{
        amrex::ParallelFor(bxg8,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ctoprim(i, j, k, sofab, sfab, q, *lparm);  });
    }

    // x-direction
    int cdir = 0;
    const Box& bxx = amrex::grow(amrex::surroundingNodes(bx,cdir),NUM_GROW-1);
        amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_eb_x(i, j, k, q, fltx[0], sfab, sofab, flag, vfrac, vfmin);   });

    // y-direction
    cdir = 1;
    const Box& byy = amrex::grow(amrex::surroundingNodes(bx,cdir),NUM_GROW-1);
        amrex::ParallelFor(byy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_eb_y(i, j, k, q, fltx[1], sfab, sofab, flag, vfrac, vfmin);   });

#if AMREX_SPACEDIM==3
    // z-direction
    cdir = 2;
    const Box& bzz = amrex::grow(amrex::surroundingNodes(bx,cdir),NUM_GROW-1);
        amrex::ParallelFor(bzz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_con_flux_eb_z(i, j, k, q, fltx[2], sfab, sofab, flag, vfrac, vfmin);   });
#endif

    Gpu::streamSynchronize();
    Gpu::synchronize();

    // ------------Computing the diffusion fluxes--------------------
    Real nudiff = 1.0/6.0;
#if AMREX_SPACEDIM==3
    nudiff = 1.0/12.0;
#endif

    // x-direction
    amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_eb_x(i, j, k, q, fldx[0], sofab, dxinv[0], dt, nudiff, flag, 
            vfrac, apx, vfmin);   });

    // y-direction
    amrex::ParallelFor(byy,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_eb_y(i, j, k, q, fldx[1], sofab, dxinv[1], dt, nudiff, flag, 
            vfrac, apy, vfmin);   });

#if AMREX_SPACEDIM==3
    // z-direction
    amrex::ParallelFor(bzz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_diff_flux_eb_z(i, j, k, q, fldx[2], sofab, dxinv[2], dt, nudiff, flag, 
            vfrac, apz, vfmin);   });
#endif

    if (do_visc == 1)
    {
       auto const& coefs = diff_coeff.array();
       if(use_const_visc == 1 ) {
          amrex::ParallelFor(bxg8,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              fct_constcoef_eb(i, j, k, flag, coefs, *lparm);
          });
       } else {
          amrex::ParallelFor(bxg8,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              fct_diffcoef_eb(i, j, k, q, flag, coefs, *lparm);
          });
       }

        const Box& bxx = amrex::grow(amrex::surroundingNodes(bx,0),NUM_GROW-1);
        amrex::ParallelFor(bxx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            fct_phys_diff_eb_x(i, j, k, q, sofab, coefs, flag, dxinv, weights, flpdx[0], react_do, *lparm);
        });

       const Box& byy = amrex::grow(amrex::surroundingNodes(bx,1),NUM_GROW-1);
       amrex::ParallelFor(byy,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
           fct_phys_diff_eb_y(i, j, k, q, sofab, coefs, flag, dxinv, weights, flpdx[1], react_do, *lparm);
       });

#if AMREX_SPACEDIM==3
       const Box& bzz = amrex::grow(amrex::surroundingNodes(bx,2),NUM_GROW-1);
       amrex::ParallelFor(bzz,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
           fct_phys_diff_eb_z(i, j, k, q, sofab, coefs, flag, dxinv, weights, flpdx[2], react_do, *lparm);
       });
#endif
    }

    Gpu::streamSynchronize();
    Gpu::synchronize();

    // We have obtained the convective and diffusive fluxes
    // Now, compute the divergence for the low-order solution

    // These are the fluxes we computed above -- they live at face centers
    AMREX_D_TERM(auto const& fx_in_arr = flux_tmp[0].array();,
                 auto const& fy_in_arr = flux_tmp[1].array();,
                 auto const& fz_in_arr = flux_tmp[2].array(););

    // // These are the fluxes on face centroids -- they are defined in eb_compute_div
    // //    and are the fluxes that go into the flux registers
    AMREX_D_TERM(auto const& fx_out_arr = flux[0]->array();,
                 auto const& fy_out_arr = flux[1]->array();,
                 auto const& fz_out_arr = flux[2]->array(););

    auto const& blo = bx.smallEnd();
    auto const& bhi = bx.bigEnd();

    // // Because we are going to redistribute, we put the divergence into divc
    // //    rather than directly into dsdtfab
    auto const& divc_arr = divc.array();

    bool l_do_visc = do_visc;
    bool l_do_reac = do_react;
    auto l_eb_weights_type = eb_weights_type;

    auto const& coefs = diff_coeff.array();
    auto const& redistwgt_arr = redistwgt.array();

    Real dtinv = 1.0/dt;

    amrex::ParallelFor(bxg7, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
       // This does the divergence but not the redistribution -- we will do that later
       // We do compute the weights here though
       fct_eb_compute_div_loworder(i,j,k,n,blo,bhi,q,sofab,
                      AMREX_D_DECL(utr[0] ,utr[1] ,utr[2]),
                      AMREX_D_DECL(udi[0] ,udi[1] ,udi[2]), udfab, dcfab,
                      AMREX_D_DECL(fltx[0] ,fltx[1] ,fltx[2]),
                      AMREX_D_DECL(fldx[0] ,fldx[1] ,fldx[2]),
                      AMREX_D_DECL(flpdx[0] ,flpdx[1] ,flpdx[2]),
                      AMREX_D_DECL(fx_out_arr,fy_out_arr,fz_out_arr),
                      flag, vfrac, bcent, coefs, redistwgt_arr,
                      AMREX_D_DECL(apx, apy, apz), AMREX_D_DECL(fcx, fcy, fcz), 
                      dxinv, dx, dtinv, dt, *lparm, l_eb_weights_type, l_do_visc, l_do_reac, 
                      rk, vfmin);
    });

    Gpu::streamSynchronize();
    Gpu::synchronize();

    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);

    auto const& optmp_arr = optmp.array();
    auto const& del_m_arr = delta_m.array();

    // // Store the low-order fluxes in dsdtfab before redistribution 
    // // (dsdtfab will be redistributed after correction step)
    amrex::ParallelFor(bxg7, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            divc_arr(i,j,k,n) = dcfab(i,j,k,n);
        });

    // Now do redistribution
    fct_flux_redistribute_loworder(bx,udfab,optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 0, vfmin);
    Gpu::streamSynchronize();
    Gpu::synchronize();

    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);

    fct_flux_redistribute_loworder(bx,utr[0],optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 0, vfmin);
    Gpu::streamSynchronize();
    Gpu::synchronize();

    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);

    fct_flux_redistribute_loworder(bx,utr[1],optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 0, vfmin);
    Gpu::streamSynchronize();
    Gpu::synchronize();

#if AMREX_SPACEDIM==3
    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);

    fct_flux_redistribute_loworder(bx,utr[2],optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 0, vfmin);
    Gpu::streamSynchronize();
    Gpu::synchronize();
#endif

    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);
    fct_flux_redistribute_loworder(bx,udi[0],optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 0, vfmin);
    Gpu::streamSynchronize();
    Gpu::synchronize();

    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);
    fct_flux_redistribute_loworder(bx,udi[1],optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 0, vfmin);
    Gpu::streamSynchronize();
    Gpu::synchronize();

#if AMREX_SPACEDIM==3
    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);
    fct_flux_redistribute_loworder(bx,udi[2],optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 0, vfmin);
    Gpu::streamSynchronize();
    Gpu::synchronize();
#endif

    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);

    // Obtain the low-order quantities
    amrex::ParallelFor(bxg7, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            AMREX_D_TERM(
            utr[0](i,j,k,n) = sofab(i,j,k,n) + dt*utr[0](i,j,k,n);,
            utr[1](i,j,k,n) = sofab(i,j,k,n) + dt*utr[1](i,j,k,n);,
            utr[2](i,j,k,n) = sofab(i,j,k,n) + dt*utr[2](i,j,k,n);
            );

            AMREX_D_TERM(
            udi[0](i,j,k,n) = sofab(i,j,k,n) + dt*udi[0](i,j,k,n);,
            udi[1](i,j,k,n) = sofab(i,j,k,n) + dt*udi[1](i,j,k,n);,
            udi[2](i,j,k,n) = sofab(i,j,k,n) + dt*udi[2](i,j,k,n);
            );

            udfab(i,j,k,n)  = sofab(i,j,k,n) + dt*udfab(i,j,k,n);
        });

    Gpu::streamSynchronize();
    Gpu::synchronize();

    if(do_react){
        // Ensure that the species mass concentration solution is within bounds [0,rho]
    amrex::ParallelFor(bxg7,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            AMREX_D_TERM(
            utr[0](i,j,k,URHOY_FUEL) = amrex::max(Real(0.0), amrex::min(utr[0](i,j,k,URHOY_FUEL), utr[0](i,j,k,URHO)));,
            utr[1](i,j,k,URHOY_FUEL) = amrex::max(Real(0.0), amrex::min(utr[1](i,j,k,URHOY_FUEL), utr[1](i,j,k,URHO)));,
            utr[2](i,j,k,URHOY_FUEL) = amrex::max(Real(0.0), amrex::min(utr[2](i,j,k,URHOY_FUEL), utr[2](i,j,k,URHO)));
            );

            AMREX_D_TERM(
            utr[0](i,j,k,URHOY_OXID) = amrex::max(Real(0.0), amrex::min(utr[0](i,j,k,URHOY_OXID), utr[0](i,j,k,URHO)));,
            utr[1](i,j,k,URHOY_OXID) = amrex::max(Real(0.0), amrex::min(utr[1](i,j,k,URHOY_OXID), utr[1](i,j,k,URHO)));,
            utr[2](i,j,k,URHOY_OXID) = amrex::max(Real(0.0), amrex::min(utr[2](i,j,k,URHOY_OXID), utr[2](i,j,k,URHO)));
            );

          
            AMREX_D_TERM(
            utr[0](i,j,k,URHOY_PROD) = amrex::max(Real(0.0), amrex::min(utr[0](i,j,k,URHOY_PROD), utr[0](i,j,k,URHO)));,
            utr[1](i,j,k,URHOY_PROD) = amrex::max(Real(0.0), amrex::min(utr[1](i,j,k,URHOY_PROD), utr[1](i,j,k,URHO)));,
            utr[2](i,j,k,URHOY_PROD) = amrex::max(Real(0.0), amrex::min(utr[2](i,j,k,URHOY_PROD), utr[2](i,j,k,URHO)));
            );

            AMREX_D_TERM(
            udi[0](i,j,k,URHOY_FUEL) = amrex::max(Real(0.0), amrex::min(udi[0](i,j,k,URHOY_FUEL), udi[0](i,j,k,URHO)));,
            udi[1](i,j,k,URHOY_FUEL) = amrex::max(Real(0.0), amrex::min(udi[1](i,j,k,URHOY_FUEL), udi[1](i,j,k,URHO)));,
            udi[2](i,j,k,URHOY_FUEL) = amrex::max(Real(0.0), amrex::min(udi[2](i,j,k,URHOY_FUEL), udi[2](i,j,k,URHO)));
            );

	    AMREX_D_TERM(
            udi[0](i,j,k,URHOY_OXID) = amrex::max(Real(0.0), amrex::min(udi[0](i,j,k,URHOY_OXID), udi[0](i,j,k,URHO)));,
            udi[1](i,j,k,URHOY_OXID) = amrex::max(Real(0.0), amrex::min(udi[1](i,j,k,URHOY_OXID), udi[1](i,j,k,URHO)));,
            udi[2](i,j,k,URHOY_OXID) = amrex::max(Real(0.0), amrex::min(udi[2](i,j,k,URHOY_OXID), udi[2](i,j,k,URHO)));
            );

            AMREX_D_TERM(
            udi[0](i,j,k,URHOY_PROD) = amrex::max(Real(0.0), amrex::min(udi[0](i,j,k,URHOY_PROD), udi[0](i,j,k,URHO)));,
            udi[1](i,j,k,URHOY_PROD) = amrex::max(Real(0.0), amrex::min(udi[1](i,j,k,URHOY_PROD), udi[1](i,j,k,URHO)));,
            udi[2](i,j,k,URHOY_PROD) = amrex::max(Real(0.0), amrex::min(udi[2](i,j,k,URHOY_PROD), udi[2](i,j,k,URHO)));
            );

 
        });
    }

    // -----------------Compute the anti-diffusive fluxes-------------------
    Real mudiff = 0.0;
#if AMREX_SPACEDIM==3
    mudiff = 1.0/12.0;
#endif

    Gpu::streamSynchronize();
    Gpu::synchronize();
    // x-direction
    const Box& bxxng2 = amrex::grow(amrex::surroundingNodes(bx,0),NUM_GROW-2);
    amrex::ParallelFor(bxxng2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_eb_x(i, j, k, q, fldx[0], sofab, utr[0], dxinv[0], dt, diff, diff_cc, mudiff, flag, 
            vfrac, apx, vfmin);   });

    // y-direction
    const Box& byyng2 = amrex::grow(amrex::surroundingNodes(bx,1),NUM_GROW-2);
    amrex::ParallelFor(byyng2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_eb_y(i, j, k, q, fldx[1], sofab, utr[1], dxinv[1], dt, diff, diff_cc, mudiff, flag, 
            vfrac, apy, vfmin);   });

#if AMREX_SPACEDIM==3
    // z-direction
    const Box& bzzng2 = amrex::grow(amrex::surroundingNodes(bx,2),NUM_GROW-2);
    amrex::ParallelFor(bzzng2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {   fct_ad_flux_eb_z(i, j, k, q, fldx[2], sofab, utr[2], dxinv[2], dt, diff, diff_cc, mudiff, flag, 
            vfrac, apz, vfmin);   });
#endif

    Gpu::streamSynchronize();
    Gpu::synchronize();

    // // Prelimit the anti-diffusive fluxes
    const Box& bxnd1 = amrex::grow(amrex::surroundingNodes(bx,0),3);
        amrex::ParallelFor(bxnd1, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_eb_x(i, j, k, n, fldx[0], udi[0], flag, vfrac, vfmin); });

    const Box& bynd1 = amrex::grow(amrex::surroundingNodes(bx,1),3);
        amrex::ParallelFor(bynd1, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_eb_y(i, j, k, n, fldx[1], udi[1], flag, vfrac, vfmin); });

#if AMREX_SPACEDIM==3
    const Box& bznd1 = amrex::grow(amrex::surroundingNodes(bx,2),3);
        amrex::ParallelFor(bznd1, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_prelimit_ad_flux_eb_z(i, j, k, n, fldx[2], udi[2], flag, vfrac, vfmin); });
#endif

    Gpu::streamSynchronize();
    Gpu::synchronize();
    
    // // Calculate the total incoming and outgoing antidiffusive fluxes in each cell
    fracin.resize(amrex::grow(bx,3),NEQNS);
    fracin.setVal<RunOn::Device>(1.0);
    auto const& finfab = fracin.array(); 

    fracou.resize(amrex::grow(bx,3),NEQNS);
    fracou.setVal<RunOn::Device>(1.0);
    auto const& foufab = fracou.array();

    amrex::ParallelFor(bxg3, NEQNS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {   fct_compute_frac_eb_fluxes(i, j, k, n, AMREX_D_DECL(fldx[0], fldx[1], fldx[2]), 
                                        finfab, foufab, udfab, flag, vfrac, vfmin);  });  

    Gpu::streamSynchronize();
    Gpu::synchronize();
    // // Compute the corrected fluxes (2 ghost cells)
    const Box& bxnd = amrex::grow(amrex::surroundingNodes(bx,0),2);
    amrex::ParallelFor(bxnd, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {   fct_correct_fluxes_eb_x(i, j, k, n, fldx[0], finfab, foufab, apx); });

    const Box& bynd = amrex::grow(amrex::surroundingNodes(bx,1),2);
    amrex::ParallelFor(bynd, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {   fct_correct_fluxes_eb_y(i, j, k, n, fldx[1], finfab, foufab, apy); });

#if AMREX_SPACEDIM==3
    const Box& bznd = amrex::grow(amrex::surroundingNodes(bx,2),2);
    amrex::ParallelFor(bznd, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {   fct_correct_fluxes_eb_z(i, j, k, n, fldx[2], finfab, foufab, apz); });
#endif

    Gpu::streamSynchronize();
    Gpu::synchronize();

    // Compute the divergence of the corrected fluxes
    amrex::ParallelFor(bxg2, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
       // This does the divergence but not the redistribution -- we will do that later
       // We do compute the weights here though
       fct_eb_compute_div_corrected(i,j,k,n,blo,bhi,q,sofab, divc_arr,
                AMREX_D_DECL(fldx[0] ,fldx[1] ,fldx[2]),
                AMREX_D_DECL(fx_out_arr,fy_out_arr,fz_out_arr),
                flag, vfrac, bcent, coefs, redistwgt_arr,
                AMREX_D_DECL(apx, apy, apz),
                AMREX_D_DECL(fcx, fcy, fcz), dxinv, dx, dtinv, *lparm, l_eb_weights_type, l_do_visc, rk, vfmin);
    });

    Gpu::streamSynchronize();
    Gpu::synchronize();
    // Carry out the redistribution of the fluxes
    if(rk == 1){
        optmp.setVal<RunOn::Device>(0.0);
        delta_m.setVal<RunOn::Device>(0.0);
        fct_flux_redistribute_corrected(bx,divc_arr,dsdtfab,optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 0, vfmin);
    }else{
        optmp.setVal<RunOn::Device>(0.0);
        delta_m.setVal<RunOn::Device>(0.0);
        fct_flux_redistribute_corrected(bx,divc_arr,dsdtfab,optmp_arr,del_m_arr,redistwgt_arr,vfrac,flag,
                          as_crse, drho_as_crse, rrflag_as_crse, as_fine, dm_as_fine, lev_mask, dt, 1, vfmin);
    }
    
    optmp.setVal<RunOn::Device>(0.0);
    delta_m.setVal<RunOn::Device>(0.0);

    Gpu::streamSynchronize();
    Gpu::synchronize();
}
