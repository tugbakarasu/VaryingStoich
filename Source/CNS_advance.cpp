
#include <CNS.H>

#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>

using namespace amrex;

Real
CNS::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    BL_PROFILE("CNS::advance()");

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);
    MultiFab dSdt(grids,dmap,NEQNS,0,MFInfo(),Factory());
    MultiFab Sborder(grids,dmap,NUM_STATE-NAUX,NUM_GROW,MFInfo(),Factory());

    // FCT code
    MultiFab Soldtmp(grids,dmap,NUM_STATE-NAUX,NUM_GROW,MFInfo(),Factory());

    MultiFab& C_new = get_new_data(Cost_Type);
    C_new.setVal(0.0);

    EBFluxRegister* fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        CNS& fine_level = getLevel(level+1);
        fr_as_crse = &fine_level.flux_reg;
    }

    EBFluxRegister* fr_as_fine = nullptr;
    if (do_reflux && level > 0) {
        fr_as_fine = &flux_reg;
    }

    if (fr_as_crse) {
        fr_as_crse->reset();
    }

    dSdt.setVal(0.);

    int numcomp = NUM_STATE-NAUX;
    int conscomp = NEQNS;

    // ---------FCT RK2 integrator------------
    // RK2 stage 1
    FillPatch(*this,Sborder,NUM_GROW,time,State_Type,0,NUM_STATE-NAUX);
    MultiFab::Copy(Soldtmp,Sborder,0,0,numcomp,NUM_GROW);

    MultiFab::Copy(S_new,S_old,SFOIL,SFOIL,1,0);

    FillPatch(*this,S_new,NUM_GROW,time,State_Type,SFOIL,1);

    if(Sborder.min(URHO,NUM_GROW) < Real(0.0) 
        || Sborder.min(UEDEN,NUM_GROW) < Real(0.0) || Sborder.min(UEINT,NUM_GROW) < Real(0.0)){
        Print() << "CNS::advance, after FillPatch(), lev = " << level << ", contains NaN() " << "\n";
        Print() << "min(ueden) = " << Sborder.min(UEDEN,NUM_GROW) << Sborder.min(URHO,NUM_GROW) << "\n";
        amrex::Error("NaN value found before RK1 advance, aborting...");
    }

    // RK step 1
    compute_dSdt(Soldtmp, Sborder, dSdt, Real(0.5)*dt, fr_as_crse, fr_as_fine, 1);

    // U^* = U^{n+1/2} = U^n + 0.5*dt*(dSdt^n)
    MultiFab::LinComb(S_new, Real(1.0), Sborder, URHO, Real(0.5)*dt, dSdt, URHO, URHO, conscomp, 0);
    computeTemp(S_new, 0, 0, time);

    dSdt.setVal(0.);

    // RK stage 2
    // After fillpatch, SBorder = U^n + 0.5*dt*(dSdt^n)    
    FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, URHO, NUM_STATE-NAUX);
    compute_dSdt(Soldtmp, Sborder, dSdt, dt, fr_as_crse, fr_as_fine, 2);
    // U^{n+1} = U^n + (dt*(dSdt^*))
    MultiFab::LinComb(S_new, Real(1.0), S_old, URHO, dt, dSdt, URHO, URHO, conscomp, 0);

    computeTemp(S_new, 0, 1, time);
    FillPatch(*this, S_new, NUM_GROW, time+dt, State_Type, URHO, NUM_STATE-NAUX);

    return dt;
}

void
CNS::compute_dSdt (const MultiFab& Sold, MultiFab& S, MultiFab& dSdt, Real dt,
                   EBFluxRegister* fr_as_crse, EBFluxRegister* fr_as_fine, int rk)
{
    BL_PROFILE("CNS::compute_dSdt()");

    const Real* dx = geom.CellSize();
    const int ncomp = dSdt.nComp();

    int as_crse = (fr_as_crse != nullptr);
    int as_fine = (fr_as_fine != nullptr);

    MultiFab& cost = get_new_data(Cost_Type);

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    MFItInfo mfiinfo;
    if (Gpu::notInLaunchRegion()) {
        mfiinfo.EnableTiling(hydro_tile_size).SetDynamic(true);
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        std::array<FArrayBox,AMREX_SPACEDIM> flux;
        FArrayBox dm_as_fine(Box::TheUnitBox(),ncomp);
        FArrayBox fab_drho_as_crse(Box::TheUnitBox(),ncomp);
        IArrayBox fab_rrflag_as_crse(Box::TheUnitBox());

        for (MFIter mfi(S, mfiinfo); mfi.isValid(); ++mfi)
        {
            auto wt = amrex::second();

            const Box& bx = mfi.tilebox();

            const auto& flag = flags[mfi];

            if (flag.getType(bx) == FabType::covered) {
                dSdt[mfi].setVal<RunOn::Device>(0.0, bx, 0, ncomp);
            } else {

                // flux is used to store centroid flux needed for reflux
                for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                    flux[idim].resize(amrex::surroundingNodes(bx,idim),NEQNS);
                    flux[idim].setVal<RunOn::Device>(0.);
                }

                if (flag.getType(amrex::grow(bx,NUM_GROW)) == FabType::regular)
                {
                    Array4<Real const>    so_arr =    Sold.array(mfi);
                    Array4<Real      >    s_arr     =    S.array(mfi);
                    Array4<Real      > dsdt_arr = dSdt.array(mfi);

                    compute_dSdt_box_fct(bx,so_arr,s_arr,dsdt_arr,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},dt,rk);

                    if(rk == 2){
                        if (fr_as_crse) {
                            fr_as_crse->CrseAdd(mfi,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},dx,dt,RunOn::Device);
                        }

                        if (fr_as_fine) {
                            fr_as_fine->FineAdd(mfi,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},dx,dt,RunOn::Device);
                        }
                    }
                    
                }
                else
                {
                    FArrayBox* p_drho_as_crse = (fr_as_crse) ?
                        fr_as_crse->getCrseData(mfi) : &fab_drho_as_crse;
                    const IArrayBox* p_rrflag_as_crse = (fr_as_crse) ?
                        fr_as_crse->getCrseFlag(mfi) : &fab_rrflag_as_crse;

                    if (fr_as_fine) {
                        dm_as_fine.resize(amrex::grow(bx,1),ncomp);
                    }

                    Array4<Real const> const&    so_arr =    Sold.array(mfi);
                    Array4<Real      > const&    s_arr =    S.array(mfi);
                    Array4<Real      > const& dsdt_arr = dSdt.array(mfi);
                    Array4<Real const> vf_arr = (*volfrac).array(mfi);
                    Array4<Real const> bcent_arr = (*bndrycent).array(mfi);

                    AMREX_D_TERM(Array4<Real const> const& apx = areafrac[0]->const_array(mfi);,
                                 Array4<Real const> const& apy = areafrac[1]->const_array(mfi);,
                                 Array4<Real const> const& apz = areafrac[2]->const_array(mfi));
                    AMREX_D_TERM(Array4<Real const> const& fcx = facecent[0]->const_array(mfi);,
                                 Array4<Real const> const& fcy = facecent[1]->const_array(mfi);,
                                 Array4<Real const> const& fcz = facecent[2]->const_array(mfi));

                    // if(eb_algorithm == 1){
                        compute_dSdt_box_eb_fct(bx,so_arr,s_arr,dsdt_arr,
                                           {AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},
                                            flags.const_array(mfi), vf_arr,
                                            AMREX_D_DECL(apx,apy,apz),AMREX_D_DECL(fcx,fcy,fcz), bcent_arr,
                                            as_crse, p_drho_as_crse->array(), p_rrflag_as_crse->const_array(),
                                            as_fine, dm_as_fine.array(), level_mask.const_array(mfi), dt, rk);
                    // }else if(eb_algorithm == 2){
                    //     compute_dSdt_box_eb(bx,s_arr,dsdt_arr,
                    //                    {AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},
                    //                     flags.const_array(mfi), vf_arr,
                    //                     AMREX_D_DECL(apx,apy,apz),AMREX_D_DECL(fcx,fcy,fcz), bcent_arr,
                    //                     as_crse, p_drho_as_crse->array(), p_rrflag_as_crse->const_array(),
                    //                     as_fine, dm_as_fine.array(), level_mask.const_array(mfi), dt, rk);
                    // }
                    
                    if(rk == 2){
                        if (fr_as_crse) {
                            fr_as_crse->CrseAdd(mfi,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},
                                            dx,dt,(*volfrac)[mfi],
                                            {AMREX_D_DECL(&((*areafrac[0])[mfi]),
                                                          &((*areafrac[1])[mfi]),
                                                          &((*areafrac[2])[mfi]))},
                                            RunOn::Device);
                        }

                        if (fr_as_fine) {
                            fr_as_fine->FineAdd(mfi,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},
                                            dx,dt,(*volfrac)[mfi],
                                            {AMREX_D_DECL(&((*areafrac[0])[mfi]),
                                                          &((*areafrac[1])[mfi]),
                                                          &((*areafrac[2])[mfi]))},
                                            dm_as_fine,
                                            RunOn::Device);
                        }
                    }
                    
                }
            }

            Gpu::streamSynchronize();
            Gpu::synchronize();

            wt = (amrex::second() - wt) / bx.d_numPts();
            cost[mfi].plus<RunOn::Device>(wt, bx);
        }
    }
}
