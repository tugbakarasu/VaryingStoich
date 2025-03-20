
#include <CNS.H>
#include <CNS_K.H>
#include <CNS_tagging.H>
#include <CNS_parm.H>
#include <cns_prob.H>

#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_EBAmrUtil.H>
#include <AMReX_EBFArrayBox.H>

#include <climits>

using namespace amrex;

BCRec     CNS::phys_bc;

int       CNS::verbose = 0;
int       CNS::NUM_GROW = 8;
IntVect   CNS::hydro_tile_size {AMREX_D_DECL(1024,16,16)};
Real      CNS::cfl       = 0.3;
int       CNS::do_reflux = 1;
int       CNS::refine_cutcells          = 1;
int       CNS::refine_max_dengrad_lev   = -1;
Real      CNS::refine_dengrad           = 1.0e10;
Vector<RealBox> CNS::refine_boxes;
RealBox*  CNS::dp_refine_boxes;

bool      CNS::do_visc        = false;  // diffusion is off by default
bool      CNS::use_const_visc = false; // diffusion does not use constant viscosity by default

int       CNS::plm_iorder     = 2;     // [1,2] 1: slopes are zero'd,  2: second order slopes
Real      CNS::plm_theta      = 2.0;   // [1,2] 1: minmod; 2: van Leer's MC
Real      CNS::gravity        = 0.0;

int       CNS::eb_weights_type = 0;   // [0,1,2,3] 0: weights are all 1
int       CNS::do_reredistribution = 1;
int       CNS::eb_algorithm = 1; 

int       CNS::tag_probspecific = 0;
int       CNS::which_comp = -1;

// FCT variables
Real      CNS::diff1                    = 0.99999;
Real      CNS::diffcc                   = 0.99999;
Real      CNS::vfc_threshold            = 0.0;

bool      CNS::do_react        = false;  // reaction is off by default

Real Parm::pre_exp_tmp = 0.0;
Real Parm::Ea_nd_tmp = 0.0;
Real Parm::q_nd_tmp = 0.0;
Real Parm::kappa_0_tmp = 0.0;

CNS::CNS ()
{}

CNS::CNS (Amr&            papa,
          int             lev,
          const Geometry& level_geom,
          const BoxArray& bl,
          const DistributionMapping& dm,
          Real            time)
    : AmrLevel(papa,lev,level_geom,bl,dm,time)
{
    if (do_reflux && level > 0) {
        flux_reg.define(bl, papa.boxArray(level-1),
                        dm, papa.DistributionMap(level-1),
                        level_geom, papa.Geom(level-1),
                        papa.refRatio(level-1), level, NEQNS);
    }

    buildMetrics();
}

CNS::~CNS ()
{}

void
CNS::init (AmrLevel& old)
{
    auto& oldlev = dynamic_cast<CNS&>(old);

    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev.state[State_Type].curTime();
    Real prev_time = oldlev.state[State_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    MultiFab& S_new = get_new_data(State_Type);
    FillPatch(old,S_new,0,cur_time,State_Type,0,NUM_STATE);

    MultiFab& C_new = get_new_data(Cost_Type);
    FillPatch(old,C_new,0,cur_time,Cost_Type,0,1);

}

void
CNS::init ()
{
    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time)/static_cast<Real>(parent->MaxRefRatio(level-1));
    setTimeLevel(cur_time,dt_old,dt);

    MultiFab& S_new = get_new_data(State_Type);
    FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NUM_STATE);

    MultiFab& C_new = get_new_data(Cost_Type);
    FillCoarsePatch(C_new, 0, cur_time, Cost_Type, 0, 1);
}

void
CNS::initData ()
{
    BL_PROFILE("CNS::initData()");

    const auto geomdata = geom.data();
    MultiFab& S_new = get_new_data(State_Type);
    S_new.setVal(0.0);

    ProbParm* hlpparm = h_prob_parm;
    ProbParm* dlpparm = d_prob_parm;

    // The function init_probparams must be defined 
    // (atleast as empty function) in cns_prob_parm.H
    // This function is used to initialized problem
    // parameters that are derived from the input parameters

    Parm const* lparm = d_parm;
    init_probparams(geomdata, *lparm, *hlpparm, *dlpparm);

#ifdef AMREX_USE_GPU
        // Cannot use Gpu::copy because ProbParm is not trivailly copyable.
        Gpu::htod_memcpy_async(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#endif

    ProbParm const* lprobparm = d_prob_parm;
    auto const& sma = S_new.arrays();
    amrex::ParallelFor(S_new,
    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        cns_initdata(i, j, k, sma[box_no], geomdata, *lparm, *lprobparm);
    });

    // Compute the initial temperature (will override what was set in initdata)
    computeTemp(S_new, 0, 0, 0.0);
    MultiFab& C_new = get_new_data(Cost_Type);
    C_new.setVal(1.0);
    cns_probspecific_func(S_new, geomdata, 0, *lparm, hlpparm, dlpparm, 0.0, 0.0, level);
}

void
CNS::computeInitialDt (int                    finest_level,
                       int                    /*sub_cycle*/,
                       Vector<int>&           n_cycle,
                       const Vector<IntVect>& /*ref_ratio*/,
                       Vector<Real>&          dt_level,
                       Real                   stop_time)
{
  //
  // Grids have been constructed, compute dt for all levels.
  //
  if (level > 0) {
    return;
  }

  Real dt_0 = std::numeric_limits<Real>::max();
  int n_factor = 1;
  for (int i = 0; i <= finest_level; i++)
  {
    dt_level[i] = getLevel(i).initialTimeStep();
    n_factor   *= n_cycle[i];
    dt_0 = std::min(dt_0,n_factor*dt_level[i]);
  }

  //
  // Limit dt's by the value of stop_time.
  //
  const Real eps = 0.001*dt_0;
  Real cur_time  = state[State_Type].curTime();
  if (stop_time >= 0.0) {
    if ((cur_time + dt_0) > (stop_time - eps))
      dt_0 = stop_time - cur_time;
  }

  n_factor = 1;
  for (int i = 0; i <= finest_level; i++)
  {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0/n_factor;
  }
}

void
CNS::computeNewDt (int                    finest_level,
                   int                    /*sub_cycle*/,
                   Vector<int>&           n_cycle,
                   const Vector<IntVect>& /*ref_ratio*/,
                   Vector<Real>&          dt_min,
                   Vector<Real>&          dt_level,
                   Real                   stop_time,
                   int                    post_regrid_flag)
{
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0) {
        return;
    }

    for (int i = 0; i <= finest_level; i++)
    {
        dt_min[i] = getLevel(i).estTimeStep();
    }

    if (post_regrid_flag == 1)
    {
        //
        // Limit dt's by pre-regrid dt
        //
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i],dt_level[i]);
        }
    }
    else
    {
        //
        // Limit dt's by change_max * old dt
        //
        static Real change_max = 1.1;
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
        }
    }

    //
    // Find the minimum over all levels
    //
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) > (stop_time - eps)) {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
CNS::post_regrid (int /*lbase*/, int /*new_finest*/)
{
}

void
CNS::post_timestep (int iteration)
{
    if (do_reflux && level < parent->finestLevel()) {
        CNS& fine_level = getLevel(level+1);
        MultiFab& S_crse = get_new_data(State_Type);
        MultiFab& S_fine = fine_level.get_new_data(State_Type);

        // Create temporary aliases for S_crse and S_fine as we do not want 
        // to reflux on sootfoil component
        MultiFab fine_tmp(S_fine,amrex::make_alias,URHO,NEQNS);
        MultiFab crse_tmp(S_crse,amrex::make_alias,URHO,NEQNS);

        if(S_crse.min(URHO, 0) < 0.0 || S_crse.min(UEDEN) < 0.0)
            Print() << "before reflux, min = " << S_crse.min(URHO, 0) << S_crse.min(UEDEN) << "\n";

        // fine_level.flux_reg.Reflux(S_crse, *volfrac, S_fine, *fine_level.volfrac);
        fine_level.flux_reg.Reflux(crse_tmp, *volfrac, fine_tmp, *fine_level.volfrac);

        if(S_crse.min(URHO, 0) < 0.0 || S_crse.min(UEDEN) < 0.0)
            Print() << "after reflux, min = " << S_crse.min(URHO, 0) << S_crse.min(UEDEN) << "\n";

        Real cur_time  = state[State_Type].curTime();
        computeTemp(S_crse, 0, 0, cur_time);
    }

    if (level < parent->finestLevel()) {
        avgDown();
    }

    ProbParm* dlpparm = d_prob_parm;
    ProbParm* hlpparm = h_prob_parm;
    Parm* lparm = d_parm;
    Real cur_time  = state[State_Type].curTime();
    Real dt        = parent->dtLevel(level);
    MultiFab& S_new = get_new_data(State_Type);
    const auto geomdata = geom.data();
    cns_probspecific_func(S_new, geomdata, 0, *lparm, hlpparm, dlpparm, cur_time, dt, level);
}

void
CNS::postCoarseTimeStep (Real time)
{
    // Call user-specified post-processing function only on level 0
    if(level == 0){
        Real dt        = parent->dtLevel(level);
        MultiFab& S = get_new_data(State_Type);
        ProbParm* dlpparm = d_prob_parm;
        ProbParm* hlpparm = h_prob_parm;
        Parm* lparm = d_parm;
        const auto geomdata = geom.data();
        cns_probspecific_func(S, geomdata, 1, *lparm, hlpparm, dlpparm, time, dt, level);
    }
    // This only computes sum on level 0
    if (verbose >= 2) {
        printTotal();
    }
}

void
CNS::printTotal () const
{
    const MultiFab& S_new = get_new_data(State_Type);
    MultiFab mf(grids, dmap, 1, 0);
    std::array<Real,5> tot;
    for (int comp = 0; comp < 5; ++comp) {
        MultiFab::Copy(mf, S_new, comp, 0, 1, 0);
        MultiFab::Multiply(mf, *volfrac, 0, 0, 1, 0);
        tot[comp] = mf.sum(0,true) * geom.ProbSize();
    }
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
            ParallelDescriptor::ReduceRealSum(tot.data(), 5, ParallelDescriptor::IOProcessorNumber());
            amrex::Print().SetPrecision(17) << "\n[CNS] Total mass       is " << tot[0] << "\n"
                                            <<   "      Total x-momentum is " << tot[1] << "\n"
                                            <<   "      Total y-momentum is " << tot[2] << "\n"
#if (AMREX_SPACEDIM == 3)
                                            <<   "      Total z-momentum is " << tot[3] << "\n"
#endif
                                            <<   "      Total energy     is " << tot[4] << "\n";
#ifdef BL_LAZY
        });
#endif
}

void
CNS::post_init (Real)
{
    if (level > 0) return;

    for (int k = parent->finestLevel()-1; k >= 0; --k) {
        getLevel(k).avgDown();
    }

    const auto geomdata = geom.data();
    ProbParm* dlpparm = d_prob_parm;
    ProbParm* hlpparm = h_prob_parm;
    Parm* lparm = d_parm;
    MultiFab& S_new = get_new_data(State_Type);
    Print() << "level = " << level << "\n";
    computeTemp(S_new, 0, 0, 0.0);
    cns_probspecific_func(S_new, geomdata, 1, *lparm, hlpparm, dlpparm, 0.0, 0.0, level);

    if (verbose >= 2) {
        printTotal();
    }
}

void
CNS::post_restart ()
{
}

void
CNS::errorEst (TagBoxArray& tags, int, int, Real /*time*/, int, int)
{
    BL_PROFILE("CNS::errorEst()");

    if (refine_cutcells && level < refine_max_dengrad_lev) {
        const MultiFab& S_new = get_new_data(State_Type);
        amrex::TagCutCells(tags, S_new);
    }

    if (!refine_boxes.empty() && level < refine_max_dengrad_lev)
    {
        const int n_refine_boxes = refine_boxes.size();
        const auto problo = geom.ProbLoArray();
        const auto dx = geom.CellSizeArray();
        auto boxes = dp_refine_boxes;

        auto const& tagma = tags.arrays();
        ParallelFor(tags,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            RealVect pos {AMREX_D_DECL((i+0.5)*dx[0]+problo[0],
                                       (j+0.5)*dx[1]+problo[1],
                                       (k+0.5)*dx[2]+problo[2])};
            for (int irb = 0; irb < n_refine_boxes; ++irb) {
                if (boxes[irb].contains(pos)) {
                    tagma[box_no](i,j,k) = TagBox::SET;
                }
            }
        });
        Gpu::synchronize();
    }

    const MultiFab& S_new = get_new_data(State_Type);

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S_new.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    // if (level < refine_max_dengrad_lev)
    // {
        // const int n_refine_boxes = refine_boxes.size();
        // const auto problo = geom.ProbLoArray();
        // const auto dx = geom.CellSizeArray();
        // auto boxes = dp_refine_boxes;

        const Real cur_time = state[State_Type].curTime();

        if(tag_probspecific == 0){
            // This tags cells based on if the gradient of quantity wc
            // at a given cell is greater than a threshold dengrad_threshold
            // The threshold value MUST be given in the inputs file
            const MultiFab& S_new = get_new_data(State_Type);
            MultiFab rho(S_new.boxArray(), S_new.DistributionMap(), 1, 1);
            FillPatch(*this, rho, rho.nGrow(), cur_time, State_Type, which_comp, 1, 0);

            const char   tagval = TagBox::SET;
//        const char clearval = TagBox::CLEAR;
            const Real dengrad_threshold = refine_dengrad;

            auto const& tagma = tags.arrays();
            auto const& rhoma = rho.const_arrays();
            auto const& flag = flags.const_arrays();
            ParallelFor(rho,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
            {
                cns_tag_denerror(i, j, k, tagma[box_no], rhoma[box_no], dengrad_threshold, tagval, flag[box_no]);
            });

            Gpu::synchronize();
        }else{
            const auto geomdata = geom.data();
            const MultiFab& S_new = get_new_data(State_Type);
            MultiFab Stag(S_new.boxArray(),S_new.DistributionMap(),NUM_STATE-NAUX,2);
            FillPatch(*this, Stag, Stag.nGrow(), cur_time, State_Type, URHO, Stag.nComp(), 0);
            // Print() << "level = " << ", max(Temp) = " << Stag.max(UTEMP, Stag.nGrow()) << "\n";
            
            ProbParm const* lprobparm = CNS::d_prob_parm;
            Parm const* lparm = CNS::d_parm;
            int lev = level;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for(MFIter mfi(Stag,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();

                auto const& flag = flags.const_array(mfi);

                Array4<Real      > const&    sarr =    Stag.array(mfi);

                const char   tagval = TagBox::SET;
                auto tag = tags.array(mfi);

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    cns_tag_probspecific(i, j, k, tag, sarr, geomdata, tagval, flag,
                        *lparm, *lprobparm, cur_time, lev);
                });

            }
            Gpu::synchronize();
        }
    // }
}

void
CNS::read_params ()
{
    ParmParse pp("cns");

    pp.query("v", verbose);

    Vector<int> tilesize(AMREX_SPACEDIM);
    if (pp.queryarr("hydro_tile_size", tilesize, 0, AMREX_SPACEDIM))
    {
        for (int i=0; i<AMREX_SPACEDIM; i++) hydro_tile_size[i] = tilesize[i];
    }

    pp.query("cfl", cfl);
    pp.query("ngrow", NUM_GROW);

    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
    pp.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        phys_bc.setLo(i,lo_bc[i]);
        phys_bc.setHi(i,hi_bc[i]);
    }

    pp.query("do_reflux", do_reflux);

    pp.query("do_visc", do_visc);
    h_parm->is_visc = do_visc;

    // FLAG TO DETERMINE IF THE EB HAS WALL LOSSES
    pp.query("eb_wallloss", h_parm->eb_wallloss);
    if(h_parm->eb_wallloss){
        // THERMAL CONDUCTIVITY OF SOLID WALL
        pp.get("ksolid", h_parm->ksolid);
        // TEMPERATURE OF THE SOLID WALL (THIS IS THE CORE SOLID TEMP. WHICH IS FIXED)
        pp.get("tempsolidwall", h_parm->tempsolidwall);
    }

    if (do_visc)
    {
        pp.query("use_const_visc",use_const_visc);
        h_parm->is_const_visc = use_const_visc;
        if (use_const_visc)
        {
            pp.get("const_visc_mu",h_parm->const_visc_mu);
            pp.query("const_visc_ki",h_parm->const_visc_ki);
            pp.query("const_lambda" ,h_parm->const_lambda);
        }
        pp.query("kappa_0", h_parm->kappa_0);
        h_parm->kappa_0 = h_parm->kappa_0 * 1.e-1;
    } else {
       use_const_visc = true;
       h_parm->const_visc_mu = 0.0;
       h_parm->const_visc_ki = 0.0;
       h_parm->const_lambda  = 0.0;
    }

    pp.query("refine_cutcells", refine_cutcells);

    pp.query("refine_max_dengrad_lev", refine_max_dengrad_lev);
    pp.query("refine_dengrad", refine_dengrad);

    int irefbox = 0;
    Vector<Real> refboxlo, refboxhi;
    while (pp.queryarr(("refine_box_lo_"+std::to_string(irefbox)).c_str(), refboxlo))
    {
        pp.getarr(("refine_box_hi_"+std::to_string(irefbox)).c_str(), refboxhi);
        refine_boxes.emplace_back(refboxlo.data(), refboxhi.data());
        ++irefbox;
    }
    if (!refine_boxes.empty()) {
#ifdef AMREX_USE_GPU
        dp_refine_boxes = (RealBox*)The_Arena()->alloc(sizeof(RealBox)*refine_boxes.size());
        Gpu::htod_memcpy(dp_refine_boxes, refine_boxes.data(), sizeof(RealBox)*refine_boxes.size());
#else
        dp_refine_boxes = refine_boxes.data();
#endif
    }

    pp.query("gravity", gravity);

    pp.query("eos_gamma", h_parm->eos_gamma);
    // eos_mu is the molecular weight of gas (must be in SI units)
    pp.query("eos_mu"   , h_parm->eos_mu);
    pp.query("Pr"       , h_parm->Pr);
    pp.query("C_S"      , h_parm->C_S);
    pp.query("T_S"      , h_parm->T_S);

    // FCT parameters
    pp.query("minro", h_parm->minro);
    pp.query("minp", h_parm->minp);
    pp.query("maxro", h_parm->maxro);
    pp.query("maxp", h_parm->maxp);
    pp.query("mindt", h_parm->mindt);
    pp.query("diff1", diff1);
    pp.query("diffcc", diffcc);
    pp.query("vfc_threshold", vfc_threshold);
    pp.query("start_sfoil_time", h_parm->start_sfoil_time);

    // NSCBC parameters
    Vector<amrex::Real> nscbc_eta(5), nscbc_lo(3), nscbc_hi(3);
    ParmParse ppns("nscbc");
    ppns.query("sigma", h_parm->sigma);
    ppns.query("beta", h_parm->beta);
    if (ppns.queryarr("eta", nscbc_eta, 0, 5))
    {
        for (int i=0; i<5; i++) h_parm->eta[i] = nscbc_eta[i];
    }

    if (ppns.queryarr("lo", nscbc_lo, 0, AMREX_SPACEDIM))
    {
        for (int i=0; i<AMREX_SPACEDIM; i++) h_parm->do_nscbc_lo[i] = nscbc_lo[i];
    }

    if (ppns.queryarr("hi", nscbc_hi, 0, AMREX_SPACEDIM))
    {
        for (int i=0; i<AMREX_SPACEDIM; i++) h_parm->do_nscbc_hi[i] = nscbc_hi[i];
    }

    // Target values for NSCBC
    ppns.query("ptarg", h_parm->ptarg);
    ppns.query("utarg", h_parm->utarg);
    ppns.query("vtarg", h_parm->vtarg);
#if AMREX_SPACEDIM==3
    ppns.query("wtarg", h_parm->wtarg);
#endif
    ppns.query("Ttarg", h_parm->Ttarg);
    if(do_react == 1) ppns.query("Ytarg", h_parm->Ytarg);

    // Tagging parameters
    pp.query("tag_probspecific", tag_probspecific);

    // Choose algorithm for cut cell region
    pp.query("eb_algorithm", eb_algorithm);
    if(eb_algorithm == 2){
        pp.query("plm_iorder", plm_iorder);
    }

    pp.query("eos_gamma", h_parm->eos_gamma);
    // eos_mu is the molecular weight of gas (in SI units kg/mol)
    pp.query("eos_mu"   , h_parm->eos_mu);

    pp.query("do_react", do_react);
    if(do_react){
        pp.get("eos_gamma", h_parm->eos_gamma);
        pp.get("eos_mu", h_parm->eos_mu);

        // 
        //std::cout << "Before Calculate_CDM_Parameters - q_nd: " << h_parm->q_nd << std::endl;

        Parm::Calculate_CDM_Parameters(1.0, h_parm->pre_exp, h_parm->Ea_nd, h_parm->q_nd, h_parm->kappa_0);

        //std::cout << "After Calculate_CDM_Parameters - q_nd: " << h_parm->q_nd << std::endl;

        pp.query("Ea_nd", h_parm->Ea_nd);
        pp.query("Tref", h_parm->Tref);
        pp.query("pref", h_parm->pref);

    // pre_exp and kappa_0 read in CGS units
        pp.query("pre_exp", h_parm->pre_exp);
        pp.query("kappa_0", h_parm->kappa_0);

    // Convert Arrhenius pre-exponential and thermal conductivity to SI units
        h_parm->pre_exp = h_parm->pre_exp * 1.e-3;
        h_parm->kappa_0 = h_parm->kappa_0 * 1.e-1;
    }

    h_parm->Initialize();

    // Calculate the dimensional heat release and activation energy (in SI units)
    h_parm->q_dim  = h_parm->q_nd  * h_parm->Rsp * h_parm->Tref;
    //std::cout << "h_parm->q_nd after update: " << h_parm->q_nd << std::endl;
    //std::cout << "h_parm->q_dim after update: " << h_parm->q_dim << std::endl;
    h_parm->Ea_dim = h_parm->Ea_nd * h_parm->Ru  * h_parm->Tref;

    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_parm, h_parm+1, d_parm);

    // eb_weights_type:
    //   0 -- weights = 1
    //   1 -- use_total_energy_as_eb_weights-
    //   2 -- use_mass_as_eb_weights
    //   3 -- use_volfrac_as_eb_weights
    pp.query("eb_weights_type", eb_weights_type);
    if (eb_weights_type < 0 || eb_weights_type > 3)
        amrex::Abort("CNS: eb_weights_type must be 0,1,2 or 3");

    pp.query("do_reredistribution", do_reredistribution);
    if (do_reredistribution != 0 && do_reredistribution != 1)
        amrex::Abort("CNS: do_reredistibution must be 0 or 1");
}

void
CNS::avgDown ()
{
    BL_PROFILE("CNS::avgDown()");

    if (level == parent->finestLevel()) return;

    auto& fine_lev = getLevel(level+1);

    MultiFab& S_crse =          get_new_data(State_Type);
    MultiFab& S_fine = fine_lev.get_new_data(State_Type);

    MultiFab volume(S_fine.boxArray(), S_fine.DistributionMap(), 1, 0);
    volume.setVal(1.0);
    amrex::EB_average_down(S_fine, S_crse, volume, fine_lev.volFrac(),
                           0, NEQNS, fine_ratio);

    const int nghost = 0;
    Real cur_time  = state[State_Type].curTime();
    computeTemp (S_crse, nghost, 0, cur_time);
}

void
CNS::buildMetrics ()
{
    BL_PROFILE("CNS::buildMetrics()");

    // make sure dx == dy == dz
    const Real* dx = geom.CellSize();
#if (AMREX_SPACEDIM == 2)
    if (std::abs(dx[0]-dx[1]) > 1.e-12*dx[0])
        amrex::Abort("CNS: must have dx == dy\n");
#else
    if (std::abs(dx[0]-dx[1]) > 1.e-12*dx[0] || std::abs(dx[0]-dx[2]) > 1.e-12*dx[0])
        amrex::Abort("CNS: must have dx == dy == dz\n");
#endif

    const auto& ebfactory = dynamic_cast<EBFArrayBoxFactory const&>(Factory());

    volfrac = &(ebfactory.getVolFrac());
    bndrycent = &(ebfactory.getBndryCent());
    areafrac = ebfactory.getAreaFrac();
    facecent = ebfactory.getFaceCent();

    Parm const* l_parm = d_parm;

    level_mask.clear();
    level_mask.define(grids,dmap,1,3);
    level_mask.BuildMask(geom.Domain(), geom.periodicity(),
                         l_parm->level_mask_covered,
                         l_parm->level_mask_notcovered,
                         l_parm->level_mask_physbnd,
                         l_parm->level_mask_interior);

    ProbParm* lpparm  = h_prob_parm;
    ProbParm* dlpparm = d_prob_parm;

    // The function init_probparams must be defined 
    // (atleast as empty function) in cns_prob_parm.H
    // This function is used to initialized problem
    // parameters that are derived from the input parameters
    const auto geomdata = geom.data();
    Parm const* lparm = d_parm;
    init_probparams(geomdata, *lparm, *lpparm, *dlpparm);
#ifdef AMREX_USE_GPU
        // Cannot use Gpu::copy because ProbParm is not trivailly copyable.
        Gpu::htod_memcpy_async(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#endif
}

Real
CNS::estTimeStep ()
{
    BL_PROFILE("CNS::estTimeStep()");

    const auto dx = geom.CellSizeArray();
    const MultiFab& S = get_new_data(State_Type);
    const Real cur_time = state[State_Type].curTime();
    MultiFab Stag(S.boxArray(),S.DistributionMap(),NUM_STATE,1);
    FillPatch(*this, Stag, Stag.nGrow(), cur_time, State_Type, URHO, Stag.nComp(), 0);
    
    Parm const* lparm = d_parm;

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    Real estdt = std::numeric_limits<Real>::max();

    // Reduce min operation
    ReduceOps<ReduceOpMin> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S,false); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const Box& bxg = amrex::grow(bx,S.nGrow());
        const auto& flag = flags[mfi];
        auto const& s_arr = S.array(mfi);
        if (flag.getType(bx) != FabType::covered)
        {
          reduce_op.eval(bx, reduce_data, [=]
             AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
             {
                 return cns_estdt(i,j,k,s_arr,dx,*lparm);
             });
        }
    } // mfi

    ReduceTuple host_tuple = reduce_data.value();
    estdt = amrex::min(estdt,amrex::get<0>(host_tuple));

    estdt *= cfl;
    ParallelDescriptor::ReduceRealMin(estdt);

    if(estdt < h_parm->mindt)
        amrex::Abort("timestep too small, less than mindt, aborting... reduce CFL");

    return estdt;
}

Real
CNS::initialTimeStep ()
{
    return estTimeStep();
}

void
CNS::computeTemp (MultiFab& State, int ng, int do_sootfoil, Real cur_time)
{
    BL_PROFILE("CNS::computeTemp()");

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(State.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    Parm const* lparm = d_parm;

    // This will reset Eint and compute Temperature
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(State,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);

        const auto& flag = flags[mfi];
        auto s_arr = State.array(mfi);

        if (flag.getType(bx) != FabType::covered) {
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                cns_compute_temperature(i, j, k, s_arr, *lparm, do_sootfoil, cur_time);
            });
        }
    }
}
