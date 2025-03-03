#ifndef CNS_PROB_H_
#define CNS_PROB_H_

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

#define PI 3.141592653589793238
#define PREATM 101325

using namespace amrex;


AMREX_GPU_DEVICE
inline
void
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    Real ly = prob_hi[1] - prob_lo[1];

    Real a0 = std::sqrt(parm.eos_gamma * parm.Rsp * prob_parm.T0);
    Real u0 = prob_parm.Mobj * a0;
    Real Pt = prob_parm.p0;

    Real rhot, phi, Tt, Y_fuel, Y_oxid, Y_prod, uxt, uyt;

    Real x_mid = (prob_lo[0] + prob_hi[0]) / 2.0;

    if (x < x_mid) {
       // Rich Region (Left Side)
       rhot = prob_parm.rich.rhot;

       Y_fuel = prob_parm.rich.Y_fuel;
       Y_oxid = prob_parm.rich.Y_oxid;
       Y_prod = prob_parm.rich.Y_prod;
       Pt = prob_parm.p0;
       uxt = u0; 
       uyt = Real(0.0);
       
 
    } else {
       // Lean Region (Right Side)
       rhot = prob_parm.lean.rhot;
             
       Y_fuel = prob_parm.lean.Y_fuel;
       Y_oxid = prob_parm.lean.Y_oxid;
       Y_prod = prob_parm.lean.Y_prod;
       Pt = prob_parm.p0;
       uxt = u0; 
       uyt = Real(0.0);    

    }

       phi = round(prob_parm.OF_st / (Y_oxid / Y_fuel) * 100) / 100.0;

       parm.updateCDMParameters(phi);

       state(i,j,k,URHO ) = rhot;
       state(i,j,k,UMX  ) = rhot * uxt;
       state(i,j,k,UMY  ) = rhot * uyt;
#if (AMREX_SPACEDIM == 3)
       state(i,j,k,UMZ  ) = Real(0.0);
#endif
       Real et = Pt/(parm.eos_gamma-Real(1.0));

       state(i,j,k,UEINT) = et;
       state(i,j,k,UEDEN) = et + Real(0.5)*(rhot * (uxt * uxt + uyt * uyt) );
       state(i,j,k,UTEMP) = Pt / (parm.Rsp * rhot);
    
       state(i, j, k, UPHI) = phi;
       state(i, j, k, URHOY_FUEL) = rhot * Y_fuel;
       state(i, j, k, URHOY_OXID) = rhot * Y_oxid;
       state(i, j, k, URHOY_PROD) = rhot * Y_prod;

       state(i, j, k, UYF_UN) = Y_fuel;
       state(i, j, k, UYA_UN) = Y_oxid;
       state(i, j, k, UY_P) = Real(1.0);


    //Real cs = std::sqrt(parm.eos_gamma * Pt / rhot);
       state(i,j,k,SFOIL) = Real(0.0);

       amrex::Print() << "CHECK INITDATA: i=" << i << ", j=" << j << ", k=" << k
                      
                      << ", Phi=" << phi
                      << ", Y_f_u=" << state(i, j, k, UYF_UN)
                      << ", Y_f_a=" << state(i, j, k, UYA_UN)
                      << ", Y_p=" << state(i, j, k, UY_P)
                      << "\n";

}


AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
init_probparams (amrex::GeometryData const& geomdata, Parm const& parm, ProbParm& prob_parm, ProbParm& dprob_parm)
{

}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_progress_variable_and_phi(int i, int j, int k, 
       amrex::Array4<amrex::Real> const& state, 
       Parm const& parm, ProbParm const& prob_parm) noexcept 
{
    using amrex::Real;

    //Current fuel and oxidizer mass fractions
    Real Y_f = state(i, j, k, URHOY_FUEL) / state(i, j, k, URHO);
    Real Y_a = state(i, j, k, URHOY_OXID) / state(i, j, k, URHO);
    Real Y_pr = state(i, j, k, URHOY_PROD)/state(i, j, k, URHO);

    // Stoichiometric fuel and oxidizer mass fractions
    Real Y_a_st = prob_parm.OF_st / (1.0 + prob_parm.OF_st);
    Real Y_f_st = 1.0 / (1.0 + prob_parm.OF_st);

    Real phi = state(i, j, k, UPHI);

    
    Real Y_f_u = 0.0;
    Real Y_a_u = 0.0;
    
    Y_f_u = Y_f + Y_f_st * Y_pr;
    Y_a_u = Y_a + Y_a_st * Y_pr;

    Real epsilon = 1e-8;
    // Ensure non-negative mass fractions
    Y_f_u = std::max(Y_f_u,epsilon);
    Y_a_u = std::max(Y_a_u,epsilon);

    // Check for small fuel mass fraction
    if (Y_f_u < 1e-6) {
        phi = 1.0; // Set phi to 1.0 if fuel is nearly depleted
    } else {
        // Calculate actual oxidizer-to-fuel ratio
        Real OF_act = Y_a_u / Y_f_u;
        // Update equivalence ratio
        phi = prob_parm.OF_st / OF_act;
    }

    //progress variable
    Real Y_p = 1.0;
    if (phi < 1.0) {
        Y_p = amrex::min(1.0, Y_f / Y_f_u);
    } else {
        Y_p = amrex::min(1.0, Y_a / Y_a_u);
    }

    state(i, j, k, UPHI) = phi;
    state(i, j, k, UY_P) = Y_p;
    state(i, j, k, UYF_UN) = Y_f_u;
    state(i, j, k, UYA_UN) = Y_a_u;

//#ifndef AMREX_USE_GPU
//amrex::Print() << "UPDATE: "
  //             << ", i=" << i << ", j=" << j << ", k=" << k
    //           << ", Phi (after update)=" << state(i, j, k, UPHI)
      //         << ", Y_f_u=" << state(i, j, k, UYF_UN)
        //       << ", Y_a_u=" << state(i, j, k, UYA_UN)
          //     << ", QPROGVAR (Y_p)=" << state(i, j, k, UY_P)
            //   << "\n";
//#endif

}



AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
cns_probspecific_bc(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NGROW][NUM_STATE],
  amrex::Real s_ext[NUM_STATE],
  const int idir,
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
}


AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
cns_tag_probspecific(int i, int j, int k, amrex::Array4<char> const& tag, 
                    amrex::Array4<amrex::Real > const& sarr, 
                    amrex::GeometryData const& geomdata, 
                    char tagval, 
                    amrex::Array4<amrex::EBCellFlag const> const& flag,
                    Parm const& parm, ProbParm const& prob_parm, amrex::Real time,
                    int level)
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i+1,j,k).isRegular() && flag(i-1,j,k).isRegular() && level < prob_parm.refuptolev){

        const Real* prob_lo = geomdata.ProbLo();
        const Real* dx      = geomdata.CellSize();
        Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
        Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
#if AMREX_SPACEDIM==3
        Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
#endif

        //Real axY = 0.0, axT = 0.0, axP = 0.0;
        Real axY_f = 0.0, axY_o = 0.0, axY_p = 0.0, axT = 0.0, axP = 0.0;
        //Real ayY = 0.0, ayT = 0.0, ayP = 0.0;
        if(flag(i,j,k).isConnected(1,0,0)){

            axY_f = amrex::Math::abs((sarr(i+1,j,k,URHOY_FUEL) / sarr(i+1,j,k,URHO)) 
                                   - (sarr(i,j,k,URHOY_FUEL) / sarr(i,j,k,URHO)));

            axY_o = amrex::Math::abs((sarr(i+1,j,k,URHOY_OXID) / sarr(i+1,j,k,URHO)) 
                                   - (sarr(i,j,k,URHOY_OXID) / sarr(i,j,k,URHO)));

            axY_p = amrex::Math::abs((sarr(i+1,j,k,URHOY_PROD) / sarr(i+1,j,k,URHO)) 
                                   - (sarr(i,j,k,URHOY_PROD) / sarr(i,j,k,URHO)));

            axT = amrex::Math::abs(sarr(i+1,j,k,UTEMP) - sarr(i,j,k,UTEMP));
            axP = (parm.eos_gamma-1.) * amrex::Math::abs(sarr(i+1,j,k,UEINT) - sarr(i,j,k,UEINT));
        }

    
        if(flag(i,j,k).isConnected(-1,0,0)){
            axY_f = amrex::max(axY_f, 
                amrex::Math::abs((sarr(i,j,k,URHOY_FUEL) / sarr(i,j,k,URHO)) 
                                - (sarr(i-1,j,k,URHOY_FUEL) / sarr(i-1,j,k,URHO))));

            axY_o = amrex::max(axY_o, 
                amrex::Math::abs((sarr(i,j,k,URHOY_OXID) / sarr(i,j,k,URHO)) 
                                - (sarr(i-1,j,k,URHOY_OXID) / sarr(i-1,j,k,URHO))));

            axY_p = amrex::max(axY_p, 
                amrex::Math::abs((sarr(i,j,k,URHOY_PROD) / sarr(i,j,k,URHO)) 
                                - (sarr(i-1,j,k,URHOY_PROD) / sarr(i-1,j,k,URHO))));

            axT = amrex::max(axT, amrex::Math::abs(sarr(i,j,k,UTEMP) - sarr(i-1,j,k,UTEMP)));
            axP = amrex::max(axP, (parm.eos_gamma-1.) * amrex::Math::abs(sarr(i,j,k,UEINT) - sarr(i-1,j,k,UEINT)));

        }

        Real gymag = std::sqrt(axY_f * axY_f + axY_o * axY_o + axY_p * axY_p);
        Real gTmag = std::sqrt(axT*axT);
        Real gPmag = std::sqrt(axP*axP);

        Real minloc = prob_parm.flameloc - prob_parm.xreflo;
        Real maxloc = prob_parm.flameloc + prob_parm.xrefhi;
    
        if((!flag(i,j,k).isCovered())){
            if(gymag >= prob_parm.deltaY 
                || gTmag >= prob_parm.deltaT
                || gPmag >= prob_parm.deltaP*PREATM
                || (x >= minloc && x <= minloc
#if AMREX_SPACEDIM==3
                && z >= prob_parm.zreflo && z <= prob_parm.zrefhi
#endif
                ) 
                ){
                    tag(i,j,k) = tagval;
            } 
        }
    }
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real
get_flame_location (int i, int j, int k, 
    amrex::GeometryData const& geomdata, 
    amrex::Array4<amrex::Real const> const& state,
    Parm const& parm) noexcept      
{
    using amrex::Real; 

    Real flamelocate = geomdata.ProbLo(0);  

    Real Y_p = state(i, j, k, UY_P);
  
    // FLAME LOCATION IS THE LARGEST VALUE OF x FOR
    // WHICH REACTANT (FUEL) MASS FRACTION DROPS BELOW 0.5

    if (Y_p < 0.5) 
       flamelocate = geomdata.ProbLo(0) + (i+0.5)*geomdata.CellSize(0);

    return flamelocate;
} 

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real
get_shock_location (int i, int j, int k, 
    amrex::GeometryData const& geomdata, 
    amrex::Array4<Real const> const& state,
    Parm const& parm) noexcept      
{
    Real shloc = geomdata.ProbLo(0);
    // SHOCK LOCATION IS THE LARGEST VALUE OF x FOR
    // WHICH PRESSURE IS GREATER THAN 1.5 atm
    if((parm.eos_gamma-1.)*state(i,j,k,UEINT) > 1.5*PREATM)
        shloc = geomdata.ProbLo(0) + (i+0.5)*geomdata.CellSize(0);
    return shloc;
} 


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real
global_energy_release_rate (int i, int j, int k, 
    amrex::GeometryData const& geomdata, 
    amrex::Array4<Real const> const& state,
    Parm const& parm) noexcept      
{
    Real Y_p = state(i, j, k, UY_P);

    Real omegarhoq = state(i,j,k,URHO) * parm.pre_exp * Y_p 
                          * std::exp(-parm.Ea_dim / (parm.Ru * state(i,j,k,UTEMP)))
                          * parm.q_dim;
    return omegarhoq;
}   

// THIS FUNCTION IS CALLED BY THE CPU AT THE END OF EVERY TIMESTEP
// (IE) TIMESTEP AT LEVEL ZERO
// THIS FUNCTION CAN BE USED TO COMPUTE THE LOCAL SHOCK AND REACTION FRONT 
// LOCATION, REACTION FRONT SPEED ETC. THESE QUANTITIES MUST BE DEFINED 
// IN prob_parm

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
cns_probspecific_func (amrex::MultiFab& State, 
                       amrex::GeometryData const& geomdata, 
                       int write_data, Parm const& parm, 
                       ProbParm& hprob_parm, 
                       ProbParm& dprob_parm, 
                       amrex::Real time, amrex::Real dt
                       , int level)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    Real shloc = hprob_parm.shloc, flameloc = hprob_parm.flameloc, energy_grr = 0.0;

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(State.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

  // THIS SECTION DEALS WITH WRITING FLAME AND SHOCK LOCATION DATA FOR SPECIFIC ZONES IN THE Y
  // DIRECTION
  for(int ii = 0; ii < hprob_parm.nzones; ++ii){

        Real ylo = hprob_parm.yloz[ii]; Real yhi = hprob_parm.yhiz[ii];

        int myproc = ParallelDescriptor::MyProc();

        Real shloc = hprob_parm.shl[ii], flameloc = hprob_parm.flamel[ii];

        ReduceOps<ReduceOpMax> reduce_op_flame;
        ReduceData<Real> reduce_data_flame(reduce_op_flame);
        using ReduceTupleFlame = typename decltype(reduce_data_flame)::Type;

        ReduceOps<ReduceOpMax> reduce_op_shock;
        ReduceData<Real> reduce_data_shock(reduce_op_shock);
        using ReduceTupleShock = typename decltype(reduce_data_shock)::Type;

#ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
#endif


        for (MFIter mfi(State, false); mfi.isValid(); ++mfi) 
        {
            const Box& bx = mfi.tilebox();          
            const auto& flag = flags[mfi];         
            auto const& s_arr = State.array(mfi);   

  
            if (flag.getType(bx) != FabType::covered) 
            {
        
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
                {
                    update_progress_variable_and_phi(i, j, k, s_arr, parm, hprob_parm);
                });
            }
        }

        for (MFIter mfi(State,false); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto& flag = flags[mfi];
            auto const& s_arr = State.array(mfi);
            if (flag.getType(bx) != FabType::covered)
            {
                // GET LOCATION OF REACTION FRONT
               reduce_op_flame.eval(bx, reduce_data_flame, [=]
               AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleFlame
               {
                    Real y = geomdata.ProbLo(1) + (j+Real(0.5))*geomdata.CellSize(1);
                    if( y >= ylo && y <= yhi ){
                        return get_flame_location(i,j,k,geomdata,s_arr,parm);
                    }else{
                        Real retval = geomdata.ProbLo(1);
                        return retval;
                    }   
              });

                // GET LOCATION OF SHOCK (SHOCK AND REACTION FRONT BECOME 
                // COUPLED ONCE DDT OCCURS)
              reduce_op_shock.eval(bx, reduce_data_shock, [=]
              AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleShock
              {
                   Real y = geomdata.ProbLo(1) + (j+Real(0.5))*geomdata.CellSize(1);

                    if( y >= ylo && y <= yhi ){
                        return get_shock_location(i,j,k,geomdata,s_arr,parm);
                    }else{
                        Real retval = geomdata.ProbLo(1);
                        return retval;
                    }
                });

            }
        } // mfi 
        
        ReduceTupleFlame host_tuple_flame = reduce_data_flame.value();
        flameloc = amrex::get<0>(host_tuple_flame);

        ReduceTupleShock host_tuple_shock = reduce_data_shock.value();
        shloc = amrex::get<0>(host_tuple_shock);

        ParallelDescriptor::ReduceRealMax(flameloc);
        ParallelDescriptor::ReduceRealMax(shloc);
        
        if(myproc == 0){
            hprob_parm.flamel[ii] = amrex::max(flameloc, hprob_parm.flamel[ii]); 
            dprob_parm.flamel[ii] = amrex::max(flameloc, dprob_parm.flamel[ii]);
            hprob_parm.shl[ii]    = amrex::max(shloc, hprob_parm.shl[ii]); 
            dprob_parm.shl[ii]    = amrex::max(shloc, dprob_parm.shl[ii]);

            if(hprob_parm.write_to_file_zones == 1 && write_data == 1){
                std::ofstream ofs;
                if(hprob_parm.append_file_zones == 0){
                    
                    if(ii == hprob_parm.nzones-1) hprob_parm.append_file_zones = 1;
                    
                    std::string filename = hprob_parm.data_file_zonebase;
                    filename = filename + std::to_string(ii) + ".txt";
                    ofs.open(filename, std::ofstream::out);
                    if(!ofs)  amrex::Abort("Failed to create file to write zone flame and shock location..exiting!");
                    Print(myproc,ofs) << "# yloz =  " << hprob_parm.yloz[ii] 
                                  << ", yhiz = "  << hprob_parm.yhiz[ii] << "\n";
                    Print(myproc,ofs) << "# time flameloc (cm) shockloc (cm) " << "\n";
                }else{
                    std::string filename = hprob_parm.data_file_zonebase;
                    filename = filename + std::to_string(ii) + ".txt";
                    ofs.open(filename, std::ofstream::app);
                }

                Print(myproc, ofs).SetPrecision(8) << std::left << std::setw(12) << time << "\t"
                    << std::left << std::setw(12) << hprob_parm.flamel[ii]*100  << "\t"  
                    << std::left << std::setw(12) << hprob_parm.shl[ii]*100 << "\n";

                ofs.close();
            }
        }
    }  // for(ii)

    ParallelDescriptor::Barrier(); 
    // shloc is a TEMPORARY COPY OF THE SHOCK LOCATION,
    // flameloc is a TEMPORARY COPY OF THE REACTION FRONT LOCATION
    // Reduce max operation
    ReduceOps<ReduceOpMax> reduce_op_flame;
    ReduceData<Real> reduce_data_flame(reduce_op_flame);
    using ReduceTupleFlame = typename decltype(reduce_data_flame)::Type;

    ReduceOps<ReduceOpMax> reduce_op_shock;
    ReduceData<Real> reduce_data_shock(reduce_op_shock);
    using ReduceTupleShock = typename decltype(reduce_data_shock)::Type;

    // REDUCE SUM OPERATION TO COMPUTE THE GLOBAL ENERGY RELEASE RATE (rho * q * omega)
    ReduceOps<ReduceOpSum> reduce_op_enrelease;
    ReduceData<Real> reduce_data_enrelease(reduce_op_enrelease);
    using ReduceTupleEnrelease = typename decltype(reduce_data_enrelease)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif


    for (MFIter mfi(State,false); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const auto& flag = flags[mfi];
        auto const& s_arr = State.array(mfi);
        if (flag.getType(bx) != FabType::covered)
        {
            // GET LOCATION OF REACTION FRONT
          reduce_op_flame.eval(bx, reduce_data_flame, [=]
             AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleFlame
             {
                 return get_flame_location(i,j,k,geomdata,s_arr,parm);
             });

          // GET LOCATION OF SHOCK (SHOCK AND REACTION FRONT BECOME 
          // COUPLED ONCE DDT OCCURS)
          reduce_op_shock.eval(bx, reduce_data_shock, [=]
             AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleShock
             {
                 return get_shock_location(i,j,k,geomdata,s_arr,parm);
             });

          if(level == 0 && write_data == 1){
            // GET THE GLOBAL ENERGY RELEASE RATE (COMPUTED ONLY ON LEVEL 0)
            reduce_op_enrelease.eval(bx, reduce_data_enrelease, [=]
             AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleEnrelease
             {
                 return global_energy_release_rate(i,j,k,geomdata,s_arr,parm);
             });
          }

        }
    } // mfi

    ReduceTupleFlame host_tuple_flame = reduce_data_flame.value();
    flameloc = amrex::get<0>(host_tuple_flame);

    ReduceTupleShock host_tuple_shock = reduce_data_shock.value();
    shloc = amrex::get<0>(host_tuple_shock);

    ReduceTupleEnrelease host_tuple_grr = reduce_data_enrelease.value();
    energy_grr = amrex::get<0>(host_tuple_grr);

    ParallelDescriptor::ReduceRealMax(flameloc);
    ParallelDescriptor::ReduceRealMax(shloc);

    ParallelDescriptor::ReduceRealSum(energy_grr);

    int myproc = ParallelDescriptor::MyProc();

    if(myproc == 0){
        hprob_parm.flameloc = amrex::max(flameloc, hprob_parm.flameloc); 
        dprob_parm.flameloc = amrex::max(flameloc, dprob_parm.flameloc);
        hprob_parm.shloc    = amrex::max(shloc, hprob_parm.shloc); 
        dprob_parm.shloc    = amrex::max(shloc, dprob_parm.shloc);

        if(hprob_parm.write_to_file == 1 && write_data == 1){
            std::ofstream ofs;
            if(hprob_parm.append_file == 0){
                hprob_parm.append_file = 1;
                ofs.open(hprob_parm.data_file, std::ofstream::out);
                if(!ofs)  amrex::Abort("Failed to create file to write flame and shock location..exiting!");
                Print(myproc,ofs) << "# time flameloc (cm) shockloc (cm) global_energy_release_rate" << "\n";
            }else{
                ofs.open(hprob_parm.data_file, std::ofstream::app);
            }

            Print(myproc, ofs).SetPrecision(8) << std::left << std::setw(12) << time << "\t"
                << std::left << std::setw(12) << hprob_parm.flameloc*100  << "\t"  
                << std::left << std::setw(12) << hprob_parm.shloc*100 << "\t"
                << std::left << std::setw(12) << energy_grr << "\n";

            ofs.close();

            // hprob_parm.flameloc = 0.0; dprob_parm.flameloc = 0.0;
            // hprob_parm.shloc = 0.0; dprob_parm.shloc = 0.0;
        }
    }

    ParallelDescriptor::Barrier(); 
}    

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
init_eb_geometry (const amrex::Geometry& geom, const int max_coarsening_level)
{



}           


#endif // CNS_PROB_H_

