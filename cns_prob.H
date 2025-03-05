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
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    Real ly = prob_hi[1] - prob_lo[1];

    Real Pt, rhot, uxt, uyt, Y_f, Y_ox, Y_p;

    Real a0 = std::sqrt(parm.eos_gamma * parm.Rsp * prob_parm.T0);
    Real u0 = prob_parm.Mobj * a0;

    Real x_mid = (prob_lo[0] + prob_hi[0]) / 2;
    
    if (x < x_mid){
    	// Rich region
    	rhot 	= prob_parm.rich_rhot;
    	Y_f	= prob_parm.rich_Yf;
    	Y_ox	= prob_parm.rich_Yox;
    	Y_p	= prob_parm.rich_Yp;
    	Pt	= prob_parm.p0;
    	uxt	= u0;
    	uyt	= Real(0.0);
    } else {
    	// Lean region
    	rhot 	= prob_parm.lean_rhot;
    	Y_f	= prob_parm.lean_Yf;
    	Y_ox	= prob_parm.lean_Yox;
    	Y_p	= prob_parm.lean_Yp;
    	Pt	= prob_parm.p0;
    	uxt	= u0;
    	uyt	= Real(0.0);
    }
    
    Real phi = prob_parm.OF_st / (Y_ox/Y_f);
    
    parm->Update_CDM_Parameters(phi);
    
    Real cs = std::sqrt(parm.eos_gamma * Pt / rhot);
    
    // Conservative variables set up
    state(i,j,k,URHO ) = rhot;
    state(i,j,k,UMX  ) = rhot * uxt;
    state(i,j,k,UMY  ) = rhot * uyt;
#if (AMREX_SPACEDIM == 3)
    state(i,j,k,UMZ  ) = Real(0.0);
#endif
    state(i,j,k,URHOY_F) = rhot * Y_f;
    state(i,j,k,URHOY_O) = rhot * Y_o;
    state(i,j,k,URHOY_P) = rhot * Y_p;
    
    Real et = Pt/(parm.eos_gamma-Real(1.0));
    state(i,j,k,UEINT) = et;
    state(i,j,k,UEDEN) = et + Real(0.5)*(rhot * (uxt * uxt + uyt * uyt) );
    state(i,j,k,UTEMP) = Pt / (parm.Rsp * rhot);
    state(i,j,k,SFOIL) = Real(0.0);
}


AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
init_probparams (amrex::GeometryData const& geomdata, Parm const& parm, ProbParm& prob_parm, ProbParm& dprob_parm)
{
    using amrex::Real;

    prob_parm.rho0 = prob_parm.p0 / (parm.Rsp * prob_parm.T0);
    Real a0 = std::sqrt(parm.eos_gamma * parm.Rsp * prob_parm.T0);
    
    Real t1 = Real(0.5) * parm.q_dim * prob_parm.rho0 * (parm.eos_gamma*parm.eos_gamma-1.0) 
            / (prob_parm.p0 * parm.eos_gamma);
    prob_parm.Dcj = a0 * (std::sqrt(1.+t1) + std::sqrt(t1));
    prob_parm.Mcj = prob_parm.Dcj / a0;

    Real od = std::sqrt(prob_parm.od_factor);
    Real mach = od * prob_parm.Mcj;

    // Get the von-Neumann state variables (use normal shock relations)
    Real eta = 1.0 / (mach * mach);

    Real dvnd0 = (parm.eos_gamma+1.0) / (parm.eos_gamma - 1.0 + 2.0*eta);
    prob_parm.rhovn = prob_parm.rho0 * dvnd0;
    
    Real pvnp0 = 1. + (((2.0*parm.eos_gamma)/(parm.eos_gamma+1.0))*((1.0/eta) - 1.0));
    prob_parm.pvn = prob_parm.p0 * pvnp0;

    prob_parm.Tvn = prob_parm.pvn / (parm.Rsp * prob_parm.rhovn);

    Real u0 = prob_parm.Mobj * a0;
    prob_parm.upvn = (od * prob_parm.Dcj * (1. - (1.0/dvnd0))) + (u0 / dvnd0);
    prob_parm.uwvn = (od*prob_parm.Dcj) - prob_parm.upvn;

    // Get the lower and upper indices (global domain indices) for the ZND profile
    // Smaller index corresponds to CJ state and larger index corresponds to von-Neumann state
    // In the wave fixed frame, a detonation propagating from left to right has negative reactant and
    // product velocity
    // xsh is the location of the shock (von-Neumann state)
    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    Real d0d1 = (Real(1.0) + (parm.eos_gamma*mach*mach)) / ((parm.eos_gamma + Real(1.0))*(mach*mach));
    prob_parm.rhocj = prob_parm.rho0 / d0d1;

    Real p1p0 = (Real(1.0) + (parm.eos_gamma*mach*mach)) / (parm.eos_gamma + Real(1.0));
    prob_parm.pcj = prob_parm.p0 * p1p0;

    prob_parm.Tcj = prob_parm.pcj / (parm.Rsp * prob_parm.rhocj);
    prob_parm.upcj = od * prob_parm.Dcj * (1. - d0d1) + (d0d1 * u0);

    prob_parm.shloc = 0.0; prob_parm.flameloc = 0.0;
    dprob_parm.shloc = 0.0; dprob_parm.flameloc = 0.0;

    Print() << "eb_wallloss = " << parm.eb_wallloss 
            << ", ksolid = " << parm.ksolid
            << ", Tsolidwall = " << parm.tempsolidwall << "\n"; 
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

    if(flag(i,j,k).isRegular() && flag(i+1,j,k).isRegular() && flag(i-1,j,k).isRegular()
        && flag(i,j+1,k).isRegular() && flag(i,j-1,k).isRegular() && level < prob_parm.refuptolev){

        const Real* prob_lo = geomdata.ProbLo();
        const Real* dx      = geomdata.CellSize();
        Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
        Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
#if AMREX_SPACEDIM==3
        Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
#endif

        Real axY = 0.0, axT = 0.0, axP = 0.0;
        Real ayY = 0.0, ayT = 0.0, ayP = 0.0;
        if(flag(i,j,k).isConnected(1,0,0)){
            axY = amrex::Math::abs( (sarr(i+1,j,k,URHOY)/sarr(i+1,j,k,URHO)) 
                - (sarr(i,j,k,URHOY)/sarr(i,j,k,URHO)) );    
            axT = amrex::Math::abs( sarr(i+1,j,k,UTEMP) - sarr(i,j,k,UTEMP) );
            axP = (parm.eos_gamma-1.)*amrex::Math::abs( sarr(i+1,j,k,UEINT) - sarr(i,j,k,UEINT) );
        }
    
        if(flag(i,j,k).isConnected(0,1,0)){
            ayY = amrex::Math::abs( (sarr(i,j+1,k,URHOY)/sarr(i,j+1,k,URHO)) 
                - (sarr(i,j,k,URHOY)/sarr(i,j,k,URHO)) );
            ayT = amrex::Math::abs( sarr(i,j+1,k,UTEMP) - sarr(i,j,k,UTEMP) );
            ayP = (parm.eos_gamma-1.)*amrex::Math::abs( sarr(i,j+1,k,UEINT) - sarr(i,j,k,UEINT) );
        }
    
        if(flag(i,j,k).isConnected(-1,0,0)){
            axY = amrex::max(axY, 
            amrex::Math::abs((sarr(i,j,k,URHOY)/sarr(i,j,k,URHO))-(sarr(i-1,j,k,URHOY)/sarr(i-1,j,k,URHO))));

            axT = amrex::max(axT,amrex::Math::abs(sarr(i,j,k,UTEMP)-sarr(i-1,j,k,UTEMP)));

            axP = amrex::max(axP,
                (parm.eos_gamma-1.)*amrex::Math::abs(sarr(i,j,k,UEINT)-sarr(i-1,j,k,UEINT)));
        }

        if(flag(i,j,k).isConnected(0,-1,0)){
            ayY = amrex::max(ayY, 
            amrex::Math::abs((sarr(i,j,k,URHOY)/sarr(i,j,k,URHO))-(sarr(i,j-1,k,URHOY)/sarr(i,j-1,k,URHO))));

            ayT = amrex::max(ayT, amrex::Math::abs(sarr(i,j,k,UTEMP)-sarr(i,j-1,k,UTEMP)));

            ayP = amrex::max(ayP, (parm.eos_gamma-1.) 
                * amrex::Math::abs(sarr(i,j,k,UEINT)-sarr(i,j-1,k,UEINT)));
        }

        Real gymag = std::sqrt(axY*axY + ayY*ayY);
        Real gTmag = std::sqrt(axT*axT + ayT*ayT);
        Real gPmag = std::sqrt(axP*axP + ayP*ayP);

        Real minloc = prob_parm.flameloc - prob_parm.xreflo;
        Real maxloc = prob_parm.flameloc + prob_parm.xrefhi;
    
        if((!flag(i,j,k).isCovered())){
            if(gymag >= prob_parm.deltaY 
                || gTmag >= prob_parm.deltaT
                || gPmag >= prob_parm.deltaP*PREATM
                || (x >= minloc && x <= minloc && y >= prob_parm.yreflo && y <= prob_parm.yrefhi
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
    // FLAME LOCATION IS THE LARGEST VALUE OF x FOR
    // WHICH REACTANT MASS FRACTION DROPS BELOW 0.5
    if(state(i,j,k,URHOY) / state(i,j,k,URHO) < 0.5)
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
    Real omegarhoq = state(i,j,k,URHO) * parm.pre_exp * state(i,j,k,URHOY) 
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

    const Real* prob_lo = geom.ProbLo();
    const Real* prob_hi = geom.ProbHi();
    const Real* dx      = geom.CellSize();

    ParmParse pp("quench");
    RealArray bx1lo, bx2lo, bx3lo, bx4lo, bx5lo, bx6lo,
              bx1hi, bx2hi, bx3hi, bx4hi, bx5hi, bx6hi;

    pp.get("box1_lo", bx1lo); pp.get("box1_hi", bx1hi);
    pp.get("box2_lo", bx2lo); pp.get("box2_hi", bx2hi);
    pp.get("box3_lo", bx3lo); pp.get("box3_hi", bx3hi);
    pp.get("box4_lo", bx4lo); pp.get("box4_hi", bx4hi);
    pp.get("box5_lo", bx5lo); pp.get("box5_hi", bx5hi);
    pp.get("box6_lo", bx6lo); pp.get("box6_hi", bx6hi);

    EB2::BoxIF box1({bx1lo[0], bx1lo[1]}, {bx1hi[0], bx1hi[1]}, false);
    EB2::BoxIF box2({bx2lo[0], bx2lo[1]}, {bx2hi[0], bx2hi[1]}, false);
    EB2::BoxIF box3({bx3lo[0], bx3lo[1]}, {bx3hi[0], bx3hi[1]}, false);
    EB2::BoxIF box4({bx4lo[0], bx4lo[1]}, {bx4hi[0], bx4hi[1]}, false);
    EB2::BoxIF box5({bx5lo[0], bx5lo[1]}, {bx5hi[0], bx5hi[1]}, false);
    EB2::BoxIF box6({bx6lo[0], bx6lo[1]}, {bx6hi[0], bx6hi[1]}, false);

    auto bu = EB2::makeUnion(box1, box2, box3, box4, box5, box6);

#ifdef AMREX_USE_GPU
    using IF_t = decltype(bu);
    IF_t* dp = (IF_t*)The_Arena()->alloc(sizeof(bu));
    Gpu::htod_memcpy_async(dp, &bu, sizeof(IF_t));
    Gpu::streamSynchronize();
    EB2::DevicePtrIF<IF_t> dp_bu{dp};
    auto gshop = EB2::makeShop(dp_bu);
#else
    auto gshop = EB2::makeShop(bu);
#endif

    EB2::Build(gshop, geom, max_coarsening_level, max_coarsening_level, 8);

}           
#endif
