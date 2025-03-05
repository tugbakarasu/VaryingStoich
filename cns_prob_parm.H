#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <vector>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p0 = 1.01325e5;
    amrex::Real T0 = 298.0;
    amrex::Real rho0 = 1.0;
    amrex::Real Dcj = 1.0;
    amrex::Real Mcj = 1.0;
    amrex::Real rhovn = 298.0;
    amrex::Real Tvn = 298.0;
    amrex::Real pvn = 101325.0;
    amrex::Real upvn = 0.0;
    amrex::Real uwvn = 0.0;

    amrex::Real Tcj = 298.0;
    amrex::Real pcj = 1.01325e5;
    amrex::Real rhocj = 1.;
    amrex::Real upcj = 0.0;

    // Half reaction distance (this is used to determine thickness of reaction zone) (in metres)
    amrex::Real xd = 1.e-5;

    amrex::Real deltaY = 0.0;
    amrex::Real deltaT = 0.0;
    amrex::Real deltaP = 0.0;
    amrex::Real od_factor = 1.0;
    amrex::Real Mobj = 0.0;

    // radhitemp : radius of high temperature burnt gas
    amrex::Real radhitemp = 0.05;
    
    // Variable stoichiometry stuff
    amrex::Real OF_st = 34.32;         //Stoichiometric oxidizer-fuel ratio
    amrex::Real Y_fuel_st = 0.0;      // Stoichiometric fuel mass fraction
    amrex::Real Y_oxid_st = 0.0;      // Stoichiometric oxidizer mass fraction
    
    amrex::Real rich_rhot = 0.0;
    amrex::Real rich_Yf = 0.0;
    amrex::Real rich_Yox = 0.0;
    amrex::Real rich_Yp = 0.0;
    
    amrex::Real lean_rhot = 0.0;
    amrex::Real lean_Yf = 0.0;
    amrex::Real lean_Yox = 0.0;
    amrex::Real lean_Yp = 0.0;

    // QUANTITIES TO LOCATE SHOCK, REACTION FRONT
    amrex::Real shloc     = 0.; 
    amrex::Real flameloc  = 0.;

    amrex::Real* shl        = nullptr;
    amrex::Real* flamel     = nullptr;
    amrex::Real* yloz       = nullptr;
    amrex::Real* yhiz       = nullptr;

    int nzones = 0;

    // SOME VARIABLES FOR WRITING DATA TO FILE
    int append_file = 0; 
    int write_to_file = 1;
    std::string data_file = "pressuredata.txt";

    // VARIABLES FOR WRITING DATA TO FILE FROM PARTICULAR Y LOCATIONS 
    // DENOTED BY ylo and yhi
    int append_file_zones = 0; 
    int write_to_file_zones = 0;
    std::string data_file_zonebase = "pressuredatazone.txt";

    // time1 (in seconds)
    // If time <= time1, then pure air (Y = 0) enters at reference conditions (p0, T0)
    // speed corresponding to Mobj
    // Set time1 as a fairly large value to ensure that the bow shock and flow behind obstacles
    // reaches a "steady" state
    amrex::Real time1 = 1.e-3;

    // time2 (MUST be greater than time1)
    // If time > time1 and time <= time2, fuel (Y=1) enters at reference conditions (p0, T0)
    // and speed corresponding to Mobj
    // This part ensures that the domain is filled with sufficient fuel for a detonation to propagate
    amrex::Real time2 = 1.2e-3;

    // t_inflow (express this as how much time above time2 will we have supersonic inflow)
    // When time > time2 and time <= time2 + t_inflow, gas at von Neumann state and moving shock at 
    // Dcj will enter the domain to initiate a detonation
    amrex::Real t_inflow = 1.e-6;
    // When time > time2 + t_inflow, the inflow boundary becomes a wall

    // FOR GETTING GOOD SOOTFOIL, WE REFINE UPTO A CERTAIN LENGTH BEHIND THE FLAME FRONT
    // BY DEFAULT WE SET THIS TO ZERO. IF xreflo = 0.01, THEN WE REFINE ALL CELLS 
    // 1 cm BEHIND THE FLAME FRONT 
    amrex::Real xreflo = 0.0;
    amrex::Real xrefhi = 0.0;
    amrex::Real yreflo = 0.0;
    amrex::Real yrefhi = 0.0;

    int refuptolev = 10;

    ProbParm ();
    ~ProbParm ();
};

#endif
