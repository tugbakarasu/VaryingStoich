
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include "CNS.H"
#include "cns_prob.H"
#include "FCT_NSCBC.H"

using namespace amrex;

struct CnsFillExtDir
{
    ProbParm const* lprobparm;
    Parm const* lparm;

    AMREX_GPU_HOST
    constexpr explicit CnsFillExtDir(const ProbParm* d_prob_parm, const Parm* d_parm)
        : lprobparm(d_prob_parm), lparm(d_parm)
    { 
    }

    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& data,
                     const int dcomp, const int numcomp,
                     GeometryData const& geom, const Real time,
                     const BCRec* bcr, const int bcomp,
                     const int orig_comp) const
        {
            using namespace amrex;

            const int* bc = bcr->data();
            // domlo is a vector of type "int" that stores the indices 
            // for the lower end of the domain in each coordinate direction
            const int* domlo = geom.Domain().loVect();
            // domhi is a vector of type "int" that stores the indices 
            // for the upper end of the domain in each coordinate direction 
            const int* domhi = geom.Domain().hiVect();
            int nc = data.nComp();

            // iv is a vector that stores the indices (i,j,k) of a ghost cell
            //  // xlo and xhi
            int idir = 0;

            // User defined BC
            if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
                const amrex::Real* prob_lo = geom.ProbLo();
                const amrex::Real* dx = geom.CellSize();
                const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
                prob_lo[0] + static_cast<amrex::Real>(iv[0] + 0.5) * dx[0],
                prob_lo[1] + static_cast<amrex::Real>(iv[1] + 0.5) * dx[1],
                prob_lo[2] + static_cast<amrex::Real>(iv[2] + 0.5) * dx[2])};
                // Here NGROW = 8, this means we can fill upto 8 ghost cells
                amrex::Real s_int[NGROW][NUM_STATE] = {0.0};
                // s_ext is an array that contains the value that is obtained from a function
                // This is to be copied into the ghost cells
                amrex::Real s_ext[NUM_STATE] = {0.0};

                for (int ng = 0; ng < NGROW; ++ng){
                    amrex::IntVect loc(AMREX_D_DECL(domlo[idir]+ng, iv[1], iv[2]));
                    for (int n = URHO; n < nc; n++) {
                        s_int[ng][n] = data(loc, n);
                    }
                }
                cns_probspecific_bc(x, s_int, s_ext, idir, iv[0], iv[1], 
#if AMREX_SPACEDIM==3
                    iv[2],
#endif 
                    -1, time, geom, *lprobparm, *lparm, data);
                for (int n = dcomp; n < dcomp+numcomp; n++) {
                    data(iv, n) = s_ext[n];
                }
            } 
            else if (
              (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
              (iv[idir] > domhi[idir])) {
                const amrex::Real* prob_lo = geom.ProbLo();
                const amrex::Real* dx = geom.CellSize();
                const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
                prob_lo[0] + static_cast<amrex::Real>(iv[0] + 0.5) * dx[0],
                prob_lo[1] + static_cast<amrex::Real>(iv[1] + 0.5) * dx[1],
                prob_lo[2] + static_cast<amrex::Real>(iv[2] + 0.5) * dx[2])};
                // Here NGROW = 8, this means we can fill upto 8 ghost cells
                amrex::Real s_int[NGROW][NUM_STATE] = {0.0};
                // s_ext is an array that contains the value that is obtained from a function
                // This is to be copied into the ghost cells
                amrex::Real s_ext[NUM_STATE] = {0.0};

                for(int ng = 0; ng < NGROW; ++ng){
                    amrex::IntVect loc(AMREX_D_DECL(domhi[idir]-ng, iv[1], iv[2]));
                    for (int n = URHO; n < nc; n++) {
                        s_int[ng][n] = data(loc, n);
                    }                 
                }
                cns_probspecific_bc(x, s_int, s_ext, idir, iv[0], iv[1], 
#if AMREX_SPACEDIM==3
                    iv[2], 
#endif            
                    1, time, geom, *lprobparm, *lparm, data);
                for (int n = dcomp; n < dcomp+numcomp; n++) {
                    data(iv, n) = s_ext[n];
                }
            }

            // y-direction
            idir = 1;
            // User defined BC
            if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
                const amrex::Real* prob_lo = geom.ProbLo();
                const amrex::Real* dx = geom.CellSize();
                const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
                prob_lo[0] + static_cast<amrex::Real>(iv[0] + 0.5) * dx[0],
                prob_lo[1] + static_cast<amrex::Real>(iv[1] + 0.5) * dx[1],
                prob_lo[2] + static_cast<amrex::Real>(iv[2] + 0.5) * dx[2])};
                // Here NGROW = 8, this means we can fill upto 8 ghost cells
                amrex::Real s_int[NGROW][NUM_STATE] = {0.0};
                // s_ext is an array that contains the value that is obtained from a function
                // This is to be copied into the ghost cells
                amrex::Real s_ext[NUM_STATE] = {0.0};

                for (int ng = 0; ng < NGROW; ++ng){
                    amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir]+ng, iv[2]));
                    for (int n = URHO; n < nc; n++) {
                        s_int[ng][n] = data(loc, n);
                    }
                }
                cns_probspecific_bc(x, s_int, s_ext, idir, iv[0], iv[1], 
#if AMREX_SPACEDIM==3
                    iv[2],
#endif 
                    -1, time, geom, *lprobparm, *lparm, data);
                for (int n = dcomp; n < dcomp+numcomp; n++) {
                    data(iv, n) = s_ext[n];
                }
            } 
            else if (
              (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
              (iv[idir] > domhi[idir])) {
                const amrex::Real* prob_lo = geom.ProbLo();
                const amrex::Real* dx = geom.CellSize();
                const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
                prob_lo[0] + static_cast<amrex::Real>(iv[0] + 0.5) * dx[0],
                prob_lo[1] + static_cast<amrex::Real>(iv[1] + 0.5) * dx[1],
                prob_lo[2] + static_cast<amrex::Real>(iv[2] + 0.5) * dx[2])};
                // Here NGROW = 8, this means we can fill upto 8 ghost cells
                amrex::Real s_int[NGROW][NUM_STATE] = {0.0};
                // s_ext is an array that contains the value that is obtained from a function
                // This is to be copied into the ghost cells
                amrex::Real s_ext[NUM_STATE] = {0.0};
                
                for(int ng = 0; ng < NGROW; ++ng){
                    amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir]-ng, iv[2]));
                    for (int n = URHO; n < nc; n++) {
                        s_int[ng][n] = data(loc, n);
                    }                 
                }
                cns_probspecific_bc(x, s_int, s_ext, idir, iv[0], iv[1], 
#if AMREX_SPACEDIM==3
                    iv[2], 
#endif            
                    1, time, geom, *lprobparm, *lparm, data);
                for (int n = dcomp; n < dcomp+numcomp; n++) {
                    data(iv, n) = s_ext[n];
                }
            }
        }
};

// bx                  : Cells outside physical domain and inside bx are filled.
// data, dcomp, numcomp: Fill numcomp components of data starting from dcomp.
// bcr, bcomp          : bcr[bcomp] specifies BC for component dcomp and so on.
// scomp               : component index for dcomp as in the descriptor set up in CNS::variableSetUp.

void cns_bcfill (Box const& bx, FArrayBox& data,
                 const int dcomp, const int numcomp,
                 Geometry const& geom, const Real time,
                 const Vector<BCRec>& bcr, const int bcomp,
                 const int scomp)
{
    const ProbParm* lprobparm = CNS::d_prob_parm;
    const Parm* lparm = CNS::d_parm;
    GpuBndryFuncFab<CnsFillExtDir> gpu_bndry_func(CnsFillExtDir{lprobparm, lparm});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}
