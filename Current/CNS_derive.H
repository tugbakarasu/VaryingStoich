#ifndef CNS_DERIVE_H_
#define CNS_DERIVE_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

#include <AMReX_ParmParse.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_BoxArray.H>
#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF_Union.H>
#include <AMReX_EB2_IF_Intersection.H>
#include <AMReX_EB2_IF_Complement.H>
#include <AMReX_EB2_IF_Scale.H>
#include <AMReX_EB2_IF_Translation.H>
#include <AMReX_EB2_IF_Lathe.H>
#include <AMReX_EB2_IF_Box.H>
#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_EB2_IF_Ellipsoid.H>
#include <AMReX_EB2_IF_Sphere.H>
#include <AMReX_EB2_IF_Plane.H>
#include <AMReX_EB2_IF_Extrusion.H>
#include <AMReX_EB2_GeometryShop.H>
#include <AMReX_EB2_IF_Difference.H>
#include <AMReX_EBFluxRegister_C.H>
#include <AMReX_EB_utils.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
get_idx(const int i, int& im, int& ip)
{
  im = i - 1;
  ip = i + 1;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
get_idx(
  const int i,
  const int dir,
  const bool all_regular,
  const amrex::EBCellFlag& flag,
  int& im,
  int& ip)
{
  if (all_regular) {
    get_idx(i, im, ip);
  } else {
    if (flag.isCovered()) {
      im = i;
      ip = i;
    } else {
      const amrex::IntVect ivm = -amrex::IntVect::TheDimensionVector(dir);
      const amrex::IntVect ivp = amrex::IntVect::TheDimensionVector(dir);
      im = i - static_cast<int>(flag.isConnected(ivm));
      ip = i + static_cast<int>(flag.isConnected(ivp));
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
get_weight(const int im, const int ip)
{
  const int diff = ip - im;
  return diff == 0 ? 0.0 : (diff == 1 ? 1.0 : 0.5);
}

void cns_derpres (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                  const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                  amrex::Real time, const int* bcrec, int level);

void cns_dervel (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);
                 
void cns_derYF (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);   
                 
void cns_derYO (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level); 
                 
void cns_derYP (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);                                                

void cns_derschlieren (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_dershadowgraph (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_dermach (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_derYFderiv (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);
                 
void cns_derYOderiv (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);
                 
void cns_derYPderiv (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);                 

void cns_derrhoderiv (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_derprederiv (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_dervort (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_deruderiv (const amrex::Box& bx, amrex::FArrayBox& uderfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_dervderiv (const amrex::Box& bx, amrex::FArrayBox& vderfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

#if AMREX_SPACEDIM==3
void cns_derwderiv (const amrex::Box& bx, amrex::FArrayBox& wderfab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);
#endif

void cns_dermu (const amrex::Box& bx, amrex::FArrayBox& mufab, int dcomp, int ncomp,
                  const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                  amrex::Real time, const int* bcrec, int level);

void cns_derdivu (const amrex::Box& bx, amrex::FArrayBox& divufab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_dertau (const amrex::Box& bx, amrex::FArrayBox& divufab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_deromdivu (const amrex::Box& bx, amrex::FArrayBox& omdivufab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_derbaroclinic (const amrex::Box& bx, amrex::FArrayBox& barofab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

void cns_deromdiff (const amrex::Box& bx, amrex::FArrayBox& omdiffab, int dcomp, int ncomp,
                 const amrex::FArrayBox& datafab, const amrex::Geometry& geomdata,
                 amrex::Real time, const int* bcrec, int level);

#if AMREX_SPACEDIM==3
void cns_deromdelu (const amrex::Box& bx, amrex::FArrayBox& omdufab, int dcomp, int /*ncomp*/,
                  const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
                  amrex::Real /*time*/, const int* /*bcrec*/, int level);
#endif

#endif
