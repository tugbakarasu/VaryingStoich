#ifndef CNS_TAGGING_H_
#define CNS_TAGGING_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_TagBox.H>
#include <cmath>

#include <AMReX_EBFArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
cns_tag_denerror (int i, int j, int k,
                  amrex::Array4<char> const& tag,
                  amrex::Array4<amrex::Real const> const& rho,
                  amrex::Real dengrad_threshold, char tagval,
                  amrex::Array4<amrex::EBCellFlag const> const& flag) noexcept
{
    amrex::Real ax = amrex::Math::abs(rho(i+1,j,k) - rho(i,j,k));
    amrex::Real ay = amrex::Math::abs(rho(i,j+1,k) - rho(i,j,k));
    amrex::Real az = 0.0;
    ax = amrex::max(ax,amrex::Math::abs(rho(i,j,k) - rho(i-1,j,k)));
    ay = amrex::max(ay,amrex::Math::abs(rho(i,j,k) - rho(i,j-1,k)));
#if AMREX_SPACEDIM==3
    az = amrex::Math::abs(rho(i,j,k+1) - rho(i,j,k));
    az = amrex::max(az,amrex::Math::abs(rho(i,j,k) - rho(i,j,k-1)));
#endif
    if (amrex::max(ax,ay,az) >= dengrad_threshold && (!flag(i,j,k).isCovered())) {
        tag(i,j,k) = tagval;
    }
}

#endif
