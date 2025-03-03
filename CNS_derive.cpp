#include "CNS_derive.H"
#include "CNS.H"
#include "CNS_parm.H"

using namespace amrex;

void cns_derpres (const Box& bx, FArrayBox& pfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& rhoefab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const rhoe = rhoefab.array();
    auto       p    = pfab.array();
    Parm const* parm = CNS::d_parm;
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        p(i,j,k,dcomp) = (parm->eos_gamma-1.)*rhoe(i,j,k);
    });
}

void cns_dervel (const Box& bx, FArrayBox& velfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       vel = velfab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,dcomp) = dat(i,j,k,1)/dat(i,j,k,0);
    });
}

void cns_derschlieren (const Box& bx, FArrayBox& schfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int level)
{

    auto const dat = datfab.array();
    auto       sch = schfab.array();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        schfab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        Real gradmag = 0.0;
        if(all_regular){
            Real gradx = amrex::Math::abs( dat(i+1,j,k,0) - dat(i,j,k,0) );
            gradx = amrex::max( gradx, amrex::Math::abs( dat(i,j,k,0) - dat(i-1,j,k,0) ) );

            Real grady = amrex::Math::abs( dat(i,j+1,k,0) - dat(i,j,k,0) );
            grady = amrex::max( grady, amrex::Math::abs( dat(i,j,k,0) - dat(i,j-1,k,0) ) );

#if AMREX_SPACEDIM==3
            Real gradz = amrex::Math::abs( dat(i,j,k+1,0) - dat(i,j,k,0) );
            gradz = amrex::max( gradz, amrex::Math::abs( dat(i,j,k,0) - dat(i,j,k-1,0) ) );
#endif

            gradmag = std::sqrt(gradx*gradx + grady*grady
#if AMREX_SPACEDIM==3
                + gradz*gradz
#endif
                );
        }else{
            if(flags(i,j,k).isCovered()) gradmag = 0.0;
            else{
                Real gradx = 0.0;
                if(flags(i,j,k).isConnected(1,0,0)) 
                    gradx = amrex::max( gradx, amrex::Math::abs( dat(i+1,j,k,0) - dat(i,j,k,0) ) );
                if(flags(i,j,k).isConnected(-1,0,0))
                    gradx = amrex::max( gradx, amrex::Math::abs( dat(i,j,k,0) - dat(i-1,j,k,0) ) );

                Real grady = 0.0;
                if(flags(i,j,k).isConnected(0,1,0)) 
                    grady = amrex::max( grady, amrex::Math::abs( dat(i,j+1,k,0) - dat(i,j,k,0) ) );
                if(flags(i,j,k).isConnected(0,-1,0))
                    grady = amrex::max( grady, amrex::Math::abs( dat(i,j,k,0) - dat(i,j-1,k,0) ) );

#if AMREX_SPACEDIM==3
                Real gradz = 0.0;
                if(flags(i,j,k).isConnected(0,0,1)) 
                    gradz = amrex::max( gradz, amrex::Math::abs( dat(i,j,k+1,0) - dat(i,j,k,0) ) );
                if(flags(i,j,k).isConnected(0,0,-1))
                    gradz = amrex::max( gradz, amrex::Math::abs( dat(i,j,k,0) - dat(i,j,k-1,0) ) );
#endif
                gradmag = std::sqrt(gradx*gradx + grady*grady
#if AMREX_SPACEDIM==3
                    + gradz*gradz
#endif
                    );
            }
        }
        sch(i,j,k,dcomp) = gradmag;

    });

}

void cns_dershadowgraph (const Box& bx, FArrayBox& shafab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int level)
{

 // if(level == 0){
    auto const dat = datfab.array();
    auto       sha = shafab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real gxx = dat(i+1,j,k,0) - 2.0*dat(i,j,k,0) + dat(i-1,j,k,0);
        Real gyy = dat(i,j+1,k,0) - 2.0*dat(i,j,k,0) + dat(i,j-1,k,0);
#if AMREX_SPACEDIM==3
        Real gzz = dat(i,j,k+1,0) - 2.0*dat(i,j,k,0) + dat(i,j,k-1,0);
#endif
        Real gmag = gxx + gyy
#if AMREX_SPACEDIM==3
            + gzz
#endif
            ;
    
        sha(i,j,k,dcomp) = amrex::Math::abs(gmag);
    });
 // }
}

void cns_dermach (const Box& bx, FArrayBox& macfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       mac = macfab.array();
    Parm const* parm = CNS::d_parm;
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
#if AMREX_SPACEDIM==2
        Real cs = std::sqrt(parm->eos_gamma * parm->Rsp * dat(i,j,k,3));
        Real velmod = (1.0/dat(i,j,k,0)) 
            * std::sqrt( dat(i,j,k,1)*dat(i,j,k,1) + dat(i,j,k,2)*dat(i,j,k,2) ); 
#else
        Real cs = std::sqrt(parm->eos_gamma * parm->Rsp * dat(i,j,k,4));
        Real velmod = (1.0/dat(i,j,k,0)) 
            * std::sqrt( dat(i,j,k,1)*dat(i,j,k,1) + dat(i,j,k,2)*dat(i,j,k,2) + dat(i,j,k,3)*dat(i,j,k,3) ); 
#endif
        mac(i,j,k,dcomp) = velmod/cs;
    });
}

void cns_derYderiv (const Box& bx, FArrayBox& yderfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat = datfab.array();
    auto       yder = yderfab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        yderfab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate dYdx, dYdy, dYdz
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

        yder(i,j,k,dcomp) = wi * dxinv[0] 
                          * ( (dat(ip,j,k,URHOY_FUEL)/dat(ip,j,k,URHO)) - (dat(im,j,k,URHOY_FUEL)/dat(im,j,k,URHO)) );
        yder(i,j,k,dcomp+1) = wj * dxinv[1] 
                          * ( (dat(i,jp,k,URHOY_FUEL)/dat(i,jp,k,URHO)) - (dat(i,jm,k,URHOY_FUEL)/dat(i,jm,k,URHO)) );
#if AMREX_SPACEDIM==3
        yder(i,j,k,dcomp+2) = wk * dxinv[2] 
                          * ( (dat(i,j,kp,URHOY_FUEL)/dat(i,j,kp,URHO)) - (dat(i,j,km,URHOY_FUEL)/dat(i,j,km,URHO)) );
#endif
        yder(i,j,k,dcomp+3) = wi * dxinv[0] 
                          * ( (dat(ip,j,k,URHOY_OXID)/dat(ip,j,k,URHO)) - (dat(im,j,k,URHOY_OXID)/dat(im,j,k,URHO)) );
        yder(i,j,k,dcomp+4) = wj * dxinv[1] 
                          * ( (dat(i,jp,k,URHOY_OXID)/dat(i,jp,k,URHO)) - (dat(i,jm,k,URHOY_OXID)/dat(i,jm,k,URHO)) );
#if AMREX_SPACEDIM==3
        yder(i,j,k,dcomp+5) = wk * dxinv[2] 
                          * ( (dat(i,j,kp,URHOY_OXID)/dat(i,j,kp,URHO)) - (dat(i,j,km,URHOY_OXID)/dat(i,j,km,URHO)) );
#endif
        yder(i,j,k,dcomp+6) = wi * dxinv[0] 
                          * ( (dat(ip,j,k,URHOY_PROD)/dat(ip,j,k,URHO)) - (dat(im,j,k,URHOY_PROD)/dat(im,j,k,URHO)) );
        yder(i,j,k,dcomp+7) = wj * dxinv[1] 
                          * ( (dat(i,jp,k,URHOY_PROD)/dat(i,jp,k,URHO)) - (dat(i,jm,k,URHOY_PROD)/dat(i,jm,k,URHO)) );
#if AMREX_SPACEDIM==3
        yder(i,j,k,dcomp+8) = wk * dxinv[2] 
                          * ( (dat(i,j,kp,URHOY_PROD)/dat(i,j,kp,URHO)) - (dat(i,j,km,URHOY_PROD)/dat(i,j,km,URHO)) );
#endif
    });
}

void cns_derrhoderiv(const Box& bx, FArrayBox& grofab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int level)
{

    auto const dat = datfab.array();
    auto       roder = grofab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        grofab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate drodx, drody, drodz
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

        roder(i,j,k,dcomp) = wi * dxinv[0] 
                          * ( dat(ip,j,k,URHO) - dat(im,j,k,URHO) );
        roder(i,j,k,dcomp+1) = wj * dxinv[1] 
                          * ( dat(i,jp,k,URHO) - dat(i,jm,k,URHO) );
#if AMREX_SPACEDIM==3
        roder(i,j,k,dcomp+2) = wk * dxinv[2] 
                          * ( dat(i,j,kp,URHO) - dat(i,j,km,URHO) );
#endif
    });

}

void cns_derprederiv(const Box& bx, FArrayBox& gprfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int level)
{

    auto const dat    = datfab.array();
    auto       preder = gprfab.array();
    const auto dxinv  = geomdata.InvCellSizeArray();
    Parm const* parm = CNS::d_parm;

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        gprfab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate dpdx, dpdy, dpdz
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

        preder(i,j,k,dcomp) = wi * dxinv[0] * (parm->eos_gamma-1.) 
                          * ( dat(ip,j,k,UEINT) - dat(im,j,k,UEINT) );
        preder(i,j,k,dcomp+1) = wj * dxinv[1] * (parm->eos_gamma-1.) 
                          * ( dat(i,jp,k,UEINT) - dat(i,jm,k,UEINT) );
#if AMREX_SPACEDIM==3
        preder(i,j,k,dcomp+2) = wk * dxinv[2] * (parm->eos_gamma-1.) 
                          * ( dat(i,j,kp,UEINT) - dat(i,j,km,UEINT) );
#endif
    });

}

void cns_dervort (const Box& bx, FArrayBox& vrtfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int level)
{

    auto const dat = datfab.array();
    auto       vrt = vrtfab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        vrtfab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate vorticity
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 ,   get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 ,   get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 ,   const amrex::Real wj = get_weight(jm, jp);
                 ,   const amrex::Real wk = get_weight(km, kp);)

        const amrex::Real dvdx = wi * dxinv[0] * ( (dat(ip,j,k,UMY)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMY)/dat(im,j,k,URHO)) );
        const amrex::Real dudy = wj * dxinv[1] * ( (dat(i,jp,k,UMX)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMX)/dat(i,jm,k,URHO)) );

#if AMREX_SPACEDIM==3
        const amrex::Real dwdy = wj * dxinv[1] * ( (dat(i,jp,k,UMZ)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMZ)/dat(i,jm,k,URHO)) );
        const amrex::Real dvdz = wk * dxinv[2] * ( (dat(i,j,kp,UMY)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMY)/dat(i,j,km,URHO)) );

        const amrex::Real dudz = wk * dxinv[2] * ( (dat(i,j,kp,UMX)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMX)/dat(i,j,km,URHO)) );
        const amrex::Real dwdx = wi * dxinv[0] * ( (dat(ip,j,k,UMZ)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMZ)/dat(im,j,k,URHO)) );
#endif

#if AMREX_SPACEDIM==2
        vrt(i,j,k,dcomp)   = dvdx - dudy;
#elif AMREX_SPACEDIM==3
        vrt(i,j,k,dcomp)   = dwdy - dvdz;
        vrt(i,j,k,dcomp+1) = dudz - dwdx;
        vrt(i,j,k,dcomp+2) = dvdx - dudy;
#endif

    });
}

void cns_deruderiv (const Box& bx, FArrayBox& uderfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat = datfab.array();
    auto       uder = uderfab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        uderfab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate dudx, dudy, dudz
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

        uder(i,j,k,dcomp) = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMX)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMX)/dat(im,j,k,URHO)) );
        uder(i,j,k,dcomp+1) = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMX)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMX)/dat(i,jm,k,URHO)) );
#if AMREX_SPACEDIM==3
        uder(i,j,k,dcomp+2) = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMX)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMX)/dat(i,j,km,URHO)) );
#endif
    });
}

void cns_dervderiv (const Box& bx, FArrayBox& vderfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat = datfab.array();
    auto       vder = vderfab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        vderfab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate dvdx, dvdy, dvdz
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

        vder(i,j,k,dcomp) = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMY)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMY)/dat(im,j,k,URHO)) );
        vder(i,j,k,dcomp+1) = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMY)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMY)/dat(i,jm,k,URHO)) );
#if AMREX_SPACEDIM==3
        vder(i,j,k,dcomp+2) = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMY)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMY)/dat(i,j,km,URHO)) );
#endif
    });
}

#if AMREX_SPACEDIM==3
void cns_derwderiv (const Box& bx, FArrayBox& wderfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat = datfab.array();
    auto       wder = wderfab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        wderfab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate dvdx, dvdy, dvdz
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

        wder(i,j,k,dcomp) = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMZ)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMZ)/dat(im,j,k,URHO)) );
        wder(i,j,k,dcomp+1) = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMZ)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMZ)/dat(i,jm,k,URHO)) );
        wder(i,j,k,dcomp+2) = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMZ)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMZ)/dat(i,j,km,URHO)) );
    });
}
#endif

void cns_dermu (const Box& bx, FArrayBox& mufab, int dcomp, int /*ncomp*/,
                  const FArrayBox& tempfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const temp = tempfab.array();
    auto       mu   = mufab.array();
    Parm const* parm = CNS::d_parm;

    const auto& flag_fab = amrex::getEBCellFlagFab(tempfab);
    const auto& typ = flag_fab.getType(bx);

    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;
    
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real mutmp = 0.0;

        if(parm->is_visc == true){
            if(parm->is_const_visc == true) mutmp = parm->const_visc_mu;
            else{
               bool cov = flags(i,j,k).isCovered();
               mutmp = cov ? -1.e10 : parm->kappa_0 * parm->Pr * std::pow(temp(i,j,k), Real(0.7)); 
            } 
        }

        mu(i,j,k,dcomp) = mutmp;
    });
}

void cns_derdivu (const Box& bx, FArrayBox& divufab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat = datfab.array();
    auto       divu = divufab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        divufab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate div(u)
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

        AMREX_D_TERM(
      const amrex::Real uhi = dat(ip, j, k, UMX) / dat(ip, j, k, URHO);
      const amrex::Real ulo = dat(im, j, k, UMX) / dat(im, j, k, URHO);
      , const amrex::Real vhi = dat(i, jp, k, UMY) / dat(i, jp, k, URHO);
      const amrex::Real vlo = dat(i, jm, k, UMY) / dat(i, jm, k, URHO);
      , const amrex::Real whi = dat(i, j, kp, UMZ) / dat(i, j, kp, URHO);
      const amrex::Real wlo = dat(i, j, km, UMZ) / dat(i, j, km, URHO););

        divu(i, j, k, dcomp) = AMREX_D_TERM(
        wi * (uhi - ulo) * dxinv[0],
        +wj * (vhi - vlo) * dxinv[1], 
        +wk * (whi - wlo) * dxinv[2]);
    });
}

void cns_dertau (const Box& bx, FArrayBox& taufab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat = datfab.array();
    auto       tau = taufab.array();
    const auto dxinv = geomdata.InvCellSizeArray();
    Parm const* parm = CNS::d_parm;

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        taufab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate tau
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {

        // Get mu, coefficient of dynamic viscosity
        Real mu = 0.0;
        if(parm->is_visc == true){
            if(parm->is_const_visc == true) mu = parm->const_visc_mu;
            else{
               bool cov = flags(i,j,k).isCovered();
               mu = cov ? -1.e10 : parm->kappa_0 * parm->Pr * std::pow(dat(i,j,k,UTEMP), Real(0.7)); 
            } 
        }
        AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 ,   get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 ,   get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 ,   const amrex::Real wj = get_weight(jm, jp);
                 ,   const amrex::Real wk = get_weight(km, kp);)

        // Get velocity derivatives in x-direction
        AMREX_D_TERM(Real dudx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMX)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMX)/dat(im,j,k,URHO)) );
                ,    Real dvdx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMY)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMY)/dat(im,j,k,URHO)) );
                ,    Real dwdx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMZ)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMZ)/dat(im,j,k,URHO)) ); )

        // Get velocity derivatives in y-direction
        AMREX_D_TERM(Real dudy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMX)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMX)/dat(i,jm,k,URHO)) );
                ,    Real dvdy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMY)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMY)/dat(i,jm,k,URHO)) );
                ,    Real dwdy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMZ)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMZ)/dat(i,jm,k,URHO)) ); )

#if AMREX_SPACEDIM==3
        // Get velocity derivatives in z-direction
        AMREX_D_TERM(Real dudz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMX)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMX)/dat(i,j,km,URHO)) );
                ,    Real dvdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMY)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMY)/dat(i,j,km,URHO)) );
                ,    Real dwdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMZ)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMZ)/dat(i,j,km,URHO)) ); )
#endif

        Real divu = AMREX_D_TERM(dudx, +dvdy, +dwdz);

        tau(i,j,k,dcomp)   =  mu * (2.0 * dudx - (2.0/3.0)*divu);
        tau(i,j,k,dcomp+1) =  mu * (2.0 * dvdy - (2.0/3.0)*divu);
#if AMREX_SPACEDIM==2
        tau(i,j,k,dcomp+2) =  mu * (dvdx + dudy);
#endif

#if AMREX_SPACEDIM==3
        tau(i,j,k,dcomp+2) =  mu * (2.0 * dwdz - (2.0/3.0)*divu);
        tau(i,j,k,dcomp+3) =  mu * (dvdx + dudy);
        tau(i,j,k,dcomp+4) =  mu * (dudz + dwdx);
        tau(i,j,k,dcomp+5) =  mu * (dvdz + dwdy);
#endif

    });
}

#if AMREX_SPACEDIM==3

void cns_deromdelu (const Box& bx, FArrayBox& wdufab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int level)
{

    auto const dat = datfab.array();
    auto       wdelu = wdufab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        wdufab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate vorticity
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        int im; int ip; int jm; int jp; int km; int kp;

        // if fab is all regular -> call regular idx and weights
        // otherwise
        get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
        get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
        get_idx(k, 2, all_regular, flags(i, j, k), km, kp);

        const amrex::Real wi = get_weight(im, ip);
        const amrex::Real wj = get_weight(jm, jp);
        const amrex::Real wk = get_weight(km, kp);

        // Get velocity derivatives in x-direction
        AMREX_D_TERM(Real dudx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMX)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMX)/dat(im,j,k,URHO)) );
                ,    Real dvdx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMY)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMY)/dat(im,j,k,URHO)) );
                ,    Real dwdx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMZ)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMZ)/dat(im,j,k,URHO)) ); )

        // Get velocity derivatives in y-direction
        AMREX_D_TERM(Real dudy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMX)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMX)/dat(i,jm,k,URHO)) );
                ,    Real dvdy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMY)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMY)/dat(i,jm,k,URHO)) );
                ,    Real dwdy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMZ)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMZ)/dat(i,jm,k,URHO)) ); )

#if AMREX_SPACEDIM==3
        // Get velocity derivatives in z-direction
        AMREX_D_TERM(Real dudz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMX)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMX)/dat(i,j,km,URHO)) );
                ,    Real dvdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMY)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMY)/dat(i,j,km,URHO)) );
                ,    Real dwdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMZ)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMZ)/dat(i,j,km,URHO)) ); )
#endif
        Real omx = dwdy - dvdz;
        Real omy = dudz - dwdx;
        Real omz = dvdx - dudy;

        wdelu(i,j,k,dcomp)   = omx*dudx + omy*dudy + omz*dudz;
        wdelu(i,j,k,dcomp+1) = omx*dvdx + omy*dvdy + omz*dvdz;
        wdelu(i,j,k,dcomp+2) = omx*dwdx + omy*dwdy + omz*dwdz;
    });
}

#endif

void cns_deromdivu (const Box& bx, FArrayBox& wdivufab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int level)
{

    auto const dat = datfab.array();
    auto       wdivu = wdivufab.array();
    const auto dxinv = geomdata.InvCellSizeArray();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        wdivufab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate vorticity
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;
        , int jm; int jp;
        , int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                    , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                    , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)

        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                    , const amrex::Real wj = get_weight(jm, jp);
                    , const amrex::Real wk = get_weight(km, kp);)

        // Get velocity derivatives in x-direction
        AMREX_D_TERM(Real dudx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMX)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMX)/dat(im,j,k,URHO)) );
                ,    Real dvdx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMY)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMY)/dat(im,j,k,URHO)) );
                ,    Real dwdx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMZ)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMZ)/dat(im,j,k,URHO)) ); )

        // Get velocity derivatives in y-direction
        AMREX_D_TERM(Real dudy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMX)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMX)/dat(i,jm,k,URHO)) );
                ,    Real dvdy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMY)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMY)/dat(i,jm,k,URHO)) );
                ,    Real dwdy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMZ)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMZ)/dat(i,jm,k,URHO)) ); )

#if AMREX_SPACEDIM==3
        // Get velocity derivatives in z-direction
        AMREX_D_TERM(Real dudz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMX)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMX)/dat(i,j,km,URHO)) );
                ,    Real dvdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMY)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMY)/dat(i,j,km,URHO)) );
                ,    Real dwdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMZ)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMZ)/dat(i,j,km,URHO)) ); )
#endif
        Real omz = dvdx - dudy;

#if AMREX_SPACEDIM==2
        wdivu(i,j,k,dcomp)    = omz*(dudx + dvdy);
#endif

#if AMREX_SPACEDIM==3
        Real omx = dwdy - dvdz;
        Real omy = dudz - dwdx;
        wdivu(i,j,k,dcomp)    = omx*(dudx + dvdy + dwdz);
        wdivu(i,j,k,dcomp+1)  = omy*(dudx + dvdy + dwdz);
        wdivu(i,j,k,dcomp+2)  = omz*(dudx + dvdy + dwdz);
#endif

    });
}

void cns_derbaroclinic (const Box& bx, FArrayBox& barofab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int level)
{

    auto const dat = datfab.array();
    auto       baro = barofab.array();
    const auto dxinv = geomdata.InvCellSizeArray();
    Parm const* parm = CNS::d_parm;

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        barofab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;

    // Calculate vorticity
    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
    {
        AMREX_D_TERM(int im; int ip;
        , int jm; int jp;
        , int km; int kp;)

        // if fab is all regular -> call regular idx and weights
        // otherwise
        AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                    , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                    , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)

        AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                    , const amrex::Real wj = get_weight(jm, jp);
                    , const amrex::Real wk = get_weight(km, kp);)

        // Get density derivatives
        AMREX_D_TERM(Real drdx = wi * dxinv[0] 
                          * ( dat(ip,j,k,URHO) - dat(im,j,k,URHO) );
                ,    Real drdy = wj * dxinv[1] 
                          * ( dat(i,jp,k,URHO) - dat(i,jp,k,URHO) );
                ,    Real drdz = wk * dxinv[2] 
                          * ( dat(i,j,kp,URHO) - dat(i,j,km,URHO) ); )

        // Get pressure derivatives
        AMREX_D_TERM(Real dpdx = wi * dxinv[0] * (parm->eos_gamma-1.) 
                          * ( dat(ip,j,k,UEINT) - dat(im,j,k,UEINT) );
                ,    Real dpdy = wj * dxinv[1] * (parm->eos_gamma-1.) 
                          * ( dat(i,jp,k,UEINT) - dat(i,jm,k,UEINT) );
                ,    Real dpdz = wk * dxinv[2] * (parm->eos_gamma-1.) 
                          * ( dat(i,j,kp,UEINT) - dat(i,j,km,UEINT) ); )

#if AMREX_SPACEDIM==3
        // Get velocity derivatives in z-direction
        AMREX_D_TERM(Real dudz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMX)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMX)/dat(i,j,km,URHO)) );
                ,    Real dvdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMY)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMY)/dat(i,j,km,URHO)) );
                ,    Real dwdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMZ)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMZ)/dat(i,j,km,URHO)) ); )
#endif
        
#if AMREX_SPACEDIM==2
        // only one components (z-direction)
        baro(i,j,k,dcomp)    = (drdx*dpdy - drdy*dpdx) / (dat(i,j,k,URHO)*dat(i,j,k,URHO));
#endif

#if AMREX_SPACEDIM==3
        baro(i,j,k,dcomp)    = (drdy*dpdz - drdz*dpdy) / (dat(i,j,k,URHO)*dat(i,j,k,URHO));
        baro(i,j,k,dcomp+1)  = -(drdx*dpdz - drdz*dpdx) / (dat(i,j,k,URHO)*dat(i,j,k,URHO));
        baro(i,j,k,dcomp+2)  = (drdx*dpdy - drdy*dpdx) / (dat(i,j,k,URHO) * dat(i,j,k,URHO));
#endif
        
    });
}

void cns_deromdiff (const Box& bx, FArrayBox& omdfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& geomdata,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{

    auto const dat = datfab.array();
    auto       omd = omdfab.array();
    const auto dxinv = geomdata.InvCellSizeArray();
    Parm const* parm = CNS::d_parm;

    const amrex::Box& bxg2 = amrex::grow(bx, 2);
    const amrex::Box& bxg1 = amrex::grow(bx, 1);

    // For 2D, tau(0) = tauxx, tau(1) = tauyy, tau(2) = tauxy

    // For 3D, tau(0) = tauxx, tau(1) = tauyy, tau(2) = tauzz
    // tau(3) = tauxy, tau(4) = tauxz, tau(5) = tauyz
    const int TXX = 0;
    const int TYY = 1;

#if AMREX_SPACEDIM==2
    const int TXY = 2;
#elif AMREX_SPACEDIM==3
    const int TZZ = 2;
    const int TXY = 3;
    const int TXZ = 4;
    const int TYZ = 5;
#endif

    // Need the shear stress (tau) in the domain + 2 ghost cells
    // to obtain its divergence in the domain + 1 ghost cell
    amrex::FArrayBox omdloc(bxg1, AMREX_SPACEDIM);

    amrex::Elixir local_eli_omd = omdloc.elixir();
    auto omdarr = omdloc.array();

    const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
    const auto& typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
        omdfab.setVal<amrex::RunOn::Device>(0.0, bx);
        return;
    }
    const auto& flags = flag_fab.const_array();
    const bool all_regular = typ == amrex::FabType::regular;


    {
        // Local definition of tau to compute componentsa of omdloc
#if AMREX_SPACEDIM==2
        amrex::FArrayBox tauloc(bxg2, 3);
#elif AMREX_SPACEDIM==3
        amrex::FArrayBox tauloc(bxg2, 6);
#endif
        amrex::Elixir local_eli_tau = tauloc.elixir();
        auto tauarr = tauloc.array();

        // Calculate tau on bxg2
        amrex::ParallelFor(bxg2, 
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
        {

            // Get mu, coefficient of dynamic viscosity
            Real mu = 0.0;
            if(parm->is_visc == true){
                if(parm->is_const_visc == true) mu = parm->const_visc_mu;
                else{
                    bool cov = flags(i,j,k).isCovered();
                    mu = cov ? -1.e10 : parm->kappa_0 * parm->Pr * std::pow(dat(i,j,k,QTEMP), Real(0.7)); 
                } 
            }
            AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

            // if fab is all regular -> call regular idx and weights
            // otherwise
            AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                    ,   get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                    ,   get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
            AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                    ,   const amrex::Real wj = get_weight(jm, jp);
                    ,   const amrex::Real wk = get_weight(km, kp);)

            // Get velocity derivatives in x-direction
            AMREX_D_TERM(Real dudx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMX)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMX)/dat(im,j,k,URHO)) );
                    ,    Real dvdx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMY)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMY)/dat(im,j,k,URHO)) );
                    ,    Real dwdx = wi * dxinv[0] 
                          * ( (dat(ip,j,k,UMZ)/dat(ip,j,k,URHO)) - (dat(im,j,k,UMZ)/dat(im,j,k,URHO)) ); )

            // Get velocity derivatives in y-direction
            AMREX_D_TERM(Real dudy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMX)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMX)/dat(i,jm,k,URHO)) );
                    ,    Real dvdy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMY)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMY)/dat(i,jm,k,URHO)) );
                    ,    Real dwdy = wj * dxinv[1] 
                          * ( (dat(i,jp,k,UMZ)/dat(i,jp,k,URHO)) - (dat(i,jm,k,UMZ)/dat(i,jm,k,URHO)) ); )

#if AMREX_SPACEDIM==3
            // Get velocity derivatives in z-direction
            AMREX_D_TERM(Real dudz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMX)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMX)/dat(i,j,km,URHO)) );
                    ,    Real dvdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMY)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMY)/dat(i,j,km,URHO)) );
                    ,    Real dwdz = wk * dxinv[2] 
                          * ( (dat(i,j,kp,UMZ)/dat(i,j,kp,URHO)) - (dat(i,j,km,UMZ)/dat(i,j,km,URHO)) ); )
#endif

            Real divu = AMREX_D_TERM(dudx, +dvdy, +dwdz);

            
            tauarr(i,j,k,TXX) =  mu * (2.0 * dudx - (2.0/3.0)*divu);
            tauarr(i,j,k,TYY) =  mu * (2.0 * dvdy - (2.0/3.0)*divu);
            tauarr(i,j,k,TXY) =  mu * (dvdx + dudy);
       
#if AMREX_SPACEDIM==3
            tauarr(i,j,k,TZZ) =  mu * (2.0 * dwdz - (2.0/3.0)*divu);
            tauarr(i,j,k,TXZ) =  mu * (dudz + dwdx);
            tauarr(i,j,k,TYZ) =  mu * (dvdz + dwdy);
#endif
        });

        // Compute the divergence of tau on valid domain + 1 ghost cell
        amrex::ParallelFor(bxg1, 
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
        {

            AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

            Real rhoinv = 1.0 / dat(i,j,k,URHO);

            // if fab is all regular -> call regular idx and weights
            // otherwise
            AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                    ,   get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                    ,   get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
            AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                    ,   const amrex::Real wj = get_weight(jm, jp);
                    ,   const amrex::Real wk = get_weight(km, kp);)

            // Get divergence of tau components

            omdarr(i,j,k,0) = AMREX_D_TERM ( wi * dxinv[0] * rhoinv * ( tauarr(ip,j,k,TXX) - tauarr(im,j,k,TXX) ),
                            + wj * dxinv[1] * rhoinv * ( tauarr(i,jp,k,TXY) - tauarr(i,jm,k,TXY) ),
                            + wk * dxinv[2] * rhoinv * ( tauarr(i,j,kp,TXZ) - tauarr(i,j,km,TXZ) ) ); 

            omdarr(i,j,k,1) = AMREX_D_TERM ( wi * dxinv[0] * rhoinv * ( tauarr(ip,j,k,TXY) - tauarr(im,j,k,TXY) ),
                            + wj * dxinv[1] * rhoinv * ( tauarr(i,jp,k,TYY) - tauarr(i,jm,k,TYY) ),
                            + wk * dxinv[2] * rhoinv * ( tauarr(i,j,kp,TYZ) - tauarr(i,j,km,TYZ) ) ); 
#if AMREX_SPACEDIM==3
            omdarr(i,j,k,2) = wi * dxinv[0] * rhoinv * ( tauarr(ip,j,k,TXZ) - tauarr(im,j,k,TXZ) )
                            + wj * dxinv[1] * rhoinv * ( tauarr(i,jp,k,TYZ) - tauarr(i,jm,k,TYZ) )
                            + wk * dxinv[2] * rhoinv * ( tauarr(i,j,kp,TZZ) - tauarr(i,j,km,TZZ) );
#endif

        });

        // Obtain the contribution of viscous diffusion to the vorticity budget
        amrex::ParallelFor(bx, 
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
        {

            AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

            // if fab is all regular -> call regular idx and weights
            // otherwise
            AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                    ,    get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                    ,    get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
            AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                    ,    const amrex::Real wj = get_weight(jm, jp);
                    ,    const amrex::Real wk = get_weight(km, kp);)

#if AMREX_SPACEDIM==2
            // Get curl of omdarr
            // Only one component ( omdiffz )
            omd(i,j,k,dcomp) = wi * dxinv[0] * ( omdarr(ip,j,k,1) - omdarr(im,j,k,1) )
                             - wj * dxinv[1] * ( omdarr(i,jp,k,0) - omdarr(i,jm,k,0) );

#elif AMREX_SPACEDIM==3
            omd(i,j,k,dcomp) = wj * dxinv[1] * ( omdarr(i,jp,k,2) - omdarr(i,jm,k,2) )
                             - wk * dxinv[2] * ( omdarr(i,j,kp,1) - omdarr(i,j,km,1) );

            omd(i,j,k,dcomp+1) = - ( wi * dxinv[0] * ( omdarr(ip,j,k,2) - omdarr(im,j,k,2) )
                                   - wk * dxinv[2] * ( omdarr(i,j,kp,0) - omdarr(i,j,km,0) ) );

            omd(i,j,k,dcomp+2) = wi * dxinv[0] * ( omdarr(ip,j,k,1) - omdarr(im,j,k,1) )
                               - wj * dxinv[1] * ( omdarr(i,jp,k,0) - omdarr(i,jm,k,0) );
#endif

        });
    }
}
