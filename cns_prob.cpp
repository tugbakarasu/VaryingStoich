
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "CNS_parm.H"
#include "cns_prob_parm.H"
#include "CNS.H"

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
        amrex::ParmParse pp("prob");

        // Read in low pressure (usually atmospheric pressure in Pa) (pressure of quiescent nonshocked gas)
        pp.query("p0", CNS::h_prob_parm->p0);
        // Temperature of quiscent nonshocked gas
        pp.query("T0", CNS::h_prob_parm->T0);
        // Get the half reaction distance xd in SI units (metres)
        pp.query("xd", CNS::h_prob_parm->xd);

        // Get the threshold of gradient of mass fraction for tagging (in metres) (this MUST be in inputs file)
        pp.get("deltaY", CNS::h_prob_parm->deltaY);
        pp.get("deltaT", CNS::h_prob_parm->deltaT);
        pp.get("deltaP", CNS::h_prob_parm->deltaP);
        pp.query("overdrive_factor", CNS::h_prob_parm->od_factor);
        pp.query("Mobj", CNS::h_prob_parm->Mobj);


        // RADIUS OF HIGH TEMPERATURE BURNT GAS (IN SI UNITS)
        pp.query("radhitemp", CNS::h_prob_parm->radhitemp);

        // INITIAL CONDITION FOR SHOCK AND FLAME LOCATION (IN METRES)
        CNS::h_prob_parm->flameloc = 0.0;
        CNS::h_prob_parm->shloc    = 0.0;


        pp.get("append_file", CNS::h_prob_parm->append_file);
        pp.get("data_file", CNS::h_prob_parm->data_file);
        pp.get("write_to_file", CNS::h_prob_parm->write_to_file);

        pp.get("xreflo", CNS::h_prob_parm->xreflo);
        pp.get("xrefhi", CNS::h_prob_parm->xrefhi);

        pp.get("yreflo", CNS::h_prob_parm->yreflo);
        pp.get("yrefhi", CNS::h_prob_parm->yrefhi);

        pp.query("nzones", CNS::h_prob_parm->nzones);

        pp.query("refuptolev", CNS::h_prob_parm->refuptolev);
        
        // Variable Stoichiometry stuff
        pp.query("OF_st", CNS::h_prob_parm->OF_st);
        
        pp.query("rich_rhot", CNS::h_prob_parm->rich_rhot);
        pp.query("rich_Yf", CNS::h_prob_parm->rich_Yf);
        pp.query("rich_Yox", CNS::h_prob_parm->rich_Yox);
        pp.query("rich_Yp", CNS::h_prob_parm->rich_Yp);
        
        pp.query("lean_rhot", CNS::h_prob_parm->lean_rhot);
        pp.query("lean_Yf", CNS::h_prob_parm->lean_Yf);
        pp.query("lean_Yox", CNS::h_prob_parm->lean_Yox);
        pp.query("lean_Yp", CNS::h_prob_parm->lean_Yp);

#ifdef AMREX_USE_GPU
        // Cannot use Gpu::copy because ProbParm is not trivailly copyable.
        Gpu::htod_memcpy_async(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(CNS::d_prob_parm, CNS::h_prob_parm, sizeof(ProbParm));
#endif

        if(CNS::h_prob_parm->nzones > 0){
            pp.get("append_file_zones", CNS::h_prob_parm->append_file_zones);
            pp.get("data_file_zonebase", CNS::h_prob_parm->data_file_zonebase);
            pp.get("write_to_file_zones", CNS::h_prob_parm->write_to_file_zones);

            const int numzones = CNS::h_prob_parm->nzones;
            
            CNS::h_prob_parm->flamel 
                = (amrex::Real*)The_Arena()->alloc(sizeof(Real)*numzones);
            CNS::h_prob_parm->shl 
                = (amrex::Real*)The_Arena()->alloc(sizeof(Real)*numzones);
            CNS::h_prob_parm->yloz 
                = (amrex::Real*)The_Arena()->alloc(sizeof(Real)*numzones);
            CNS::h_prob_parm->yhiz 
                = (amrex::Real*)The_Arena()->alloc(sizeof(Real)*numzones);

            Gpu::HostVector<Real> flamel(CNS::h_prob_parm->nzones);
            Gpu::HostVector<Real> shl(CNS::h_prob_parm->nzones);
            Gpu::HostVector<Real> yloz(CNS::h_prob_parm->nzones);
            Gpu::HostVector<Real> yhiz(CNS::h_prob_parm->nzones);

            Vector<amrex::Real> yloa(CNS::h_prob_parm->nzones), yhia(CNS::h_prob_parm->nzones);
            pp.getarr("yloz", yloa, 0, CNS::h_prob_parm->nzones);
            pp.getarr("yhiz", yhia, 0, CNS::h_prob_parm->nzones);

            for(int ii = 0; ii < CNS::h_prob_parm->nzones; ++ii){
                yloz[ii] = yloa[ii]; yhiz[ii] = yhia[ii];
                flamel[ii] = 0.0; shl[ii] = 0.0;
                // Print() << "ii = " << ii << ", flamel = " << flamel[ii] << ", shl = " << shl[ii]
                //         << ", yloz = " << yloz[ii] << ", yhiz = " << yhiz[ii] << "\n";
            }

            Gpu::copyAsync(Gpu::hostToDevice, flamel.data(), flamel.data() + CNS::h_prob_parm->nzones,
                       CNS::h_prob_parm->flamel);
            Gpu::copyAsync(Gpu::hostToDevice, shl.data(), shl.data() + CNS::h_prob_parm->nzones,
                       CNS::h_prob_parm->shl);
            Gpu::copyAsync(Gpu::hostToDevice, yloz.data(), yloz.data() + CNS::h_prob_parm->nzones,
                       CNS::h_prob_parm->yloz);
            Gpu::copyAsync(Gpu::hostToDevice, yhiz.data(), yhiz.data() + CNS::h_prob_parm->nzones,
                       CNS::h_prob_parm->yhiz);
            Gpu::streamSynchronize();
        }

    }
}
