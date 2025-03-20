
#include <CNS.H>
#include <CNS_derive.H>

#define NSCBCOUT 1002
#define NSCBCIN  1001

using namespace amrex;

int CNS::num_state_data_types = 0;
Parm* CNS::h_parm = nullptr;
Parm* CNS::d_parm = nullptr;
ProbParm* CNS::h_prob_parm = nullptr;
ProbParm* CNS::d_prob_parm = nullptr;

static Box the_same_box (const Box& b) { return b; }
static Box grow_box_by_one (const Box& b) { return amrex::grow(b,1); }
static Box grow_box_by_two (const Box& b) { return amrex::grow(b,2); }
static Box grow_box_by_three (const Box& b) { return amrex::grow(b,3); }

using BndryFunc = StateDescriptor::BndryFunc;

//
// Components are:
//  Interior, Inflow, Outflow,  Symmetry,     SlipWall,     NoSlipWall
//
static int scalar_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_even, NSCBCOUT, NSCBCIN
};

static int norm_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_odd,  BCType::reflect_odd,  BCType::reflect_odd, NSCBCOUT, NSCBCIN
};

static int tang_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_odd, NSCBCOUT, NSCBCIN
};

static
void
set_scalar_bc (BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        // Print() << "i = " << i << ", lo_bc[i] = " << lo_bc[i] << ", hi_bc[i] = " << hi_bc[i] 
        //         << ", scalar_bc = " << scalar_bc[lo_bc[i]] << "\n";
        bc.setLo(i,scalar_bc[lo_bc[i]]);
        bc.setHi(i,scalar_bc[hi_bc[i]]);
    }
}

static
void
set_x_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,norm_vel_bc[lo_bc[0]]);
    bc.setHi(0,norm_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

static
void
set_y_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,norm_vel_bc[lo_bc[1]]);
    bc.setHi(1,norm_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

#if (AMREX_SPACEDIM == 3)
static
void
set_z_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);
    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);
    bc.setLo(2,norm_vel_bc[lo_bc[2]]);
    bc.setHi(2,norm_vel_bc[hi_bc[2]]);
}
#endif

void
CNS::variableSetUp ()
{
    h_parm = new Parm{}; // This is deleted in CNS::variableCleanUp().
    h_prob_parm = new ProbParm{};
    d_parm = (Parm*)The_Arena()->alloc(sizeof(Parm));
    d_prob_parm = (ProbParm*)The_Arena()->alloc(sizeof(ProbParm));

    read_params();

    bool state_data_extrap = false;
    bool store_in_checkpoint = true;
    desc_lst.addDescriptor(State_Type,IndexType::TheCellType(),
                           StateDescriptor::Point,NUM_GROW,NUM_STATE,
                           &eb_mf_cell_cons_interp,state_data_extrap,store_in_checkpoint);

    Vector<BCRec>       bcs(NUM_STATE);
    Vector<std::string> name(NUM_STATE);
    BCRec bc;
    int cnt = 0;
    set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "density";
    cnt++; set_x_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "xmom";
    cnt++; set_y_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "ymom";
#if (AMREX_SPACEDIM == 3)
    cnt++; set_z_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "zmom";
#endif
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "rho_E";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "rhoY_Fuel";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "rhoY_Oxid";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "rhoY_Prod";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "rho_e";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "Temp";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "sootfoil";

    StateDescriptor::BndryFunc bndryfunc(cns_bcfill);
    bndryfunc.setRunOnGPU(true);

    desc_lst.setComponent(State_Type,
                          URHO,
                          name,
                          bcs,
                          bndryfunc);

    desc_lst.addDescriptor(Cost_Type, IndexType::TheCellType(), StateDescriptor::Point,
                           0,1, &pc_interp);
    desc_lst.setComponent(Cost_Type, 0, "Cost", bc, bndryfunc);

    num_state_data_types = desc_lst.size();

    StateDescriptor::setBndryFuncThreadSafety(true);

    // DEFINE DERIVED QUANTITIES

    // Pressure
    derive_lst.add("pressure",IndexType::TheCellType(),1,
                   cns_derpres,the_same_box);
    derive_lst.addComponent("pressure",desc_lst,State_Type,UEINT,1);

    // Velocities
    derive_lst.add("x_velocity",IndexType::TheCellType(),1,
                   cns_dervel,the_same_box);
    derive_lst.addComponent("x_velocity",desc_lst,State_Type,URHO,1);
    derive_lst.addComponent("x_velocity",desc_lst,State_Type,UMX,1);

    derive_lst.add("y_velocity",IndexType::TheCellType(),1,
                   cns_dervel,the_same_box);
    derive_lst.addComponent("y_velocity",desc_lst,State_Type,URHO,1);
    derive_lst.addComponent("y_velocity",desc_lst,State_Type,UMY,1);

#if (AMREX_SPACEDIM == 3)
    derive_lst.add("z_velocity",IndexType::TheCellType(),1,
                   cns_dervel,the_same_box);
    derive_lst.addComponent("z_velocity",desc_lst,State_Type,URHO,1);
    derive_lst.addComponent("z_velocity",desc_lst,State_Type,UMZ,1);
#endif

    // Fuel mass fraction
    derive_lst.add("yF",IndexType::TheCellType(),1,
                   cns_dervel,the_same_box);
    derive_lst.addComponent("yF",desc_lst,State_Type,URHO,1);
    derive_lst.addComponent("YF",desc_lst,State_Type,URHOY_F,1);
    
    // Oxidizer mass fraction
    derive_lst.add("yO",IndexType::TheCellType(),1,
                   cns_dervel,the_same_box);
    derive_lst.addComponent("yO",desc_lst,State_Type,URHO,1);
    derive_lst.addComponent("YO",desc_lst,State_Type,URHOY_O,1);
    
    // Product mass fraction
    derive_lst.add("yP",IndexType::TheCellType(),1,
                   cns_dervel,the_same_box);
    derive_lst.addComponent("yP",desc_lst,State_Type,URHO,1);
    derive_lst.addComponent("YP",desc_lst,State_Type,URHOY_P,1);
	
    // Numerical schlieren
    derive_lst.add("schlieren",IndexType::TheCellType(),1,
                   cns_derschlieren,grow_box_by_one);
    derive_lst.addComponent("schlieren",desc_lst,State_Type,URHO,1);

    derive_lst.add("shadowgraph",IndexType::TheCellType(),1,
                   cns_dershadowgraph,grow_box_by_one);
    derive_lst.addComponent("shadowgraph",desc_lst,State_Type,URHO,1);

    // Mach number
    derive_lst.add("mach",IndexType::TheCellType(),1,
                   cns_dermach,the_same_box);
    derive_lst.addComponent("mach",desc_lst,State_Type,URHO,1);
    derive_lst.addComponent("mach",desc_lst,State_Type,UMX,1);
    derive_lst.addComponent("mach",desc_lst,State_Type,UMY,1);
#if AMREX_SPACEDIM==3
    derive_lst.addComponent("mach",desc_lst,State_Type,UMZ,1);
#endif
    derive_lst.addComponent("mach",desc_lst,State_Type,UTEMP,1);

    // Gradient of fuel mass fraction vector (to track flame location)
    amrex::Vector<std::string> var_names_gy(AMREX_SPACEDIM);
    var_names_gy[0]="dYFdx";
    var_names_gy[1]="dYFdy";
#if AMREX_SPACEDIM==3
    var_names_gy[2]="dYFdz";
#endif
    derive_lst.add("dYF",IndexType::TheCellType(), AMREX_SPACEDIM,
                   var_names_gy, cns_derYFderiv, grow_box_by_one);
    derive_lst.addComponent("dYF",desc_lst,State_Type,URHO,NUM_STATE);
    
    // Gradient of oxidizer mass fraction vector (to track flame location)
    amrex::Vector<std::string> var_names_go(AMREX_SPACEDIM);
    var_names_go[0]="dYOdx";
    var_names_go[1]="dYOdy";
#if AMREX_SPACEDIM==3
    var_names_go[2]="dYOdz";
#endif
    derive_lst.add("dYO",IndexType::TheCellType(), AMREX_SPACEDIM,
                   var_names_go, cns_derYOderiv, grow_box_by_one);
    derive_lst.addComponent("dYO",desc_lst,State_Type,URHO,NUM_STATE);
    
    // Gradient of Product mass fraction vector (to track flame location)
    amrex::Vector<std::string> var_names_gp(AMREX_SPACEDIM);
    var_names_gp[0]="dYPdx";
    var_names_gp[1]="dYPdy";
#if AMREX_SPACEDIM==3
    var_names_gp[2]="dYPdz";
#endif
    derive_lst.add("dYP",IndexType::TheCellType(), AMREX_SPACEDIM,
                   var_names_gp, cns_derYPderiv, grow_box_by_one);
    derive_lst.addComponent("dYP",desc_lst,State_Type,URHO,NUM_STATE);

    // Gradient of mass density (vector)
    amrex::Vector<std::string> var_names_grho(AMREX_SPACEDIM);
    var_names_grho[0]="drhodx";
    var_names_grho[1]="drhody";
#if AMREX_SPACEDIM==3
    var_names_grho[2]="drhodz";
#endif

    derive_lst.add("drho",IndexType::TheCellType(), AMREX_SPACEDIM,
                   var_names_grho, cns_derrhoderiv, grow_box_by_one);
    derive_lst.addComponent("drho",desc_lst,State_Type,URHO,NUM_STATE);

    // Gradient of pressure (vector)
    amrex::Vector<std::string> var_names_gpre(AMREX_SPACEDIM);
    var_names_gpre[0]="dpredx";
    var_names_gpre[1]="dpredy";
#if AMREX_SPACEDIM==3
    var_names_gpre[2]="dpredz";
#endif

    derive_lst.add("dpre",IndexType::TheCellType(), AMREX_SPACEDIM,
                   var_names_gpre, cns_derprederiv, grow_box_by_one);
    derive_lst.addComponent("dpre",desc_lst,State_Type,URHO,NUM_STATE);

    // Vorticity vector definition
#if AMREX_SPACEDIM==2
    amrex::Vector<std::string> var_names_vort(1);
    var_names_vort[0]="vort_z";

    derive_lst.add("vort",IndexType::TheCellType(), 1,
                   var_names_vort, cns_dervort, grow_box_by_one);
    derive_lst.addComponent("vort",desc_lst,State_Type,URHO,NUM_STATE);
#elif AMREX_SPACEDIM==3
    amrex::Vector<std::string> var_names_vort(3);
    var_names_vort[0]="vort_x";
    var_names_vort[1]="vort_y";
    var_names_vort[2]="vort_z";

    derive_lst.add("vort",IndexType::TheCellType(), 3,
                   var_names_vort, cns_dervort, grow_box_by_one);
    derive_lst.addComponent("vort",desc_lst,State_Type,URHO,NUM_STATE);
#endif

    // velocity derivatives (dudx, dudy, dudz)
    amrex::Vector<std::string> var_names_uderiv(AMREX_SPACEDIM);
    AMREX_D_TERM(var_names_uderiv[0]="dudx";
                 , var_names_uderiv[1]="dudy";
                 , var_names_uderiv[2]="dudz";)

    // Get velocity derivatives
    derive_lst.add("uderiv",IndexType::TheCellType(),AMREX_SPACEDIM,
                   var_names_uderiv, cns_deruderiv,grow_box_by_one);
    derive_lst.addComponent("uderiv",desc_lst,State_Type,URHO,NUM_STATE);

    // velocity derivatives (dvdx, dvdy, dvdz)
    amrex::Vector<std::string> var_names_vderiv(AMREX_SPACEDIM);
    AMREX_D_TERM(  var_names_vderiv[0]="dvdx";
                 , var_names_vderiv[1]="dvdy";
                 , var_names_vderiv[2]="dvdz";)

    // Get velocity derivatives
    derive_lst.add("vderiv",IndexType::TheCellType(),AMREX_SPACEDIM,
                   var_names_vderiv, cns_dervderiv,grow_box_by_one);
    derive_lst.addComponent("vderiv",desc_lst,State_Type,URHO,NUM_STATE);

#if AMREX_SPACEDIM==3
    // velocity derivatives (dwdx, dwdy, dwdz)
    amrex::Vector<std::string> var_names_wderiv(AMREX_SPACEDIM);
    AMREX_D_TERM(var_names_wderiv[0]="dwdx";
                 , var_names_wderiv[1]="dwdy";
                 , var_names_wderiv[2]="dwdz";)

    derive_lst.add("wderiv",IndexType::TheCellType(),AMREX_SPACEDIM,
                   var_names_wderiv, cns_dervderiv,grow_box_by_one);
    derive_lst.addComponent("wderiv",desc_lst,State_Type,URHO,NUM_STATE);
#endif

    // Get the coefficients of viscosity (mu)
    derive_lst.add(
    "mu", amrex::IndexType::TheCellType(), 1, cns_dermu, the_same_box);
    derive_lst.addComponent("mu", desc_lst, State_Type, UTEMP, 1);

    // Div(u)
  derive_lst.add(
    "divu", amrex::IndexType::TheCellType(), 1, cns_derdivu,
    amrex::DeriveRec::GrowBoxByOne);
  derive_lst.addComponent("divu", desc_lst, State_Type, URHO, NUM_STATE);

  // Shear stresses (tau)
#if AMREX_SPACEDIM==2
  amrex::Vector<std::string> var_names_tau(3);
    var_names_tau[0]="tauxx";
    var_names_tau[1]="tauyy";
    var_names_tau[2]="tauxy";

    derive_lst.add("tau",IndexType::TheCellType(),3,
                   var_names_tau, cns_dertau,grow_box_by_one);
    derive_lst.addComponent("tau",desc_lst,State_Type,URHO,NUM_STATE);
#endif

#if AMREX_SPACEDIM==3
    amrex::Vector<std::string> var_names_tau(6);
    var_names_tau[0]="tauxx";
    var_names_tau[1]="tauyy";
    var_names_tau[2]="tauzz";
    var_names_tau[3]="tauxy";
    var_names_tau[4]="tauxz";
    var_names_tau[5]="tauyz";

    derive_lst.add("tau",IndexType::TheCellType(),6,
                   var_names_tau, cns_dertau,grow_box_by_one);
    derive_lst.addComponent("tau",desc_lst,State_Type,URHO,NUM_STATE);
#endif

    // Get the terms of the vorticity equation
#if AMREX_SPACEDIM==3
    // Vortex stretching due to flow gradients (zero in 2D) 
    // (this is (omega.del)u or omega_i*du_i/dx_j) 
    amrex::Vector<std::string> var_names_omdelu(3);
    var_names_omdelu[0]="omdelux";
    var_names_omdelu[1]="omdeluy";
    var_names_omdelu[2]="omdeluz";

    derive_lst.add("omdelu",IndexType::TheCellType(),3,
                   var_names_omdelu, cns_deromdelu,grow_box_by_one);
    derive_lst.addComponent("omdelu",desc_lst,State_Type,URHO,NUM_STATE);
#endif

    // Vortex stretching due to flow compressibility
    // This describes effects of expansion on vorticity field
    // the form of this term is omega * div(u) = omega_i * du_j/dx_j
#if AMREX_SPACEDIM==2
    amrex::Vector<std::string> var_names_omdivu(1);
    var_names_omdivu[0]="omdivuz";
    derive_lst.add("omdivu",IndexType::TheCellType(),1,
                   var_names_omdivu, cns_deromdivu,grow_box_by_one);
    derive_lst.addComponent("omdivu",desc_lst,State_Type,URHO,NUM_STATE);
#endif

#if AMREX_SPACEDIM==3
    amrex::Vector<std::string> var_names_omdivu(AMREX_SPACEDIM);
    AMREX_D_TERM(var_names_omdivu[0]="omdivux";
                 , var_names_omdivu[1]="omdivuy";
                 , var_names_omdivu[2]="omdivuz";)
    derive_lst.add("omdivu",IndexType::TheCellType(),AMREX_SPACEDIM,
                   var_names_omdivu, cns_deromdivu,grow_box_by_one);
    derive_lst.addComponent("omdivu",desc_lst,State_Type,URHO,NUM_STATE);
#endif

    // Baroclinic torque contribution in vorticity equation
    // This results in a generation of vorticity due to non aligned density 
    // and pressure gradients ( this is curl( grad(rho), grad(p) )/(rho*rho) )
#if AMREX_SPACEDIM==2
    amrex::Vector<std::string> var_names_baro(1);
    var_names_baro[0]="baroz";
    derive_lst.add("baro",IndexType::TheCellType(),1,
                   var_names_baro, cns_derbaroclinic,grow_box_by_one);
    derive_lst.addComponent("baro",desc_lst,State_Type,URHO,NUM_STATE);
#endif

#if AMREX_SPACEDIM==3
    amrex::Vector<std::string> var_names_baro(AMREX_SPACEDIM);
    var_names_baro[0]="barox";
    var_names_baro[1]="baroy";
    var_names_baro[2]="baroz";
    derive_lst.add("baro",IndexType::TheCellType(),AMREX_SPACEDIM,
                   var_names_baro, cns_derbaroclinic, grow_box_by_one);
    derive_lst.addComponent("baro",desc_lst,State_Type,URHO,NUM_STATE);
#endif

    // Diffusion of vorticity due to viscous effects
#if AMREX_SPACEDIM==2
    amrex::Vector<std::string> var_names_omdiff(1);
    var_names_omdiff[0]="omdiffz";
    derive_lst.add("omdiff",IndexType::TheCellType(),1,
                   var_names_omdiff, cns_deromdiff,grow_box_by_three);
    derive_lst.addComponent("omdiff",desc_lst,State_Type,URHO,NUM_STATE);
#endif

#if AMREX_SPACEDIM==3
    amrex::Vector<std::string> var_names_omdiff(3);
    var_names_omdiff[0]="omdiffx";
    var_names_omdiff[1]="omdiffy";
    var_names_omdiff[2]="omdiffz";
    derive_lst.add("omdiff",IndexType::TheCellType(),AMREX_SPACEDIM,
                   var_names_omdiff, cns_deromdiff, grow_box_by_three);
    derive_lst.addComponent("omdiff",desc_lst,State_Type,URHO,NUM_STATE);
#endif

}

void
CNS::variableCleanUp ()
{
    delete h_parm;
    delete h_prob_parm;
    The_Arena()->free(d_parm);
    The_Arena()->free(d_prob_parm);
    desc_lst.clear();
    derive_lst.clear();

#ifdef AMREX_USE_GPU
    The_Arena()->free(dp_refine_boxes);
#endif
}
