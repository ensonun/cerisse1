#include "prob.H"

extern "C" {
void amrex_probinit(const int* /*init*/, const int* /*name*/, const int* /*namelen*/,
                    const amrex::Real* problo, const amrex::Real* probhi)
{
  // Parse params
  {
    amrex::ParmParse pp("prob");
    pp.query("record_statistics", CNS::h_prob_parm->record_statistics);
    pp.query("clean_aux_on_restart", CNS::h_prob_parm->clean_aux_on_restart);
    pp.query("inflow_turbulence", CNS::h_prob_parm->inflow_turbulence);
  }
  if constexpr (NUM_AUX != 16) {
    if (CNS::h_prob_parm->record_statistics) {
      amrex::Abort("Please compile with NUM_AUX=16 to record statistics");
    }
  }

  // Fuel inlet
  CNS::h_prob_parm->T1 = 600;
  CNS::h_prob_parm->p1 = 1120000;
  CNS::h_prob_parm->u1 = 178000;
  CNS::h_prob_parm->X1[H2_ID] = 1.0;

  // Vitiated air inlet
  CNS::h_prob_parm->T2 = 1270;
  CNS::h_prob_parm->p2 = 1070000;
  CNS::h_prob_parm->u2 = 124400; // from meassurement or 141700 from inflow condition
  CNS::h_prob_parm->X2[O2_ID] = 0.201;
  CNS::h_prob_parm->X2[N2_ID] = 0.544;
  CNS::h_prob_parm->X2[H2O_ID] = 0.255;

  // Atmospheric
  CNS::h_prob_parm->T3 = 296;
  CNS::h_prob_parm->p3 = 1013250;
  CNS::h_prob_parm->u3 = 100; // very small to avoid backflow
  CNS::h_prob_parm->X3[O2_ID] = 0.246;
  CNS::h_prob_parm->X3[N2_ID] = 0.74;
  CNS::h_prob_parm->X3[H2O_ID] = 0.014;

  // Check inlet Mach numbers
  auto eos = pele::physics::PhysicsType::eos();
  Real Y1[NUM_SPECIES] = {0.0};
  Real Y2[NUM_SPECIES] = {0.0};
  eos.X2Y(CNS::h_prob_parm->X1.data(), Y1);
  eos.X2Y(CNS::h_prob_parm->X2.data(), Y2);
  Real rho1, rho2;
  eos.PYT2R(CNS::h_prob_parm->p1, Y1, CNS::h_prob_parm->T1, rho1);
  eos.PYT2R(CNS::h_prob_parm->p2, Y2, CNS::h_prob_parm->T2, rho2);
  Real c1, c2;
  eos.RTY2Cs(rho1, CNS::h_prob_parm->T1, Y1, c1);  
  eos.RTY2Cs(rho2, CNS::h_prob_parm->T2, Y2, c2);
  amrex::Print() << "Confirm inlet Mach number: M_fuel = 1: " << CNS::h_prob_parm->u1 / c1
                 << " M_coflow = 2: " << CNS::h_prob_parm->u2 / c2 << std::endl;

  Gpu::copy(Gpu::hostToDevice, CNS::h_prob_parm, CNS::h_prob_parm + 1,
            CNS::d_prob_parm);
}
}