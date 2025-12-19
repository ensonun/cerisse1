#include "prob.H"

using namespace amrex;

Gpu::DeviceVector<Real> y_profile_vec_d;
Gpu::DeviceVector<Real> u_profile_vec_d;
Gpu::DeviceVector<Real> T_profile_vec_d;

extern "C" {
void amrex_probinit(const int* /*init*/, const int* /*name*/, const int* /*namelen*/,
                    const Real* /*problo*/, const Real* /*probhi*/)
{
  auto eos = pele::physics::PhysicsType::eos();

  Real M, p0 = -1.0_rt, T0 = -1.0_rt, p = -1.0_rt, T = -1.0_rt, mom_ratio,
          Tj = -1.0_rt, T0j = -1.0_rt, M_j = 1.0_rt;
  std::string inflow_file;
  {
    ParmParse pp("prob");
    pp.get("M", M);                 // inflow Mach number
    pp.query("p0", p0);             // total pressure [Ba]
    pp.query("T0", T0);             // inflow total temperature [K]
    pp.query("p", p);               // static pressure [Ba]
    pp.query("T", T);               // inflow static temperature [K]
    pp.get("mom_ratio", mom_ratio); // jet to inflow momentum ratio
    pp.query("T_j", Tj);
    pp.query("T0_j", T0j);
    // pp.query("A", CNS::h_prob_parm->A);
    pp.query("theta0", CNS::h_prob_parm->theta0);
    pp.query("M_j", M_j);
    pp.query("r_j", CNS::h_prob_parm->r_j);
    pp.query("do_spark", CNS::h_prob_parm->do_spark);
    pp.query("record_statistics", CNS::h_prob_parm->record_statistics);
    pp.query("clean_aux_on_restart", CNS::h_prob_parm->clean_aux_on_restart);
    pp.get("inflow_file", inflow_file); // inflow boundary layer profile file
  }
  if constexpr (NUM_AUX != 18) {
    if (CNS::h_prob_parm->record_statistics)
      amrex::Abort("Please compile with NUM_AUX=18 to record statistics");
  }
  if (!((p0 > 0.0_rt && T0 > 0.0_rt && p < 0.0_rt && T < 0.0_rt) ||
        (p0 < 0.0_rt && T0 < 0.0_rt && p > 0.0_rt && T > 0.0_rt))) {
    amrex::Abort("Please specify either (p0, T0) or (p, T)");
  }
  if (Tj < 0.0_rt && T0j < 0.0_rt) {
    amrex::Abort("Please specify either T_j or T0_j");
  }

  // Calculate inflow conditions
  CNS::h_prob_parm->Y[O2_ID] = 0.26_rt;
  CNS::h_prob_parm->Y[N2_ID] = 0.74_rt;
  Real rho, gam = 1.4_rt;
  int iter = 0;
  if (p0 > 0.0_rt && T0 > 0.0_rt && p < 0.0_rt && T < 0.0_rt) {
    // Total T, p: Iterate to find T, p, rho, gam
    Real gam_old = 1.0e10_rt;
    while (std::abs(gam - gam_old) > 1.0e-4_rt && iter < 20) {
      iter += 1;
      gam_old = gam;

      // Isentropic relations
      T = T0 / (1.0_rt + 0.5_rt * (gam - 1.0_rt) * M * M);
      p = p0 * std::pow(1.0_rt + 0.5_rt * (gam - 1.0_rt) * M * M, -gam / (gam - 1.0_rt));
      eos.PYT2R(p, CNS::h_prob_parm->Y.begin(), T, rho);
      eos.RTY2G(rho, T, CNS::h_prob_parm->Y.begin(), gam);
    }}
  else if (p0 < 0.0_rt && T0 < 0.0_rt && p > 0.0_rt && T > 0.0_rt) {
    // Static T, p: Find T0, p0, rho, gam directly
    eos.PYT2R(p, CNS::h_prob_parm->Y.begin(), T, rho);
    eos.RTY2G(rho, T, CNS::h_prob_parm->Y.begin(), gam);
    T0 = T * (1.0_rt + 0.5_rt * (gam - 1.0_rt) * M * M);
    p0 = p * std::pow(1.0_rt + 0.5_rt * (gam - 1.0_rt) * M * M, gam / (gam - 1.0_rt));
  }
  Real ei;
  eos.RTY2E(rho, T, CNS::h_prob_parm->Y.begin(), ei);
  Real cs = std::sqrt(gam * p / rho);
  Real u = M * cs;

  // CNS::h_prob_parm->ei_inf = ei;
  CNS::h_prob_parm->rho_inf = rho;
  CNS::h_prob_parm->T_inf = T;
  CNS::h_prob_parm->u_inf = u;
  amrex::Print() << "Inflow (gamma, T0, p0, rho, T, p, u, ei) = " << gam << ", "
                 << T0 << ", " << p0 << ", " << rho << ", " << T << ", " << p << ", "
                 << u << ", " << ei << " converged in " << iter << " iterations\n";

  // Calculate jet conditions
  CNS::h_prob_parm->Y_j[H2_ID] = 1.0_rt;
  Real cs_j;
  if (Tj < 0.0_rt) {
    // Given T0_j, assume gamma = 1.4, find T_j
    Tj = T0j / (1.0_rt + 0.5_rt * (1.4_rt - 1.0_rt) * M_j * M_j);
  }
  CNS::h_prob_parm->T_j = Tj;
  eos.RTY2Cs(1.0_rt, CNS::h_prob_parm->T_j, CNS::h_prob_parm->Y_j.begin(), cs_j); // cs should be independent of rho
  Real u_j = M_j * cs_j;
  Real rho_j = mom_ratio * rho * u * u / (u_j * u_j);
  Real ei_j, gam_j, p_j;
  eos.RTY2E(rho_j, CNS::h_prob_parm->T_j, CNS::h_prob_parm->Y_j.begin(), ei_j);
  eos.RTY2G(rho_j, CNS::h_prob_parm->T_j, CNS::h_prob_parm->Y_j.begin(), gam_j);
  eos.RTY2P(rho_j, CNS::h_prob_parm->T_j, CNS::h_prob_parm->Y_j.begin(), p_j);
  Real T0_j = CNS::h_prob_parm->T_j * (1.0_rt + 0.5_rt * (gam_j - 1.0_rt) * M_j * M_j);
  Real p0_j = p_j * std::pow(1.0_rt + 0.5_rt * (gam_j - 1.0_rt) * M_j * M_j, gam_j / (gam_j - 1.0_rt));

  CNS::h_prob_parm->ei_j = ei_j;
  CNS::h_prob_parm->rho_j = rho_j;
  CNS::h_prob_parm->u_j = u_j;
  amrex::Print() << "Jet    (gamma, T0, p0, rho, T, p, u, ei) = " << gam_j << ", " 
                 << T0_j << ", " << p0_j << ", " << rho_j << ", " << CNS::h_prob_parm->T_j 
                 << ", " << p_j << ", " << u_j << ", " << ei_j << "\n";

  // Report some global numbers
  auto tp = pele::physics::PhysicsType::transport();
  auto* tparm = &CNS::trans_parms.host_trans_parm();
  const bool wtr_get_xi = false;
  const bool wtr_get_mu = true;
  const bool wtr_get_lam = false;
  const bool wtr_get_Ddiag = false;
  const bool wtr_get_chi = false;
  Real muloc, xiloc, lamloc;
  Real *Ddiag = nullptr, *chi_mix = nullptr;
  tp.transport(wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi, T,
               rho, CNS::h_prob_parm->Y.begin(), Ddiag, chi_mix, muloc, xiloc,
               lamloc, tparm);
  Real Re_inf = rho * u / muloc * 100.0_rt; // convert to m^-1
  tp.transport(wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi,
               CNS::h_prob_parm->T_j, rho_j, CNS::h_prob_parm->Y_j.begin(), Ddiag,
               chi_mix, muloc, xiloc, lamloc, tparm);
  Real Re_j = rho_j * u_j / muloc * 100.0_rt; // convert to m^-1
  Real A = 100.0_rt;
  Real A_j = M_PI * CNS::h_prob_parm->r_j * CNS::h_prob_parm->r_j;
  Real ER = rho_j * u_j * A_j * CNS::h_prob_parm->Y_j[H2_ID] /
            (rho * u * A * CNS::h_prob_parm->Y[O2_ID]) * 8.0_rt;
  amrex::Print() << "Re_inf = " << Re_inf << " [m^-1], " << "Re_jet = " << Re_j << " [m^-1], "
                 << "ER = " << ER << "\n";  // not closed system, ER is meaningless

  // Read inflow profile
  std::ifstream infile(inflow_file);
  if (!infile.is_open()) {
    amrex::Abort("Error opening inflow profile file: " + inflow_file);
  }
  std::vector<Real> y_profile_vec;
  std::vector<Real> u_profile_vec;
  std::vector<Real> T_profile_vec;
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    Real y_val, u_val, T_val;
    if (!(iss >> y_val >> u_val >> T_val)) {
      amrex::Abort("Error reading inflow profile file: " + inflow_file);
    }
    y_profile_vec.push_back(y_val);
    u_profile_vec.push_back(u_val);
    T_profile_vec.push_back(T_val);
  }
  infile.close();
  CNS::h_prob_parm->npts = y_profile_vec.size();
  y_profile_vec_d.resize(CNS::h_prob_parm->npts);
  u_profile_vec_d.resize(CNS::h_prob_parm->npts);
  T_profile_vec_d.resize(CNS::h_prob_parm->npts);
  Gpu::copy(Gpu::hostToDevice, y_profile_vec.begin(), y_profile_vec.end(),
            y_profile_vec_d.begin());
  Gpu::copy(Gpu::hostToDevice, u_profile_vec.begin(), u_profile_vec.end(),
            u_profile_vec_d.begin());
  Gpu::copy(Gpu::hostToDevice, T_profile_vec.begin(), T_profile_vec.end(),
            T_profile_vec_d.begin());
  CNS::h_prob_parm->y_profile = y_profile_vec_d.data();
  CNS::h_prob_parm->u_profile = u_profile_vec_d.data();
  CNS::h_prob_parm->T_profile = T_profile_vec_d.data();
  amrex::Print() << "Inflow profile loaded from " << inflow_file << " with "
                 << CNS::h_prob_parm->npts << " points.\n";

  Gpu::copy(Gpu::hostToDevice, CNS::h_prob_parm, CNS::h_prob_parm + 1,
            CNS::d_prob_parm);
}
}
