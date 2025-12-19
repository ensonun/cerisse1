#ifndef CNS_PROB_H
#define CNS_PROB_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "CNS.H"
#include "PelePhysics.H"
#include "bc_util.H"
#include "prob_parm.H"

namespace {
// From PelePhysics
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int locate(const amrex::Real* xtable,
                                                    const int n,
                                                    const amrex::Real& x)
{
  // If x is out of bounds, return boundary index
  if (x >= xtable[n - 1]) { return n - 1; }
  if (x <= xtable[0]) { return 0; }

  // Do the bisection
  int idxlo = 0;
  int idxhi = n - 1;
  bool notdone = true;
  while (notdone) {
    if (idxhi - idxlo <= 1) {
      notdone = false;
    } else {
      const int idxmid = (idxhi + idxlo) / 2;
      if (x >= xtable[idxmid]) {
        idxlo = idxmid;
      } else {
        idxhi = idxmid;
      }
    }
  }
  return idxlo;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void bl_profile(const amrex::Real& d,
                                                    ProbParm const& pp,
                                                    amrex::Real& u_ratio,
                                                    amrex::Real& T_ratio)
{
  // Linear interpolation
  const int idx = locate(pp.y_profile, pp.npts, d);
  if (idx == pp.npts - 1) {
    u_ratio = pp.u_profile[idx];
    T_ratio = pp.T_profile[idx];
    return;
  }
  amrex::Real w = (d - pp.y_profile[idx]) / (pp.y_profile[idx + 1] - pp.y_profile[idx]);
  u_ratio = pp.u_profile[idx + 1] * w + pp.u_profile[idx] * (1.0_rt - w);
  T_ratio = pp.T_profile[idx + 1] * w + pp.T_profile[idx] * (1.0_rt - w);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real jet_profile(const amrex::Real& r,
                                                            const int pow)
{
  // pow >= 1
  return amrex::Real(pow + 1) / amrex::Real(pow) * (1.0_rt - std::pow(r, pow));
}
} // namespace

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_initdata(int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, ProbParm const& pp)
{
  // Geometry
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();
  const amrex::Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
  const amrex::Real d = y - prob_lo[1]; // amrex::min(x[1] - prob_lo[1], prob_hi[1] - x[1]);

  amrex::Real u_ratio, T_ratio;
  bl_profile(d / pp.theta0, pp, u_ratio, T_ratio);

  amrex::Real rho = pp.rho_inf / T_ratio;
  amrex::Real u = pp.u_inf * u_ratio;
  amrex::Real T = pp.T_inf * T_ratio;
  amrex::Real ei;
  auto eos = pele::physics::PhysicsType::eos();
  eos.RTY2E(rho, T, pp.Y.begin(), ei);

  // Ventilated air
  for (int nf = 0; nf <= NUM_FIELD; ++nf) {
    state(i, j, k, nf * NVAR + URHO) = rho;
    state(i, j, k, nf * NVAR + UMX) = rho * u;
    state(i, j, k, nf * NVAR + UMY) = 0.0_rt;
    state(i, j, k, nf * NVAR + UMZ) = 0.0_rt;
    state(i, j, k, nf * NVAR + UEDEN) = rho * ei + 0.5_rt * rho * u * u;
    for (int n = 0; n < NUM_SPECIES; ++n)
      state(i, j, k, nf * NVAR + UFS + n) = rho * pp.Y[n];
    state(i, j, k, nf * NVAR + UTEMP) = T;
  }

  for (int na = 0; na < NUM_AUX; ++na) state(i, j, k, UFA + na) = 0.0_rt;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void bcnormal(
  const amrex::Real x[AMREX_SPACEDIM], const amrex::Real* /*s_int[LEN_STATE]*/,
  const amrex::Real s_refl[LEN_STATE], amrex::Real s_ext[LEN_STATE], const int idir,
  const int sgn, const amrex::Real time, amrex::GeometryData const& geomdata,
  ProbParm const& pp)
{
  if (idir == 0) {
    // Inflow / Ventilated air    
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real d = x[1] - prob_lo[1]; // amrex::min(x[1] - prob_lo[1], prob_hi[1] - x[1]);
    
    amrex::Real u_ratio, T_ratio;
    bl_profile(d / pp.theta0, pp, u_ratio, T_ratio);

    amrex::Real rho = pp.rho_inf / T_ratio;
    amrex::Real u = pp.u_inf * u_ratio;
    amrex::Real T = pp.T_inf * T_ratio;
    amrex::Real v = 0.0_rt;
    amrex::Real w = 0.0_rt;
    amrex::Real ei;
    auto eos = pele::physics::PhysicsType::eos();
    eos.RTY2E(rho, T, pp.Y.begin(), ei);

    for (int nf = 0; nf <= NUM_FIELD; ++nf) {
      // v = amrex::RandomNormal(0.0_rt, 0.03_rt * pp.u_inf);
      // w = amrex::RandomNormal(0.0_rt, 0.03_rt * pp.u_inf);

      s_ext[nf * NVAR + URHO] = rho;
      s_ext[nf * NVAR + UMX] = rho * u;
      s_ext[nf * NVAR + UMY] = rho * v;
      s_ext[nf * NVAR + UMZ] = rho * w;
      s_ext[nf * NVAR + UEDEN] = rho * ei + 0.5_rt * rho * (u * u + v * v + w * w);
      for (int n = 0; n < NUM_SPECIES; ++n)
        s_ext[nf * NVAR + UFS + n] = rho * pp.Y[n];
      s_ext[nf * NVAR + UTEMP] = T;
    }
  } else if (idir == 1) {
    const amrex::Real rsqr = AMREX_D_TERM(, x[0] * x[0], +x[2] * x[2]);
    if (rsqr > pp.r_j * pp.r_j) {      
      // Adiabatic no-slip wall
      for (int nf = 0; nf <= NUM_FIELD; ++nf)
        bc_adiabatic_wall(&s_ext[nf * NVAR], &s_refl[nf * NVAR], idir);
    } else {
      // Fuel jet @ x = 0, z = 0
      // amrex::Real damping = 1.0; // = time > 2e-4 ? 1.0 : 1.0 - std::exp(-5.0 * time / 2e-4);
      amrex::Real rho = pp.rho_j;
      amrex::Real v = amrex::Real(sgn) * pp.u_j; // * damping * jet_profile(std::sqrt(rsqr) / pp.r_j, 4);
      amrex::Real ei = pp.ei_j;
      amrex::Real T = pp.T_j;

      for (int nf = 0; nf <= NUM_FIELD; ++nf) {
        s_ext[nf * NVAR + URHO] = rho;
        s_ext[nf * NVAR + UMX] = 0.0_rt;
        s_ext[nf * NVAR + UMY] = rho * v;
        s_ext[nf * NVAR + UMZ] = 0.0_rt;
        s_ext[nf * NVAR + UEDEN] = rho * ei + 0.5_rt * rho * v * v;
        for (int n = 0; n < NUM_SPECIES; ++n)
          s_ext[nf * NVAR + UFS + n] = rho * pp.Y_j[n];
        s_ext[nf * NVAR + UTEMP] = T;
      }
    }
  }

  for (int na = 0; na < NUM_AUX; ++na) { s_ext[UFA + na] = 0.0; }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void fill_ext_src(
  int i, int j, int k, amrex::Real time, amrex::GeometryData const& geomdata,
  amrex::Array4<const amrex::Real> const& state,
  amrex::Array4<amrex::Real> const& ext_src, ProbParm const& pp)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();
  const amrex::Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
  const amrex::Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
#if AMREX_SPACEDIM == 3
  // const amrex::Real Lz = prob_hi[2] - prob_lo[2];
  const amrex::Real z = prob_lo[2] + (k + 0.5_rt) * dx[2];
#else
  // const amrex::Real Lz = 1.0_rt;
  const amrex::Real z = 0.0_rt;
#endif

  // Trip BL
  // amrex::Real theta = 0.3;
  // amrex::Real rsq = (x - prob_lo[0] - 5.0_rt) * (x - prob_lo[0] - 5.0_rt) +
  //                   (y - theta) * (y - theta);
  // if (rsq < theta * theta) {
  //   for (int nf = 0; nf <= NUM_FIELD; ++nf) {
  //     ext_src(i, j, k, nf * NVAR + UMY) +=
  //       pp.rho_inf * pp.u_inf * 10.0_rt * std::exp(-5.0_rt * rsq / theta / theta) *
  //       std::cos(12.0_rt * M_PI * z / Lz) *
  //       (std::sin(1e-6_rt * time) + std::sin(1e-7_rt * time));
  //     ext_src(i, j, k, nf * NVAR + UMZ) +=
  //       pp.rho_inf * pp.u_inf * 10.0_rt * std::exp(-5.0_rt * rsq / theta / theta) *
  //       std::cos(12.0_rt * M_PI * z / Lz) *
  //       (std::sin(1e-6_rt * time) + std::sin(1e-7_rt * time));
  //   }
  // }  
  // amrex::Real Amp[2] = {0.05, 0.05}; // perturbation amplitude (*u_inf)
  // amrex::Real omg[2] = {2./7., -2./7.}; // perturbation frequency (/theta0)
  // amrex::Real phi[2] = {9., 9.}; // perturbation wavenumber (*cs/theta0)
  // amrex::Real fg = std::exp(-std::pow((x / pp.theta0 - prob_lo[0] / pp.theta0 - 5.0_rt / pp.theta0), 2.0_rt)) *
  //                  std::exp(-std::pow(y / pp.theta0 - 1.0_rt, 2.0_rt)) *
  //                  (1.0_rt + 0.1_rt * std::exp(-std::pow(z / Lz - 1.0_rt / 3.0_rt, 2.0_rt)));
  // amrex::Real omg = 0.9_rt * 54802.0_rt / pp.theta0;
  // amrex::Real phi = 8.0_rt * M_PI;
  // amrex::Real dt = 1.0e-2; // TODO: this is an estimate because we cannot know real dt
  // for (int nf = 0; nf <= NUM_FIELD; ++nf) {
  //   ext_src(i, j, k, nf * NVAR + UMY) -= (state(i, j, k, nf * NVAR + UMY) / state(i, j, k, nf * NVAR + URHO) 
  //   - pp.A * pp.u_inf * fg * (std::sin(omg * time - phi * z / Lz) + std::sin(omg * time + phi * z / Lz))) / dt;
  // }

  // Spark @ 4*r_j downstream injection
  amrex::Real rsq = (x - 4.0_rt * pp.r_j) * (x - 4.0_rt * pp.r_j) + (y - pp.r_j) * (y - pp.r_j) + z * z;
  if (pp.do_spark && rsq < pp.r_j * pp.r_j) {
    for (int nf = 0; nf <= NUM_FIELD; ++nf) {
      ext_src(i, j, k, nf * NVAR + UEDEN) += state(i, j, k, nf * NVAR + URHO) * 1e8_rt;
    }
  }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_post_restart(int i, int j, int k, amrex::Array4<amrex::Real> const& state,
                  amrex::GeometryData const& geomdata, ProbParm const& pp)
{
#if NUM_AUX > 0
  if (pp.clean_aux_on_restart) { // Clear time avg
    for (int n = 0; n < NUM_AUX; ++n) { state(i, j, k, UFA + n) = 0.0; }
  }
#endif
}

#if NUM_AUX == 18
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_get_aux_name(amrex::Vector<std::string>& aux_name)
{
  aux_name.resize(NUM_AUX);

  aux_name[0] = "mean_rho";
  aux_name[1] = "mean_rhou";
  aux_name[2] = "mean_rhov";
  aux_name[3] = "mean_rhow";
  aux_name[4] = "mean_T";
  aux_name[5] = "mean_p";
  aux_name[6] = "mean_rhoY_H2";
  aux_name[7] = "mean_rhoY_OH";
  aux_name[8] = "mean_rhoY_H2O";
  aux_name[9] = "mean_rhoY_HO2";
  aux_name[10] = "mean_hrr";
  aux_name[11] = "mean_pp";
  aux_name[12] = "mean_rhouu";
  aux_name[13] = "mean_rhovv";
  aux_name[14] = "mean_rhoww";
  aux_name[15] = "mean_rhouv";
  aux_name[16] = "mean_rhouw";
  aux_name[17] = "mean_rhovw";
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_post_timestep(int i, int j, int k, const amrex::Real /*curtime*/,
                   const amrex::Real dtlev, amrex::Array4<amrex::Real> const& state,
                   amrex::Array4<const amrex::Real> const& reactions,
                   amrex::GeometryData const& /*geomdata*/, ProbParm const& pp)
{
#if NUM_AUX > 0
  if (pp.record_statistics) { // Record time averages (mean field)
    amrex::Real rho = state(i, j, k, URHO);
    amrex::Real rhoinv = 1.0_rt / rho;
    amrex::Real vz = 0.0;
    AMREX_D_TERM(amrex::Real vx = state(i, j, k, UMX) * rhoinv;
                 , amrex::Real vy = state(i, j, k, UMY) * rhoinv;
                 , vz = state(i, j, k, UMZ) * rhoinv;);
    amrex::Real ke = 0.5_rt * (AMREX_D_TERM(vx * vx, +vy * vy, +vz * vz));
    amrex::Real ei = state(i, j, k, UEDEN) * rhoinv - ke;
    amrex::Real Y[NUM_SPECIES];
    for (int n = 0; n < NUM_SPECIES; ++n) {
      Y[n] = state(i, j, k, UFS + n) * rhoinv;
    }
    amrex::Real T = state(i, j, k, UTEMP), p;
    auto eos = pele::physics::PhysicsType::eos();
    eos.REY2T(rho, ei, Y, T);
    eos.RTY2P(rho, T, Y, p);

    state(i, j, k, UFA + 0) += rho * dtlev;                          // rho
    state(i, j, k, UFA + 1) += state(i, j, k, UMX) * dtlev;          // rhou
    state(i, j, k, UFA + 2) += state(i, j, k, UMY) * dtlev;          // rhov
    state(i, j, k, UFA + 3) += state(i, j, k, UMZ) * dtlev;          // rhow
    state(i, j, k, UFA + 4) += T * dtlev;                            // T
    state(i, j, k, UFA + 5) += p * dtlev;                            // p
    state(i, j, k, UFA + 6) += state(i, j, k, UFS + H2_ID) * dtlev;  // rhoY_H2
    state(i, j, k, UFA + 7) += state(i, j, k, UFS + OH_ID) * dtlev;  // rhoY_OH
    state(i, j, k, UFA + 8) += state(i, j, k, UFS + H2O_ID) * dtlev; // rhoY_H2O
    state(i, j, k, UFA + 9) += state(i, j, k, UFS + HO2_ID) * dtlev; // rhoY_HO2
    state(i, j, k, UFA + 10) += reactions(i, j, k, NUM_SPECIES) * dtlev; // heat release rate
    state(i, j, k, UFA + 11) += p * p * dtlev;                       // p^2
    state(i, j, k, UFA + 12) += rho * vx * vx * dtlev;               // rhouu
    state(i, j, k, UFA + 13) += rho * vy * vy * dtlev;               // rhovv
    state(i, j, k, UFA + 14) += rho * vz * vz * dtlev;               // rhoww
    state(i, j, k, UFA + 15) += rho * vx * vy * dtlev;               // rhouv
    state(i, j, k, UFA + 16) += rho * vx * vz * dtlev;               // rhouw
    state(i, j, k, UFA + 17) += rho * vy * vz * dtlev;               // rhovw
  }
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_tag_error(int i, int j, int k, amrex::Array4<char> const& tagarr,
               amrex::Array4<const amrex::Real> const& sarr, int level, char tagval,
               const amrex::Real time, amrex::GeometryData const& geomdata,
               ProbParm const& /*pp*/)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  // const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();
  const amrex::Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
  const amrex::Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
  const amrex::Real z = prob_lo[2] + (k + 0.5_rt) * dx[2];

  // Untag outflow boundaries
  if ((x > 9.0_rt) || (y > 4.5_rt) || (std::abs(z) > 4.0_rt)) {
    tagarr(i, j, k) = amrex::TagBox::TagVal::CLEAR;
  }
}

#endif
