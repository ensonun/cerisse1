#ifndef CNS_PROB_H
#define CNS_PROB_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "CNS.H"
#include "PelePhysics.H"
#include "bc_util.H"
#include "index_macros.H"
#include "mechanism.H"
#include "prob_parm.H"

namespace{
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real 
inflow_profile(const amrex::Real r_d, const amrex::Real y1,
               const amrex::Real y2, const amrex::Real y3)
{
  // Profile function fitted from experimental data at x/D = 0.85
  constexpr amrex::Real s1 = 7.0;  // steepness of tanh
  constexpr amrex::Real s2 = 5.0;
  constexpr amrex::Real t1 = 0.53;  // transition location
  constexpr amrex::Real t2 = 3.76;
  return y1 + (y2 - y1) * 0.5_rt * (std::tanh(s1 * (r_d - t1)) + 1.0_rt) +
         (y3 - y2) * 0.5_rt * (std::tanh(s2 * (r_d - t2)) + 1.0_rt);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real 
u_profile_corr(const amrex::Real r_d)
{
  // To reproduce the velocity dip from experimental data at x/D = 0.85
  constexpr amrex::Real s1 = 7.0;  // steepness of tanh
  constexpr amrex::Real s2 = 5.0;
  constexpr amrex::Real t1 = 0.53;  // transition location
  constexpr amrex::Real t2 = 3.76;
  constexpr amrex::Real slp = 14000.0_rt;  // control slope
  return slp * 0.5_rt * (r_d - t2 + 0.5_rt) *
         (std::tanh(s1 * (r_d - t1)) - std::tanh(s2 * (r_d - t2)));
}
} // namespace

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_initdata(int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, ProbParm const& pp)
{
  // Geometry
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
  const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
  const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
  const amrex::Real r_D = std::sqrt(AMREX_D_TERM(, y * y, +z * z)) / pp.D;

  const amrex::Real T = inflow_profile(r_D, pp.T1, pp.T2, pp.T3);
  const amrex::Real p = inflow_profile(r_D, pp.p1, pp.p2, pp.p3);
  const amrex::Real u = inflow_profile(r_D, pp.u1, pp.u2, pp.u3) + u_profile_corr(r_D);
  amrex::Real X[NUM_SPECIES] = {0.0};
  X[H2_ID] = inflow_profile(r_D, pp.X1[H2_ID], pp.X2[H2_ID], pp.X3[H2_ID]);
  X[O2_ID] = inflow_profile(r_D, pp.X1[O2_ID], pp.X2[O2_ID], pp.X3[O2_ID]);
  X[N2_ID] = inflow_profile(r_D, pp.X1[N2_ID], pp.X2[N2_ID], pp.X3[N2_ID]);
  X[H2O_ID] = inflow_profile(r_D, pp.X1[H2O_ID], pp.X2[H2O_ID], pp.X3[H2O_ID]);

  amrex::Real Y[NUM_SPECIES];
  amrex::Real rho, ei;
  auto eos = pele::physics::PhysicsType::eos();
  eos.X2Y(X, Y);
  eos.PYT2RE(p, Y, T, rho, ei);

  for (int nf = 0; nf <= NUM_FIELD; ++nf) {
    state(i, j, k, nf * NVAR + URHO) = rho;
    state(i, j, k, nf * NVAR + UMX) = rho * u;
    state(i, j, k, nf * NVAR + UMY) = 0.0;
    state(i, j, k, nf * NVAR + UMZ) = 0.0;
    state(i, j, k, nf * NVAR + UEDEN) = rho * ei + 0.5_rt * rho * u * u;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      state(i, j, k, nf * NVAR + UFS + n) = rho * Y[n];
    }
    state(i, j, k, nf * NVAR + UTEMP) = T;
  }

#if NUM_AUX > 0
  // Aux variables for storing time averages
  for (int n = 0; n < NUM_AUX; ++n) { state(i, j, k, UFA + n) = 0.0; }
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void bcnormal(
  const amrex::Real x[AMREX_SPACEDIM], const amrex::Real s_int[NUM_FIELD * NVAR],
  const amrex::Real* /*s_refl[NUM_FIELD * NVAR]*/,
  amrex::Real s_ext[NUM_FIELD * NVAR], const int idir, const int sgn,
  const amrex::Real /*time*/, amrex::GeometryData const& /*geomdata*/, ProbParm const& pp)
{
  BL_PROFILE("bcnormal()");

  const amrex::Real r_D = std::sqrt(AMREX_D_TERM(, x[1] * x[1], +x[2] * x[2])) / pp.D;

  const amrex::Real T = inflow_profile(r_D, pp.T1, pp.T2, pp.T3);
  const amrex::Real p = inflow_profile(r_D, pp.p1, pp.p2, pp.p3);
  const amrex::Real u = inflow_profile(r_D, pp.u1, pp.u2, pp.u3) + u_profile_corr(r_D);
  amrex::Real X[NUM_SPECIES] = {0.0};
  X[H2_ID] = inflow_profile(r_D, pp.X1[H2_ID], pp.X2[H2_ID], pp.X3[H2_ID]);
  X[O2_ID] = inflow_profile(r_D, pp.X1[O2_ID], pp.X2[O2_ID], pp.X3[O2_ID]);
  X[N2_ID] = inflow_profile(r_D, pp.X1[N2_ID], pp.X2[N2_ID], pp.X3[N2_ID]);
  X[H2O_ID] = inflow_profile(r_D, pp.X1[H2O_ID], pp.X2[H2O_ID], pp.X3[H2O_ID]);

  amrex::Real Y[NUM_SPECIES];
  amrex::Real rho, ei;
  auto eos = pele::physics::PhysicsType::eos();
  eos.X2Y(X, Y);
  eos.PYT2RE(p, Y, T, rho, ei);

  amrex::Real v = 0.0;
  amrex::Real w = 0.0;
  if (pp.inflow_turbulence) {
    // do something
  }

  for (int nf = 0; nf <= NUM_FIELD; ++nf) {
    if (idir == 0 && sgn == 1) {
      s_ext[nf * NVAR + URHO] = rho;
      s_ext[nf * NVAR + UMX] = rho * u;
      s_ext[nf * NVAR + UMY] = rho * v;
      s_ext[nf * NVAR + UMZ] = rho * w;
      s_ext[nf * NVAR + UEDEN] = rho * ei + 0.5_rt * rho * (u * u + v * v + w * w);
      for (int n = 0; n < NUM_SPECIES; ++n) {
        s_ext[nf * NVAR + UFS + n] = rho * Y[n];
      }
      s_ext[nf * NVAR + UTEMP] = T;

    } else if (idir == 0 && sgn == -1) {
      bc_foextrap_outlet(idir, sgn, &s_ext[nf * NVAR], &s_int[nf * NVAR]);
    }
  }

#if NUM_AUX > 0
  // Aux variables for storing time averages
  for (int n = 0; n < NUM_AUX; ++n) { s_ext[UFA + n] = 0.0; }
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void fill_ext_src(
  int i, int j, int k, amrex::Real time, amrex::GeometryData const& geomdata,
  amrex::Array4<const amrex::Real> const& /*state*/,
  amrex::Array4<amrex::Real> const& ext_src, ProbParm const& pp)
{
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_post_restart(int i, int j, int k, amrex::Array4<amrex::Real> const& state,
                  amrex::GeometryData const& geomdata, ProbParm const& pp)
{
#if NUM_AUX > 0
  if (pp.clean_aux_on_restart) {
    // Clear time avg
    for (int n = 0; n < NUM_AUX; ++n) { state(i, j, k, UFA + n) = 0.0; }
  }
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_tag_error(int i, int j, int k, amrex::Array4<char> const& tagarr,
               amrex::Array4<const amrex::Real> const& sarr, int level, char tagval,
               const amrex::Real /*time*/, amrex::GeometryData const& geomdata,
               ProbParm const& /*pp*/)
{
  BL_PROFILE("prob_tag_error()");

  // Geometry
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
  const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
#if AMREX_SPACEDIM == 2
  const amrex::Real z = 0.0;
#else
  const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
#endif
  const amrex::Real rsqr = y * y + z * z;

  // Tag cylinder
  // if (level == 0 && rsqr <= 2.0 * 2.0) { tagarr(i, j, k) = tagval; }
  // if (level == 1 && rsqr <= 1.7 * 1.7) { tagarr(i, j, k) = tagval; }
  // if (level == 2 && rsqr <= 0.3 * 0.3) { tagarr(i, j, k) = tagval; }

  // Tag injector
  if ((rsqr < 1.0_rt) && (x < 1.0_rt)) { tagarr(i, j, k) = tagval; }

  // Tag flame
  // amrex::Real Y_OH = sarr(i, j, k, UFS + OH_ID) / sarr(i, j, k, URHO);
  // amrex::Real Y_HO2 = sarr(i, j, k, UFS + HO2_ID) / sarr(i, j, k, URHO);
  // if (Y_OH >= 0.02 || Y_HO2 >= 1e-3) { tagarr(i, j, k) = tagval; }

  // Tag Y_H2
  const amrex::Real Y_H2 = sarr(i, j, k, UFS + H2_ID) / sarr(i, j, k, URHO);
  if (Y_H2 > 0.03) { tagarr(i, j, k) = tagval; }

  // Untag outflow (70D)
  if (x > 16.5_rt) { tagarr(i, j, k) = amrex::TagBox::TagVal::CLEAR; }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_post_timestep(int i, int j, int k, const amrex::Real /*curtime*/,
                   const amrex::Real dtlev, amrex::Array4<amrex::Real> const& state,
                   amrex::Array4<const amrex::Real> const& reactions,
                   amrex::GeometryData const& /*geomdata*/, ProbParm const& pp)
{
#if NUM_AUX > 0
  if (pp.recor_D_statistics) {
    Real rho = state(i, j, k, URHO);
    Real rhoinv = Real(1.0) / rho;
    Real vx = state(i, j, k, UMX) * rhoinv;
    Real vy = state(i, j, k, UMY) * rhoinv;
    Real vz = state(i, j, k, UMZ) * rhoinv;
    Real ei = state(i, j, k, UEDEN) * rhoinv -
              0.5 * (AMREX_D_TERM(vx * vx, +vy * vy, +vz * vz));
    Real massfrac[NUM_SPECIES];
    for (int n = 0; n < NUM_SPECIES; ++n) {
      massfrac[n] = state(i, j, k, UFS + n) * rhoinv;
    }

    Real T = 0.0;
    Real X[NUM_SPECIES];
    auto eos = pele::physics::PhysicsType::eos();
    eos.REY2T(rho, ei, massfrac, T);
    eos.Y2X(massfrac, X);

    // Mean
    state(i, j, k, UFA + 0) += vx * dtlev;        // U
    state(i, j, k, UFA + 1) += vy * dtlev;        // V
    state(i, j, k, UFA + 2) += vz * dtlev;        // W
    state(i, j, k, UFA + 3) += T * dtlev;         // T
    state(i, j, k, UFA + 4) += X[H2_ID] * dtlev;  // X(H2)
    state(i, j, k, UFA + 5) += X[OH_ID] * dtlev;  // X(OH)
    state(i, j, k, UFA + 6) += X[H2O_ID] * dtlev; // X(H2O)
    state(i, j, k, UFA + 7) += X[N2_ID] * dtlev;  // X(N2)
    state(i, j, k, UFA + 8) += X[O2_ID] * dtlev;  // X(O2)
    state(i, j, k, UFA + 9) +=
      reactions(i, j, k, NUM_SPECIES) * dtlev; // heat release rate

    // RMS^2
    state(i, j, k, UFA + 10) += T * T * dtlev;                 // T
    state(i, j, k, UFA + 11) += X[H2_ID] * X[H2_ID] * dtlev;   // X(H2)
    state(i, j, k, UFA + 12) += X[OH_ID] * X[OH_ID] * dtlev;   // X(OH)
    state(i, j, k, UFA + 13) += X[H2O_ID] * X[H2O_ID] * dtlev; // X(H2O)
    state(i, j, k, UFA + 14) += X[N2_ID] * X[N2_ID] * dtlev;   // X(N2)
    state(i, j, k, UFA + 15) += X[O2_ID] * X[O2_ID] * dtlev;   // X(O2)
  }
#endif
}

#if NUM_AUX > 0
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
prob_get_aux_name(amrex::Vector<std::string>& aux_name)
{
  aux_name.resize(NUM_AUX);

  aux_name[0] = "time_avg_u";
  aux_name[1] = "time_avg_v";
  aux_name[2] = "time_avg_w";
  aux_name[3] = "time_avg_T";
  aux_name[4] = "time_avg_X_H2";
  aux_name[5] = "time_avg_X_OH";
  aux_name[6] = "time_avg_X_H2O";
  aux_name[7] = "time_avg_X_N2";
  aux_name[8] = "time_avg_X_O2";
  aux_name[9] = "time_avg_hrr";
  aux_name[10] = "time_avg_T_sqr";
  aux_name[11] = "time_avg_X_H2_sqr";
  aux_name[12] = "time_avg_X_OH_sqr";
  aux_name[13] = "time_avg_X_H2O_sqr";
  aux_name[14] = "time_avg_X_N2_sqr";
  aux_name[15] = "time_avg_X_O2_sqr";
}
#endif

#endif
