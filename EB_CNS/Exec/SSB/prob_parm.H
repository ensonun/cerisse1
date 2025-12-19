#ifndef CNS_PROB_PARM_H
#define CNS_PROB_PARM_H

#include <AMReX_GpuMemory.H>
#include <AMReX_REAL.H>

struct ProbParm
{
  // Fuel inlet
  amrex::Real T1, p1, u1;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> X1 = {0.0};
  amrex::Real D = 0.236;  // fuel jet diameter [cm]

  // Vitiated air inlet
  amrex::Real T2, p2, u2;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> X2 = {0.0};

  // Atmospheric
  amrex::Real T3, p3, u3;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> X3 = {0.0};

  bool record_statistics = false;
  bool clean_aux_on_restart = false;
  bool inflow_turbulence = false;
};

#endif
