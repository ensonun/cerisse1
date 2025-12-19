# Generate laminar boundary layer profile for inflow

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize  import minimize, LinearConstraint

# Define gas
class GammaGas:
  """ A class for gamma-law air """

  def __init__(self, R=8.31446261815324/28.97e-3, gamma=1.4, Pr=0.72):
    self.R     = R     # specific gas constant for air J/kgK
    self.gamma = gamma # specific heats ratio
    self.Pr    = Pr    # Prandtl number

    # Sutherland's law parameters
    self.mu0 = 1.716e-05
    self.T0  = 273.15
    self.S   = 110.4

  def cp(self, T=0):
    """ Calculate specific heat capcity for constant pressure (cp) given temperature (T) """
    return self.R * self.gamma / (self.gamma - 1.0)

  def T2h(self, T):
    """ Calculate enthalpy (h) given temperature (T) """
    h = self.cp() * T
    return h

  def h2T(self, h):
    """ Calculate temperature (T) given enthalpy (h) """
    return h / self.cp()

  def mu(self, T):
    """ Calculate viscosity (mu) given temperature (T) using Sutherland's law """
    return self.mu0 * (T/self.T0)**1.5 * (self.T0 + self.S)/(T + self.S)

# Define solver
class CBL:
  """ Similarity solution of compressible boundary layer (CBL). """

  def __init__(self, gas, Me, Te, pe, eta, adia, Tw=300.0):
    self.gas = gas
    self.Me = Me
    self.Te = Te
    self.pe = pe
    self.adia = adia
    self.Tw = Tw
    self.eta = eta

    # Calculate and store edge states
    self.he = gas.T2h(Te)
    self.hw = gas.T2h(Tw)
    self.ue = Me * np.sqrt(gas.gamma * gas.R * Te)
    self.mue = gas.mu(Te)
    self.rhoe = pe / (gas.R * Te)

  def _rhs(self, t, y):
    h = y[3] * self.he
    T = self.gas.h2T(h)
    mu = self.gas.mu(T)
    C = self.Te / T * mu / self.mue

    rhs = np.zeros((5,))
    rhs[0] = y[1]
    rhs[1] = y[2]
    rhs[2] = -y[0]*y[2]/C
    rhs[3] = y[4]
    rhs[4] = -self.gas.Pr/C*y[0]*y[4] - self.gas.Pr*(self.ue**2)/self.he*y[2]**2
    return rhs

  def integrate(self, y0):
    """ Integrate the CBL equation given solution at wall (y0) """
    sol = solve_ivp(self._rhs, [self.eta[0],self.eta[-1]], y0, t_eval=self.eta)
    return sol.y

  def solve(self, iguess):
    """ Find the solution to the CBL equation subjected to boundary conditions at edge (u/ue = T/Te = 1) """
    def get_y0(bcs):
      if (self.adia):
        return np.array([0., 0., bcs[0], bcs[1], 0.])
      else:
        return np.array([0., 0., bcs[0], self.hw/self.he, bcs[1]])

    def err(bcs):
      y0 = get_y0(bcs)
      y = self.integrate(y0)
      return (y[1,-1] - 1.0)**2 + (y[3,-1] - 1.0)**2 # L2 error

    if (self.adia):
      enthapy_constraint = LinearConstraint(np.array([0.,1.]), lb=1e-10, keep_feasible=True)
    else:
      enthapy_constraint = LinearConstraint(np.array([0.,1.]), lb=-self.Me**2/2, ub=self.Me**2/2, keep_feasible=True)

    res = minimize(err, iguess, constraints=enthapy_constraint)
    y0 = get_y0(res.x)
    print("#Iter =", res.nit, ", Final y(0) = ", y0)
    return self.integrate(y0)

if __name__ == "__main__":
  # Run
  gas = GammaGas(gamma=1.31)
  Te = 1400
  pe = 40000
  ue = 1775
  Tw = 293
  rhoe = pe / (gas.R * Te)
  mue = gas.mu(Te)
  ce = np.sqrt(gas.gamma * gas.R * Te)
  M = ue / ce
  print("Re  ~", rhoe * ue / mue, "[m^-1]")
  print("M   ~", M)
  print("T0  ~", Te + 0.5 * ue**2 / gas.cp())

  eta = np.linspace(0, 10, 2000)
  prob = CBL(gas, Me=M, Te=Te, pe=pe, eta=eta, adia=False, Tw=Tw)
  sol = prob.solve([1.0, 2.0])  # may need to play around with the initial guess
  us = sol[1]
  Ts = sol[3]

  # Convert to physical coordinates, normalised by 99% delta
  y = np.cumsum(np.gradient(eta) * Ts)
  bl_edge_pos = np.argmax(us > 0.99)
  output_per = int(bl_edge_pos / 20)  # outputs ~20 points
  y_output = y[:bl_edge_pos:output_per] / y[bl_edge_pos]
  us_output = us[:bl_edge_pos:output_per]
  Ts_output = Ts[:bl_edge_pos:output_per]
  np.savetxt("inflow_bl.csv", np.stack([y_output, us_output, Ts_output]).T, delimiter=" ")