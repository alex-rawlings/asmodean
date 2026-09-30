Physics and numerics
====================

Equations of motion
-------------------

Each active body ``i`` obeys

.. code-block:: text

   a_i = a_ext(x_i) - sum_{j != i} G m_j g(r_ij, h_ij) (x_i - x_j) + a_df,i

where ``a_ext`` is the static lanfear potential's acceleration, ``g`` is the
Gadget spline-softened pair force and ``a_df`` the dynamical friction. The C++
core integrates in the potential's internal units (``G = 1``, length and mass
units the potential's scale radius and field mass); the python layer converts
to and from physical units.

The potential
-------------

Any lanfear potential -- SCF (``lanfear.Potential``), disc basis
(``lanfear.DiscPotential``) or per-species composite
(``lanfear.MultiComponentPotential``), with its softened black holes -- is
rebuilt inside asmodean's C++ core from the state lanfear pickles it with. The
lanfear C++ core is header-only and compiled into asmodean, so the forces are
exactly lanfear's (the test suite checks agreement to 1e-13) with no python in
the integration loop. The figure never rotates: a lanfear pattern speed is
ignored with a warning.

Softening
---------

As in Gadget (Springel 2005; Springel et al. 2021), each body's ``softening``
is the Plummer-equivalent length epsilon, and the force uses the cubic-spline
kernel with compact support ``h = 2.8 epsilon``: exactly Newtonian for
``r >= h``, with a central potential depth of ``-G m / epsilon``. Two bodies
of different softening interact with ``h_ij = max(h_i, h_j)``. Zero softening
is exactly Newtonian.

Integration
-----------

An embedded Runge-Kutta pair with adaptive error control advances the bodies:
Fehlberg 7(8) (``stepper="rkf78"``, default) or Dormand-Prince 5(4)
(``"dopri5"``). Two caps keep encounters resolved:

* every step is limited to ``step_factor`` times the shortest pairwise
  crossing time ``r_ij / |v_ij|`` or free-fall time ``sqrt(r_ij^3 / G(m_i +
  m_j))``, so no collision or close passage is stepped over;
* while a pair is inside (1.2 times) its softening kernel, each step moves it
  at most ``kernel_step_factor * h_ij``. The spline force is only piecewise
  smooth (breakpoints at ``h/2`` and ``h``), which fools a high-order error
  estimate: without this cap a passage through the kernel can lose ~1e-4 of
  the energy regardless of the tolerance.

Steps are also shortened to land exactly on the uniformly spaced output times.

Mergers
-------

When two bodies come within ``collision_distance`` they are lumped together.
The moment of contact is located by bisection within the step; the pair is
replaced by one body with their total mass, centre-of-mass position and
velocity (so momentum is conserved exactly) and the larger softening. The more
massive body survives (the lower index on a tie) and the other is marked
``"merged"``. Several contacts in one step are merged closest first.

Ejections
---------

An escaped body is removed and no longer integrated, nor does it act on the
others. With ``ejection_reference="system"`` (default) a body is ejected when
it is beyond ``ejection_radius`` from the centre of mass of the remaining
bodies, receding from it, unbound from their combined mass (Keplerian), and
not bound to any single one of them -- so a member of a bound pair that happens
to be far from the rest is kept. With ``"centre"`` it must be beyond
``ejection_radius`` from the potential centre, receding, with positive
specific energy in the potential plus the other bodies' softened field.
Several candidates in one step are removed farthest first, re-testing the rest
after each removal.

Stopping
--------

The integration stops at the first of: the time limit; every body merged into
one (``"all_merged"``); at most one body left after an ejection
(``"bodies_ejected"``); the step limit (``"step_limit"``); or a non-finite
state or step-size underflow (``"integration_error"``). A single body alone
from the start runs to the time limit.

Dynamical friction
------------------

The drag is Chandrasekhar's formula for a static, isotropic Maxwellian
background (Binney & Tremaine 2008, eq. 8.7):

.. code-block:: text

   a_df = -4 pi G^2 M rho(r) lnL / v^3 [erf(X) - 2X/sqrt(pi) exp(-X^2)] v,
   X = v / (sqrt(2) sigma(r)),

with tabulated profiles ``rho(r)`` and ``sigma(r)`` about the potential centre
(log-log interpolation, held at the end values outside the table), built from
the potential (Gauss's theorem + isotropic Jeans equation), from simulation
particles, or constant. The Coulomb logarithm is constant or
``0.5 ln(1 + Lambda^2)`` with ``Lambda = r (v^2 + sigma^2) / (G M)``.

With ``mode="system"`` (default) the bodies are treated as one object: the
drag on their combined mass, moving with their centre-of-mass velocity at
their centre of mass, is distributed among them in proportion to their masses.
Every body feels the same deceleration, which slows the group without
disturbing its internal orbits. With ``mode="individual"`` each body feels the
drag of its own mass and velocity.

At low speed the drag is linear, ``a = -k v`` with ``k ~ G^2 M rho /
sigma^3``. A background that is dense and cold where the bodies are (e.g. the
Jeans dispersion of a stellar cusp that ignores the black holes at its centre)
makes ``1/k`` tiny and the equations stiff; ``integrate`` warns when that
happens. Pass the bodies' mass as ``central_mass`` to
:meth:`~asmodean.DynamicalFriction.from_potential` so the dispersion rises as
``sqrt(G M / r)`` near them.

Energy bookkeeping
------------------

Every sample records the total energy ``E`` of the active bodies (kinetic +
external + softened pairwise), the cumulative energy ``E_removed`` carried off
by events (``E`` just before minus just after each merger or ejection) and the
friction work ``W``, integrated alongside the orbits. ``E + E_removed - W`` is
conserved to integration accuracy; ``result.energy_error()`` reports its
relative drift. Close encounters reach kinetic energies far above ``|E_0|``,
so a given per-step accuracy appears amplified by ``KE_peri / |E_0|``.

Chaos
-----

Few-body scattering is chaotic: individual outcomes are sensitive to every
numerical detail, including the tolerances and the output sampling (which sets
where steps are shortened). As with any few-body integrator, draw conclusions
from ensembles (:func:`~asmodean.integrate_ensemble`), not single runs.
