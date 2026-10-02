Running the driver scripts
==========================

Three scripts in ``scripts/`` cover the whole workflow without writing any
python (a fourth, ``impact_parameter_survey.py``, runs a survey of a
snapshot's black holes; see below):

.. code-block:: bash

   python scripts/make_system.py scripts/example_config.toml -o setup.npz
   python scripts/run_scattering.py setup.npz -o result.npz
   python scripts/analyse.py result.npz --figdir figures

``make_system.py``
   Reads a TOML configuration and builds the external potential, the bodies,
   the dynamical-friction model and the integration settings, saving them
   together as a :class:`~asmodean.ScatteringSetup` (``.npz``).

``run_scattering.py``
   Integrates a saved setup and writes the :class:`~asmodean.ScatteringResult`
   (``.npz``). Any integration setting can be overridden on the command line
   (``--n-periods``, ``--period``, ``--n-samples``, ``--collision-distance``,
   ``--ejection-radius``, ``--ejection-reference``,
   ``--hard-binary-dispersion``, ``--rel-tol``,
   ``--abs-tol``, ``--stepper``), and ``--no-friction`` switches friction off.

``analyse.py``
   Prints the outcome (see :func:`~asmodean.classify_outcome`), the final bound
   pairs with their orbital elements and the ejected bodies' escape speeds, and
   writes figures of the trajectories, pairwise separations, energy-budget
   error and the initial binary's orbital elements over time. Trajectories are
   projected on x-y, x-z and y-z (``--plane`` picks one instead) and coloured
   by time on one shared scale (``--no-time-colour`` colours by body);
   ``--frame centre_of_mass`` plots them relative to the bodies' centre of
   mass.

Impact-parameter survey of a snapshot's black holes
---------------------------------------------------

``impact_parameter_survey.py`` asks how the eccentricity of a forming massive
black hole binary depends on the pair's impact parameter:

.. code-block:: bash

   python scripts/impact_parameter_survey.py snap_030.hdf5 -o survey --potential-from stars
   python scripts/impact_parameter_survey.py --plot-only survey/survey.npz

It reads a Gadget-4 snapshot, prepares it with lanfear (centred with
``--centre``, static figure) and builds an SCF potential (``--n-max``,
``--l-max``) from every field particle (``--potential-from all``, default) or
from the stars only (``--potential-from stars``). The snapshot's two most
massive black holes (or ``--bh-ids``) are integrated as bodies (softening
``--bh-softening``), with dynamical friction from the potential's isotropic
Jeans profile (``--friction-mode``, default ``"individual"`` so the pair can
sink and bind; ``--coulomb-logarithm``; ``--friction-range``).

The impact parameter is that of the pair's relative orbit at the snapshot
time, ``b = |r x v| / |v|``. Each run rotates the relative velocity within the
orbital plane (keeping its magnitude and an approaching radial velocity) to
set ``b``, so the positions, energy and centre-of-mass motion stay those of
the snapshot. The impact parameters are ``--impact-parameters`` (physical
lengths) or ``--n-b`` values spanning ``--b-range`` (in units of the initial
separation, default 0 to 0.9). Only impact parameters deflecting the pair by at
least ``--min-deflection`` (default 30 degrees), ``b <= b90 / tan(theta_min /
2)`` with ``b90 = G (m_1 + m_2) / v^2``, are kept: the ``--n-b`` samples are
spread over that part of ``--b-range``, and explicit impact parameters beyond
it are dropped. All runs are integrated in parallel for
``--n-periods`` circular periods at the pair's initial radius, stopping when
the binary becomes hard; ``--hard-dispersion`` sets ``sigma`` in
``a_h = G mu / (4 sigma^2)``: a velocity, ``"influence"`` (default; the
dispersion of the potential's particles within the influence radius, which
encloses twice the pair's mass) or ``"friction"`` (local, from the friction
profile).

The output directory holds ``survey.npz`` (impact parameters, stop reasons,
and the time, semimajor axis and eccentricity at hardening, NaN where the
binary did not become hard), ``setup.npz`` (the unperturbed setup, for
``run_scattering.py``), ``runs/run_XXX.npz`` (each result, for ``analyse.py``)
and ``impact_parameter_eccentricity.png``, the eccentricity at hardening
against the two-body deflection angle ``2 arctan(b90 / b)``, where
``b90 = G (m_1 + m_2) / v^2`` for the pair's relative speed ``v`` at the
snapshot (the same in every run; stored in ``survey.npz``). The top axis gives
the impact parameter in units of the initial separation, and runs that did not
harden are marked along the bottom.

The configuration file
----------------------

``scripts/example_config.toml`` documents every option; it runs out of the box
(the potential is a lanfear fit to a synthetic Hernquist sphere). The tables
are:

``[units]``
   ``G``, the gravitational constant of the physical unit system.

``[potential]``
   ``type`` selects the source:

   * ``"lanfear"`` -- build a lanfear potential from a Gadget ``snapshot`` with
     ``basis = "scf"`` (``n_max``, ``l_max``), ``"disc"`` (``n_radial``,
     ``n_vert``) or ``"multi"`` (one ``[potential.components.<SPECIES>]`` table
     per species). The snapshot is prepared with lanfear's
     ``ParticleSystem.prepare(pattern_speed="none")``: the figure never rotates.
     ``include_black_holes = false`` drops the snapshot's black holes from the
     potential, and ``bodies_from_black_holes = true`` integrates them as bodies
     instead (softening ``bh_body_softening``).
   * ``"hernquist"`` -- a lanfear SCF fit to a random Hernquist sphere
     (``mass``, ``scale_radius``, ``n_particles``, ``n_max``, ``l_max``).
   * ``"file"`` -- a potential saved with :meth:`asmodean.ExternalPotential.save`.
   * ``"none"`` -- no background potential.

``[friction]``
   ``enabled``, ``mode`` (``"system"`` or ``"individual"``),
   ``coulomb_logarithm`` (a number or ``"variable"``) and the profile
   ``source``: ``"potential"`` (``r_min``, ``r_max``, ``n_bins``,
   ``central_mass`` -- a mass or ``"bodies"``), ``"particles"`` (binned from
   the lanfear snapshot) or ``"constant"`` (``density``, ``dispersion``).

``[[binary]]``, ``[[incoming]]``, ``[[body]]``
   Any number of each (at most 10 bodies in total), added in that order: a
   Keplerian binary from its orbital elements (angles in degrees), a body on a
   hyperbolic approach to ``target`` bodies (``v_infinity``,
   ``impact_parameter``, ``distance``, ``orientation_deg``), or a body with an
   explicit ``position`` and ``velocity``. ``softening`` is the Gadget
   Plummer-equivalent softening.

``[integration]``
   Any :class:`~asmodean.IntegrationSettings` field.

Progress and logging
--------------------

Pass ``-v`` to ``make_system.py``/``run_scattering.py`` (or call
``am.set_verbosity("INFO")`` from python) for progress messages. See
:doc:`logging`.
