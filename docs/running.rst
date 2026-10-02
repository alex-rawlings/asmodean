Running the driver scripts
==========================

Three scripts in ``scripts/`` cover the whole workflow without writing any
python:

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
