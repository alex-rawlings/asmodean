Usage
=====

This page walks through one experiment from python. Everything is in physical
units -- by default Gadget's kpc, 1e10 Msun and km/s (time unit kpc/(km/s)
~ 0.978 Gyr), with ``G = am.DEFAULT_G``. See :doc:`running` for the
equivalent driver scripts and :doc:`physics` for what the integrator does.

Quickstart
----------

.. code-block:: python

   import lanfear as lf
   import asmodean as am

   # 1. The background potential, built by lanfear. The figure never rotates.
   ps = lf.ParticleSystem.from_gadget_hdf5("snapshot.hdf5")
   ps.prepare(pattern_speed="none")
   lpot = lf.Potential.from_particles(ps, n_max=12, l_max=4)
   #   (or lf.DiscPotential / lf.MultiComponentPotential -- any lanfear potential)
   pot = am.ExternalPotential.from_lanfear(lpot, include_black_holes=False)

   # 2. The bodies (at most 10). Softening is Gadget's Plummer-equivalent epsilon.
   bodies = am.BodySystem()
   bodies.add_binary([0.01, 0.01], semimajor_axis=0.005, eccentricity=0.3,
                     softening=1e-5, labels=["BH1", "BH2"])
   bodies.add_incoming_body(0.005, v_infinity=50.0, impact_parameter=0.003,
                            distance=0.1, softening=1e-5, label="BH3")
   #   or: bodies = am.BodySystem.from_particles(ps.black_holes, softening=1e-5)

   # 3. Optional dynamical friction from the same potential (isotropic Jeans);
   #    the bodies' own mass at the centre keeps the dispersion realistic.
   friction = am.DynamicalFriction.from_potential(
       pot, r_min=1e-4, r_max=50.0, central_mass=bodies.total_mass
   )

   # 4. Integrate: 2000 periods of the binary, merging bodies closer than
   #    1e-5 and removing bodies that escape beyond 0.05 from the others.
   result = am.integrate(
       bodies, pot, friction=friction,
       n_periods=2000, period="binary",
       collision_distance=1e-5, ejection_radius=0.05,
   )
   print(result.summary())

   # 5. Analyse and save.
   outcome = am.classify_outcome(result)     # merger / flyby / exchange / ionisation
   print(outcome.describe(result.labels))
   for pair in am.bound_pairs(result):       # final binaries, most bound first
       print(pair.bodies, pair.semimajor_axis, pair.eccentricity)
   result.plot_trajectories(plane="all", frame="centre_of_mass")  # x-y, x-z, y-z,
                                             # coloured by time (one shared scale)
   result.save("result.npz")                 # am.ScatteringResult.load(...)

Choosing the time limit
-----------------------

The integration runs for ``n_periods`` reference periods, where ``period`` is
a physical time or is estimated from the initial conditions
(:func:`asmodean.estimate_period`):

* ``"internal"`` (default) -- Keplerian period of all the bodies' mass at their
  mass-weighted RMS radius about their centre of mass;
* ``"binary"`` -- Keplerian period of the most tightly bound pair;
* ``"circular"`` -- circular period of the external potential at the bodies'
  mass-weighted mean distance from its centre.

It stops earlier if every body has merged into one, or if at most one body is
left after an ejection (see :doc:`physics`). ``result.stop_reason`` says which.

Reading a result
----------------

A :class:`~asmodean.ScatteringResult` holds uniformly sampled trajectories
(``times``, ``positions``, ``velocities``, ``masses``; NaN once a body has
merged away or been ejected), the :class:`~asmodean.Event` list
(``result.mergers``, ``result.ejections``), each body's ``status`` and final
state, and the energy bookkeeping. ``result.energy_error()`` is the relative
error of the conserved energy budget, the check that the integration was
accurate.

.. code-block:: python

   result.separation("BH1", "BH2")            # (n_samples,)
   result.centre_of_mass()                    # CoM of the active bodies
   am.pair_elements(result, "BH1", "BH2")     # orbital elements at every sample
   result.plot_trajectories(plane="xz")          # one projection
   result.plot_trajectories(plane=["xy", "yz"])  # a chosen set of projections
   result.plot_trajectories(colour_by_time=False)  # one colour per body instead
   result.plot_separations()
   result.plot_energy()

By default trajectories are coloured by time on a single colour scale (first
to last sample) shared by every body and panel, with one colourbar; bodies are
distinguished by the marker at their starting point.

Ensembles
---------

Scattering outcomes are chaotic, so conclusions come from many realisations.
:func:`asmodean.integrate_ensemble` integrates a list of systems (same number
of bodies) in parallel with OpenMP; ``record_trajectory=False`` keeps only the
initial and final states:

.. code-block:: python

   import numpy as np
   from scipy.spatial.transform import Rotation

   rng = np.random.default_rng(1)
   systems = []
   for k in range(1000):
       s = am.BodySystem()
       s.add_binary([0.01, 0.01], 0.005, eccentricity=0.3, softening=1e-5,
                    true_anomaly=rng.uniform(0, 2 * np.pi))
       s.add_incoming_body(0.005, 50.0, rng.uniform(0, 0.01), 0.1,
                           rotation=Rotation.random(random_state=k).as_matrix(),
                           softening=1e-5)
       systems.append(s)
   results = am.integrate_ensemble(systems, pot, n_periods=2000, period="binary",
                                   ejection_radius=0.05, record_trajectory=False)
   kinds = [am.classify_outcome(r).kind for r in results]

Set ``OMP_NUM_THREADS`` to control the number of threads.
