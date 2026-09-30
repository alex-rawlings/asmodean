#include <array>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "asmodean/dynamical_friction.hpp"
#include "asmodean/external_potential.hpp"
#include "asmodean/integrator.hpp"
#include "asmodean/nbody.hpp"
#include "asmodean/softening.hpp"

namespace py = pybind11;
using asmodean::DynamicalFriction;
using asmodean::ExternalPotential;
using asmodean::IntegrationSettings;

namespace {

using CArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<double> to_vector(const py::handle& obj) {
    const CArray arr = py::cast<CArray>(obj);
    return std::vector<double>(arr.data(), arr.data() + arr.size());
}

// --- ExternalPotential from a python spec -----------------------------------
//
// The spec is a dict describing a lanfear potential (see
// asmodean.ExternalPotential, which builds it from a lanfear potential's pickled
// state):
//   {"kind": "none"}
//   {"kind": "scf", "n_max", "l_max", "coefficients_cos", "coefficients_sin",
//    "black_holes"}
//   {"kind": "disc", "a", "b", "coefficients", "black_holes"}
//   {"kind": "composite", "components": [{"potential": <scf|disc spec>,
//    "coord_scale", "phi_weight", "acc_weight", "label"}, ...], "black_holes"}
// "black_holes" is a (k, 5) array of (mass, x, y, z, softening) rows, in the
// potential's internal units (lanfear's convention).

template <class Pot>
void add_black_holes(Pot& pot, const py::dict& spec) {
    if (!spec.contains("black_holes")) return;
    const CArray bh = py::cast<CArray>(spec["black_holes"]);
    if (bh.size() == 0) return;
    if (bh.ndim() != 2 || bh.shape(1) != 5)
        throw std::runtime_error("black_holes must have shape (k, 5)");
    auto b = bh.unchecked<2>();
    for (py::ssize_t i = 0; i < bh.shape(0); ++i)
        pot.add_black_hole(b(i, 0), b(i, 1), b(i, 2), b(i, 3), b(i, 4));
}

lanfear::SCFPotential scf_from_spec(const py::dict& spec) {
    lanfear::SCFPotential pot(spec["n_max"].cast<int>(), spec["l_max"].cast<int>(),
                              to_vector(spec["coefficients_cos"]),
                              to_vector(spec["coefficients_sin"]));
    add_black_holes(pot, spec);
    return pot;
}

lanfear::DiscPotential disc_from_spec(const py::dict& spec) {
    lanfear::DiscPotential pot(to_vector(spec["a"]), to_vector(spec["b"]));
    pot.set_coefficients(to_vector(spec["coefficients"]));
    add_black_holes(pot, spec);
    return pot;
}

lanfear::CompositePotential composite_from_spec(const py::dict& spec) {
    std::vector<lanfear::CompositeComponent> comps;
    for (auto item : spec["components"].cast<py::list>()) {
        const py::dict c = item.cast<py::dict>();
        const py::dict sub = c["potential"].cast<py::dict>();
        const std::string kind = sub["kind"].cast<std::string>();
        const double coord_scale = c["coord_scale"].cast<double>();
        const double phi_weight = c["phi_weight"].cast<double>();
        const double acc_weight = c["acc_weight"].cast<double>();
        const std::string label = c["label"].cast<std::string>();
        if (kind == "scf") {
            comps.push_back({lanfear::PotentialVariant(scf_from_spec(sub)), coord_scale,
                             phi_weight, acc_weight, label});
        } else if (kind == "disc") {
            comps.push_back({lanfear::PotentialVariant(disc_from_spec(sub)),
                             coord_scale, phi_weight, acc_weight, label});
        } else {
            throw std::runtime_error("composite components must be 'scf' or 'disc', got '" +
                                     kind + "'");
        }
    }
    lanfear::CompositePotential pot(std::move(comps));
    add_black_holes(pot, spec);
    return pot;
}

ExternalPotential potential_from_spec(const py::dict& spec) {
    const std::string kind = spec["kind"].cast<std::string>();
    if (kind == "none") return ExternalPotential();
    if (kind == "scf") return ExternalPotential(scf_from_spec(spec));
    if (kind == "disc") return ExternalPotential(disc_from_spec(spec));
    if (kind == "composite") return ExternalPotential(composite_from_spec(spec));
    throw std::runtime_error("unknown potential kind '" + kind + "'");
}

// --- helpers ------------------------------------------------------------------

void check_points(const CArray& pts) {
    if (pts.ndim() != 2 || pts.shape(1) != 3)
        throw std::runtime_error("points must have shape (N, 3)");
}

// (n,) masses, (n, 6) states, (n,) spline softening lengths -> flat vectors.
void unpack_bodies(const CArray& masses, const CArray& states, const CArray& softening,
                   std::vector<double>& m, std::vector<double>& x,
                   std::vector<double>& h) {
    if (masses.ndim() != 1)
        throw std::runtime_error("masses must have shape (n,)");
    const py::ssize_t n = masses.shape(0);
    if (states.ndim() != 2 || states.shape(0) != n || states.shape(1) != 6)
        throw std::runtime_error("states must have shape (n, 6)");
    if (softening.ndim() != 1 || softening.shape(0) != n)
        throw std::runtime_error("softening must have shape (n,)");
    m.assign(masses.data(), masses.data() + n);
    x.assign(states.data(), states.data() + 6 * n);
    h.assign(softening.data(), softening.data() + n);
}

asmodean::ActiveBodies make_active(const std::vector<double>& m,
                                  const std::vector<double>& h) {
    asmodean::ActiveBodies bodies;
    for (std::size_t i = 0; i < m.size(); ++i)
        bodies.push_back(static_cast<int>(i), m[i], h[i]);
    return bodies;
}

template <class T>
py::array_t<double> as_array(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
    py::array_t<double> out(shape);
    double* p = out.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = static_cast<double>(v[i]);
    return out;
}

py::dict result_to_dict(const asmodean::ScatteringResult& r) {
    const py::ssize_t n = r.n_bodies;
    const py::ssize_t n_out = static_cast<py::ssize_t>(r.times.size());
    const py::ssize_t n_ev = static_cast<py::ssize_t>(r.events.size());
    py::dict d;
    d["stop_reason"] = r.stop_reason;
    d["t_end"] = r.t_end;
    d["n_steps"] = r.n_steps;
    d["n_failed_steps"] = r.n_failed_steps;
    d["times"] = as_array(r.times, {n_out});
    d["states"] = as_array(r.states, {n_out, n, 6});
    d["masses"] = as_array(r.masses, {n_out, n});
    d["energy"] = as_array(r.energy, {n_out});
    d["energy_removed"] = as_array(r.energy_removed, {n_out});
    d["friction_work"] = as_array(r.friction_work, {n_out});

    py::array_t<double> ev_time(n_ev), ev_mass(n_ev), ev_value(n_ev);
    py::array_t<int> ev_type(n_ev), ev_body(n_ev), ev_partner(n_ev);
    py::array_t<double> ev_state({n_ev, static_cast<py::ssize_t>(6)});
    for (py::ssize_t i = 0; i < n_ev; ++i) {
        const auto& e = r.events[static_cast<std::size_t>(i)];
        ev_time.mutable_data()[i] = e.time;
        ev_type.mutable_data()[i] = e.type;
        ev_body.mutable_data()[i] = e.body;
        ev_partner.mutable_data()[i] = e.partner;
        ev_mass.mutable_data()[i] = e.mass;
        ev_value.mutable_data()[i] = e.value;
        for (int c = 0; c < 6; ++c) ev_state.mutable_data()[6 * i + c] = e.state[c];
    }
    d["event_time"] = ev_time;
    d["event_type"] = ev_type;
    d["event_body"] = ev_body;
    d["event_partner"] = ev_partner;
    d["event_mass"] = ev_mass;
    d["event_state"] = ev_state;
    d["event_value"] = ev_value;

    d["status"] = py::array_t<int>(static_cast<py::ssize_t>(r.status.size()),
                                   r.status.data());
    d["merged_into"] = py::array_t<int>(static_cast<py::ssize_t>(r.merged_into.size()),
                                        r.merged_into.data());
    d["final_time"] = as_array(r.final_time, {n});
    d["final_mass"] = as_array(r.final_mass, {n});
    d["final_state"] = as_array(r.final_state, {n, 6});
    return d;
}

py::dict integrate_py(const ExternalPotential& pot, CArray masses, CArray states,
                      CArray softening, const IntegrationSettings& settings,
                      const DynamicalFriction& friction) {
    std::vector<double> m, x, h;
    unpack_bodies(masses, states, softening, m, x, h);
    asmodean::ScatteringResult r;
    {
        py::gil_scoped_release release;
        r = asmodean::integrate_system(pot, friction, settings, m, x, h);
    }
    return result_to_dict(r);
}

py::list integrate_batch_py(const ExternalPotential& pot, CArray masses, CArray states,
                            CArray softening, CArray t_max,
                            const IntegrationSettings& settings,
                            const DynamicalFriction& friction, bool progress) {
    if (masses.ndim() != 2)
        throw std::runtime_error("masses must have shape (n_systems, n_bodies)");
    const py::ssize_t n_sys = masses.shape(0);
    const py::ssize_t n = masses.shape(1);
    if (n < 1 || n > asmodean::kMaxBodies)
        throw std::runtime_error("number of bodies must be between 1 and " +
                                 std::to_string(asmodean::kMaxBodies));
    if (states.ndim() != 3 || states.shape(0) != n_sys || states.shape(1) != n ||
        states.shape(2) != 6)
        throw std::runtime_error("states must have shape (n_systems, n_bodies, 6)");
    if (softening.ndim() != 2 || softening.shape(0) != n_sys || softening.shape(1) != n)
        throw std::runtime_error("softening must have shape (n_systems, n_bodies)");
    if (t_max.ndim() != 1 || t_max.shape(0) != n_sys)
        throw std::runtime_error("t_max must have shape (n_systems,)");
    std::vector<asmodean::ScatteringResult> results;
    {
        py::gil_scoped_release release;
        results = asmodean::integrate_batch(
            pot, friction, settings, static_cast<std::size_t>(n_sys),
            static_cast<int>(n), masses.data(), states.data(), softening.data(),
            t_max.data(), progress);
    }
    py::list out;
    for (const auto& r : results) out.append(result_to_dict(r));
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "asmodean C++ core: softened few-body scattering in a static lanfear "
        "analytical potential, with mergers, ejections and dynamical friction.";

    m.attr("MAX_BODIES") = asmodean::kMaxBodies;
    m.attr("SPLINE_TO_PLUMMER") = asmodean::kSplineToPlummer;
    m.attr("STOP_REASONS") = std::vector<std::string>{
        "time_limit", "all_merged", "bodies_ejected", "step_limit",
        "integration_error"};
    m.attr("EVENT_TYPES") = std::vector<std::string>{"merger", "ejection"};
    m.attr("BODY_STATUSES") = std::vector<std::string>{"active", "merged", "ejected"};

    // --- ExternalPotential ---
    py::class_<ExternalPotential>(m, "ExternalPotential")
        .def(py::init(&potential_from_spec), py::arg("spec"),
             "Build from a spec dict (see asmodean.ExternalPotential).")
        .def(py::init<>(), "No potential (pure few-body problem).")
        .def_property_readonly("kind", &ExternalPotential::kind)
        .def_property_readonly("is_null", &ExternalPotential::is_null)
        .def_property_readonly("num_black_holes", &ExternalPotential::num_black_holes)
        .def("potential",
             [](const ExternalPotential& p, double x, double y, double z) {
                 return p.potential(x, y, z);
             },
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Potential at one point (internal units).")
        .def("acceleration",
             [](const ExternalPotential& p, double x, double y, double z) {
                 return p.acceleration(x, y, z);
             },
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Acceleration at one point (internal units).")
        .def("potential_batch",
             [](const ExternalPotential& p, CArray pts) {
                 check_points(pts);
                 const py::ssize_t n = pts.shape(0);
                 auto q = pts.unchecked<2>();
                 py::array_t<double> out(n);
                 auto o = out.mutable_unchecked<1>();
                 {
                     py::gil_scoped_release release;
                     #pragma omp parallel for schedule(dynamic, 256)
                     for (py::ssize_t i = 0; i < n; ++i)
                         o(i) = p.potential(q(i, 0), q(i, 1), q(i, 2));
                 }
                 return out;
             },
             py::arg("points"), "Potential at points (N, 3) -> (N,).")
        .def("acceleration_batch",
             [](const ExternalPotential& p, CArray pts) {
                 check_points(pts);
                 const py::ssize_t n = pts.shape(0);
                 auto q = pts.unchecked<2>();
                 py::array_t<double> out({n, static_cast<py::ssize_t>(3)});
                 auto o = out.mutable_unchecked<2>();
                 {
                     py::gil_scoped_release release;
                     #pragma omp parallel for schedule(dynamic, 256)
                     for (py::ssize_t i = 0; i < n; ++i) {
                         const auto a = p.acceleration(q(i, 0), q(i, 1), q(i, 2));
                         o(i, 0) = a[0];
                         o(i, 1) = a[1];
                         o(i, 2) = a[2];
                     }
                 }
                 return out;
             },
             py::arg("points"), "Acceleration at points (N, 3) -> (N, 3).");

    // --- DynamicalFriction ---
    py::class_<DynamicalFriction>(m, "DynamicalFriction")
        .def(py::init<>(), "Disabled dynamical friction.")
        .def(py::init([](CArray radius, CArray density, CArray dispersion,
                         double coulomb_logarithm, bool variable_coulomb_logarithm,
                         int mode) {
                 if (mode != 0 && mode != 1)
                     throw std::runtime_error("mode must be 0 (system) or 1 (individual)");
                 return DynamicalFriction(
                     to_vector(radius), to_vector(density), to_vector(dispersion),
                     coulomb_logarithm, variable_coulomb_logarithm,
                     static_cast<asmodean::FrictionMode>(mode));
             }),
             py::arg("radius"), py::arg("density"), py::arg("dispersion"),
             py::arg("coulomb_logarithm"), py::arg("variable_coulomb_logarithm"),
             py::arg("mode"),
             "Chandrasekhar friction from tabulated density/dispersion profiles "
             "(internal units). mode: 0 system (shared in proportion to mass), "
             "1 individual.")
        .def_property_readonly("enabled", &DynamicalFriction::enabled)
        .def_property_readonly("mode",
                               [](const DynamicalFriction& f) {
                                   return static_cast<int>(f.mode());
                               })
        .def("density",
             [](const DynamicalFriction& f, CArray r) {
                 return py::vectorize([&f](double x) { return f.density(x); })(r);
             },
             py::arg("r"))
        .def("dispersion",
             [](const DynamicalFriction& f, CArray r) {
                 return py::vectorize([&f](double x) { return f.dispersion(x); })(r);
             },
             py::arg("r"))
        .def("acceleration",
             [](const DynamicalFriction& f, double mass, const asmodean::Vec3& pos,
                const asmodean::Vec3& vel) { return f.acceleration(mass, pos, vel); },
             py::arg("mass"), py::arg("position"), py::arg("velocity"),
             "Chandrasekhar deceleration of one mass (internal units).");

    // --- IntegrationSettings ---
    py::class_<IntegrationSettings>(m, "IntegrationSettings")
        .def(py::init<>())
        .def_readwrite("t_max", &IntegrationSettings::t_max)
        .def_readwrite("n_samples", &IntegrationSettings::n_samples)
        .def_readwrite("abs_tol", &IntegrationSettings::abs_tol)
        .def_readwrite("rel_tol", &IntegrationSettings::rel_tol)
        .def_readwrite("collision_distance", &IntegrationSettings::collision_distance)
        .def_readwrite("ejection_radius", &IntegrationSettings::ejection_radius)
        .def_readwrite("ejection_reference", &IntegrationSettings::ejection_reference)
        .def_readwrite("step_factor", &IntegrationSettings::step_factor)
        .def_readwrite("kernel_step_factor", &IntegrationSettings::kernel_step_factor)
        .def_readwrite("max_steps", &IntegrationSettings::max_steps)
        .def_readwrite("stepper", &IntegrationSettings::stepper)
        .def_readwrite("record_trajectory", &IntegrationSettings::record_trajectory);

    // --- integration ---
    m.def("integrate", &integrate_py, py::arg("potential"), py::arg("masses"),
          py::arg("states"), py::arg("softening"), py::arg("settings"),
          py::arg("friction"),
          "Integrate one experiment. masses (n,), states (n, 6), softening (n,) "
          "spline-kernel lengths, all in the potential's internal units. Returns "
          "a dict of the samples, events and per-body outcome. GIL released.");
    m.def("integrate_batch", &integrate_batch_py, py::arg("potential"),
          py::arg("masses"), py::arg("states"), py::arg("softening"), py::arg("t_max"),
          py::arg("settings"), py::arg("friction"), py::arg("progress") = false,
          "Integrate independent experiments (OpenMP over experiments): masses "
          "(B, n), states (B, n, 6), softening (B, n), t_max (B,). Returns a list "
          "of result dicts. Set progress=True to print '<X>% of experiments "
          "integrated' every 10%.");

    m.def("accelerations",
          [](const ExternalPotential& pot, CArray masses, CArray states,
             CArray softening, const DynamicalFriction& friction) {
              std::vector<double> mm, x, h;
              unpack_bodies(masses, states, softening, mm, x, h);
              const auto bodies = make_active(mm, h);
              std::vector<double> s(x);
              s.push_back(0.0);
              std::vector<double> acc(3 * mm.size());
              asmodean::compute_accelerations(pot, friction, bodies, s.data(), acc.data());
              return as_array(acc, {static_cast<py::ssize_t>(mm.size()), 3});
          },
          py::arg("potential"), py::arg("masses"), py::arg("states"),
          py::arg("softening"), py::arg("friction"),
          "Accelerations (n, 3) of n bodies (internal units, spline softening).");
    m.def("total_energy",
          [](const ExternalPotential& pot, CArray masses, CArray states,
             CArray softening) {
              std::vector<double> mm, x, h;
              unpack_bodies(masses, states, softening, mm, x, h);
              const auto bodies = make_active(mm, h);
              return asmodean::total_energy(pot, bodies, x.data());
          },
          py::arg("potential"), py::arg("masses"), py::arg("states"),
          py::arg("softening"),
          "Kinetic + external + softened pairwise energy (internal units).");
    m.def("spline_potential", &asmodean::spline_softened_potential, py::arg("r"),
          py::arg("h"), "Gadget spline-softened potential of a unit mass.");
    m.def("spline_force_factor", &asmodean::spline_softened_force_factor,
          py::arg("r"), py::arg("h"),
          "Gadget spline-softened force factor (a = -factor * dx) of a unit mass.");
}
