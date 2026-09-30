// Standalone C++ sanity check (no python): an equal-mass Keplerian binary with
// no background potential. In G = 1 units with total mass M = 2 and semimajor
// axis a = 1 the period is T = 2 pi sqrt(a^3 / M). After an integer number of
// periods the binary must return to its initial state, and the energy must be
// conserved. Then two bodies falling together from rest must merge.

#include <cmath>
#include <cstdio>
#include <vector>

#include "asmodean/integrator.hpp"

int main() {
    const double e = 0.5;
    const double m = 1.0;
    const double a = 1.0;
    const double M = 2.0 * m;
    const double T = 2.0 * M_PI * std::sqrt(a * a * a / M);

    // Relative orbit at pericentre, split about the centre of mass.
    const double r_p = a * (1.0 - e);
    const double v_p = std::sqrt(M / a * (1.0 + e) / (1.0 - e));
    std::vector<double> mass{m, m};
    std::vector<double> states{0.5 * r_p,  0.0, 0.0, 0.0, 0.5 * v_p,  0.0,
                               -0.5 * r_p, 0.0, 0.0, 0.0, -0.5 * v_p, 0.0};
    std::vector<double> softening{0.0, 0.0};

    asmodean::ExternalPotential pot;
    asmodean::DynamicalFriction friction;
    asmodean::IntegrationSettings settings;
    const int n_orbits = 20;
    settings.t_max = n_orbits * T;
    settings.n_samples = n_orbits + 1;

    const auto res =
        asmodean::integrate_system(pot, friction, settings, mass, states, softening);
    const std::size_t last = res.times.size() - 1;
    double pos_err = 0.0;
    for (int c = 0; c < 6; ++c)
        pos_err = std::max(pos_err, std::abs(res.states[last * 12 + c] - states[c]));
    const double e_err = std::abs((res.energy[last] - res.energy[0]) / res.energy[0]);
    printf("Kepler binary e=%.1f, %d periods: %ld steps, |dstate|=%.2e, |dE/E|=%.2e\n",
           e, n_orbits, res.n_steps, pos_err, e_err);

    // Radial infall from rest at separation d0 to contact at d_c: for two
    // point masses the time is sqrt(d0^3 / (2 M)) [sqrt(x(1-x)) + acos(sqrt x)]
    // with x = d_c / d0.
    const double d0 = 1.0, d_c = 0.01;
    std::vector<double> fall{0.5 * d0, 0, 0, 0, 0, 0, -0.5 * d0, 0, 0, 0, 0, 0};
    settings.t_max = 10.0;
    settings.collision_distance = d_c;
    const auto res2 =
        asmodean::integrate_system(pot, friction, settings, mass, fall, softening);
    const double x = d_c / d0;
    const double t_exact =
        std::sqrt(d0 * d0 * d0 / (2.0 * M)) * (std::sqrt(x * (1 - x)) + std::acos(std::sqrt(x)));
    const double t_err = std::abs(res2.t_end - t_exact) / t_exact;
    printf("Radial infall: merged at t=%.10f (exact %.10f, rel err %.2e), stop=%s\n",
           res2.t_end, t_exact, t_err, asmodean::stop_reason_name(res2.stop_reason));

    const bool ok = pos_err < 1e-7 && e_err < 1e-10 && t_err < 1e-8 &&
                    res2.stop_reason == static_cast<int>(asmodean::StopReason::kAllMerged);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
