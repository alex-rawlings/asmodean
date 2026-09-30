#pragma once

// Softened few-body equations of motion in a static analytical potential, with
// optional dynamical friction.
//
// Only the *active* bodies (not yet merged away or ejected) are integrated. Their
// phase-space coordinates are packed into one flat State vector,
//
//   s = (x_0, y_0, z_0, vx_0, vy_0, vz_0, x_1, ..., vz_{n-1}, W),
//
// where body k occupies s[6k .. 6k+5] and the final element W is the work done
// on the bodies by dynamical friction, integrated alongside the orbits
// (dW/dt = sum_k m_k a_df,k . v_k). Carrying W makes the energy budget exact:
// E(t) - W(t) is conserved to integration accuracy whether or not friction is
// on. ActiveBodies maps each slot back to the original body index.
//
// The acceleration of body i is
//
//   a_i = a_ext(x_i) - sum_{j != i} m_j g(r_ij, h_ij) (x_i - x_j) + a_df,i
//
// with the Gadget spline-softened pair force g (see softening.hpp) and G = 1.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "common.hpp"
#include "dynamical_friction.hpp"
#include "external_potential.hpp"
#include "softening.hpp"

namespace asmodean {

using State = std::vector<double>;

struct ActiveBodies {
    std::vector<int> index;         // original body index of each slot
    std::vector<double> mass;       // mass of each slot (G = 1 units)
    std::vector<double> softening;  // spline-kernel length h of each slot

    std::size_t size() const { return index.size(); }
    std::size_t state_size() const { return 6 * size() + 1; }
    void clear() {
        index.clear();
        mass.clear();
        softening.clear();
    }
    void push_back(int i, double m, double h) {
        index.push_back(i);
        mass.push_back(m);
        softening.push_back(h);
    }
};

// Accelerations (acc[3k + c]) of every active body in state `s`. Returns the
// power sum_k m_k a_df,k . v_k delivered by dynamical friction.
inline double compute_accelerations(const ExternalPotential& pot,
                                    const DynamicalFriction& friction,
                                    const ActiveBodies& bodies, const double* s,
                                    double* acc) {
    const std::size_t n = bodies.size();
    for (std::size_t k = 0; k < n; ++k) {
        const double* x = s + 6 * k;
        const auto a = pot.acceleration(x[0], x[1], x[2]);
        acc[3 * k] = a[0];
        acc[3 * k + 1] = a[1];
        acc[3 * k + 2] = a[2];
    }
    for (std::size_t i = 0; i < n; ++i) {
        const double* xi = s + 6 * i;
        for (std::size_t j = i + 1; j < n; ++j) {
            const double* xj = s + 6 * j;
            const double dx = xi[0] - xj[0];
            const double dy = xi[1] - xj[1];
            const double dz = xi[2] - xj[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double g = spline_softened_force_factor(
                r, pair_softening(bodies.softening[i], bodies.softening[j]));
            const double gi = bodies.mass[j] * g;
            const double gj = bodies.mass[i] * g;
            acc[3 * i] -= gi * dx;
            acc[3 * i + 1] -= gi * dy;
            acc[3 * i + 2] -= gi * dz;
            acc[3 * j] += gj * dx;
            acc[3 * j + 1] += gj * dy;
            acc[3 * j + 2] += gj * dz;
        }
    }

    if (!friction.enabled() || n == 0) return 0.0;
    double power = 0.0;
    if (friction.mode() == FrictionMode::kSystem) {
        // One drag force on the combined mass at the centre of mass, shared in
        // proportion to mass -> the same deceleration for every body.
        double m_tot = 0.0;
        Vec3 x_com{0.0, 0.0, 0.0}, v_com{0.0, 0.0, 0.0};
        for (std::size_t k = 0; k < n; ++k) {
            const double m = bodies.mass[k];
            m_tot += m;
            for (int c = 0; c < 3; ++c) {
                x_com[c] += m * s[6 * k + c];
                v_com[c] += m * s[6 * k + 3 + c];
            }
        }
        for (int c = 0; c < 3; ++c) {
            x_com[c] /= m_tot;
            v_com[c] /= m_tot;
        }
        const Vec3 a = friction.acceleration(m_tot, x_com, v_com);
        for (std::size_t k = 0; k < n; ++k)
            for (int c = 0; c < 3; ++c) acc[3 * k + c] += a[c];
        power = m_tot * dot(a, v_com);
    } else {
        for (std::size_t k = 0; k < n; ++k) {
            const double* x = s + 6 * k;
            const Vec3 v{x[3], x[4], x[5]};
            const Vec3 a = friction.acceleration(bodies.mass[k], {x[0], x[1], x[2]}, v);
            for (int c = 0; c < 3; ++c) acc[3 * k + c] += a[c];
            power += bodies.mass[k] * dot(a, v);
        }
    }
    return power;
}

// Right-hand side ds/dt for odeint. Freezes on NaN so the stepper cannot spin
// on a diverged state; the driver checks nan_hit.
struct EquationsOfMotion {
    const ExternalPotential& pot;
    const DynamicalFriction& friction;
    const ActiveBodies& bodies;
    bool nan_hit = false;
    std::vector<double> acc;

    void operator()(const State& s, State& dsdt, double /*t*/) {
        const std::size_t n = bodies.size();
        for (std::size_t k = 0; k < 6 * n; ++k) {
            if (std::isnan(s[k])) {
                nan_hit = true;
                std::fill(dsdt.begin(), dsdt.end(), 0.0);
                return;
            }
        }
        acc.resize(3 * n);
        const double power =
            compute_accelerations(pot, friction, bodies, s.data(), acc.data());
        for (std::size_t k = 0; k < n; ++k) {
            for (int c = 0; c < 3; ++c) {
                dsdt[6 * k + c] = s[6 * k + 3 + c];
                dsdt[6 * k + 3 + c] = acc[3 * k + c];
            }
        }
        dsdt[6 * n] = power;
    }
};

// Softened pair potential energy of slots i and j (G = 1).
inline double pair_energy(const ActiveBodies& bodies, const double* s,
                          std::size_t i, std::size_t j) {
    const double* xi = s + 6 * i;
    const double* xj = s + 6 * j;
    const double dx = xi[0] - xj[0];
    const double dy = xi[1] - xj[1];
    const double dz = xi[2] - xj[2];
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    return bodies.mass[i] * bodies.mass[j] *
           spline_softened_potential(
               r, pair_softening(bodies.softening[i], bodies.softening[j]));
}

// Total energy of the active bodies: kinetic + external potential + softened
// pairwise potential energy.
inline double total_energy(const ExternalPotential& pot, const ActiveBodies& bodies,
                           const double* s) {
    const std::size_t n = bodies.size();
    double e = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double* x = s + 6 * k;
        const double v2 = x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
        e += bodies.mass[k] * (0.5 * v2 + pot.potential(x[0], x[1], x[2]));
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) e += pair_energy(bodies, s, i, j);
    return e;
}

// Specific energy of slot k in the field of the potential and every other
// active body: 0.5 v^2 + Phi_ext(x) + sum_j m_j phi(r_kj).
inline double specific_energy(const ExternalPotential& pot, const ActiveBodies& bodies,
                              const double* s, std::size_t k) {
    const double* x = s + 6 * k;
    const double v2 = x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
    double e = 0.5 * v2 + pot.potential(x[0], x[1], x[2]);
    for (std::size_t j = 0; j < bodies.size(); ++j)
        if (j != k) e += pair_energy(bodies, s, k, j) / bodies.mass[k];
    return e;
}

}  // namespace asmodean
