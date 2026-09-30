#pragma once

// Integration of one scattering experiment: at most kMaxBodies softened bodies
// in a static analytical potential, with optional dynamical friction, mergers
// and ejections.
//
// Stepping. An embedded Runge-Kutta pair (Fehlberg 7(8) by default, or
// Dormand-Prince 5(4)) with adaptive error control advances the packed state
// (nbody.hpp). Each step is additionally capped at
//
//   dt <= eta * min_{pairs} min(r_ij / |v_ij|, sqrt(r_ij^3 / (m_i + m_j))),
//
// (r_ij floored at the pair's softening length in the second term) so that no
// pair can cross a large fraction of its separation, or of its mutual orbit,
// in one step: close encounters, and so collisions, are not stepped over.
// While a pair is within 1.2 kernel lengths of each other the step is further
// capped at dt <= eta_h h_ij / |v_ij| (eta_h = kernel_step_factor). The spline
// force is a piecewise polynomial (breakpoints at r = h/2 and h), and a
// high-order embedded error estimate badly underestimates the error of steps
// that straddle a breakpoint; resolving the kernel explicitly keeps passages
// through it as accurate as the tolerance promises.
// Steps are also shortened to land exactly on the uniformly spaced output
// times, so recorded samples need no interpolation.
//
// Events, checked after every accepted step:
//
// * Merger: two bodies closer than `collision_distance` are lumped together.
//   The moment of contact is located by bisection within the step, and the
//   pair is replaced by one body carrying their total mass, centre-of-mass
//   position and velocity (so momentum is conserved exactly) and the larger
//   softening. The more massive body (lower index on a tie) survives; the
//   other is marked merged.
// * Ejection: a body that has escaped is removed and no longer integrated
//   (nor does it act on the others). With the kSystem reference a body is
//   ejected when it is beyond `ejection_radius` from the centre of mass of the
//   remaining bodies, receding from it, unbound from their combined mass, and
//   not bound to any single one of them (so a member of an escaping-looking
//   bound pair is kept). With the kCentre reference it must instead be beyond
//   `ejection_radius` from the potential centre, receding, with positive
//   specific energy in the potential plus the other bodies' field.
//
// Integration stops at the first of: the time limit t_max; every body merged
// into one; at most one body left after an ejection (no interactions remain);
// the step limit; or an integration error (non-finite state or step-size
// underflow).
//
// Energy bookkeeping. Each sample records the bodies' total energy E, the
// cumulative energy carried away by events (E just before minus E just after
// each merger/ejection) and the cumulative dynamical-friction work W, so that
// E + E_removed - W is conserved to integration accuracy.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/numeric/odeint.hpp>

#include "common.hpp"
#include "dynamical_friction.hpp"
#include "external_potential.hpp"
#include "nbody.hpp"
#include "softening.hpp"

namespace asmodean {

enum class StopReason : int {
    kTimeLimit = 0,
    kAllMerged = 1,
    kBodiesEjected = 2,
    kStepLimit = 3,
    kIntegrationError = 4,
};
enum class EventType : int { kMerger = 0, kEjection = 1 };
enum class BodyStatus : int { kActive = 0, kMerged = 1, kEjected = 2 };
enum class EjectionReference : int { kSystem = 0, kCentre = 1 };
enum class StepperType : int { kFehlberg78 = 0, kDormandPrince5 = 1 };

inline const char* stop_reason_name(int r) {
    static const char* const names[] = {"time_limit", "all_merged",
                                        "bodies_ejected", "step_limit",
                                        "integration_error"};
    return (r >= 0 && r < 5) ? names[r] : "unknown";
}

// Every length/time/mass here is in the potential's internal (G = 1) units.
struct IntegrationSettings {
    double t_max = 1.0;               // integration time limit
    int n_samples = 1000;             // uniformly spaced output samples in [0, t_max]
    double abs_tol = 1e-14;           // absolute error tolerance per step
    double rel_tol = 1e-11;           // relative error tolerance per step
    double collision_distance = 0.0;  // merger separation; <= 0 disables mergers
    double ejection_radius = 0.0;     // ejection distance; <= 0 disables ejections
    int ejection_reference = static_cast<int>(EjectionReference::kSystem);
    double step_factor = 0.1;         // eta in the encounter step cap
    double kernel_step_factor = 0.01; // eta_h in the softening-kernel step cap
    long max_steps = 100000000;       // accepted-step limit
    int stepper = static_cast<int>(StepperType::kFehlberg78);
    bool record_trajectory = true;    // false: record only the first/last states
};

struct Event {
    double time = 0.0;
    int type = 0;        // EventType
    int body = -1;       // merger: surviving body; ejection: ejected body
    int partner = -1;    // merger: absorbed body; ejection: -1
    double mass = 0.0;   // mass of `body` after the event
    std::array<double, 6> state{};  // merged body's / ejected body's state
    double value = 0.0;  // merger: relative speed; ejection: escape energy
};

struct ScatteringResult {
    int n_bodies = 0;
    int stop_reason = 0;
    double t_end = 0.0;
    long n_steps = 0;
    long n_failed_steps = 0;
    // Samples (n_out of them). Inactive bodies are NaN.
    std::vector<double> times;           // (n_out,)
    std::vector<double> states;          // (n_out, n_bodies, 6)
    std::vector<double> masses;          // (n_out, n_bodies)
    std::vector<double> energy;          // (n_out,) total energy of active bodies
    std::vector<double> energy_removed;  // (n_out,) cumulative, carried off by events
    std::vector<double> friction_work;   // (n_out,) cumulative work of friction
    std::vector<Event> events;
    // Per-body outcome. final_* hold the body's last state: at t_end if still
    // active, else at the moment it merged away or was ejected.
    std::vector<int> status;             // BodyStatus
    std::vector<int> merged_into;        // index absorbing a merged body, else -1
    std::vector<double> final_time;      // (n_bodies,)
    std::vector<double> final_mass;      // (n_bodies,)
    std::vector<double> final_state;     // (n_bodies, 6)
};

// Pairs closer than this many kernel lengths h get the kernel step cap.
constexpr double kKernelMargin = 1.2;

class ScatteringIntegrator {
public:
    // `states` is (n, 6) row-major; `softening` is the spline-kernel length h
    // of each body (h = 2.8 epsilon; zero for no softening).
    ScatteringIntegrator(const ExternalPotential& pot,
                         const DynamicalFriction& friction,
                         const IntegrationSettings& settings,
                         const std::vector<double>& mass,
                         const std::vector<double>& states,
                         const std::vector<double>& softening)
        : pot_(pot),
          friction_(friction),
          settings_(settings),
          n_(static_cast<int>(mass.size())),
          mass_(mass),
          soft_(softening) {
        if (n_ < 1 || n_ > kMaxBodies)
            throw std::invalid_argument("number of bodies must be between 1 and " +
                                        std::to_string(kMaxBodies));
        if (states.size() != 6 * mass.size() || softening.size() != mass.size())
            throw std::invalid_argument(
                "states must be (n, 6) and softening (n,) for n masses");
        body_state_.resize(n_);
        for (int b = 0; b < n_; ++b) {
            if (!(mass_[b] > 0.0) || !std::isfinite(mass_[b]))
                throw std::invalid_argument("body masses must be positive and finite");
            if (!(soft_[b] >= 0.0) || !std::isfinite(soft_[b]))
                throw std::invalid_argument("softening must be non-negative and finite");
            for (int c = 0; c < 6; ++c) {
                if (!std::isfinite(states[6 * b + c]))
                    throw std::invalid_argument("initial states must be finite");
                body_state_[b][c] = states[6 * b + c];
            }
        }
        if (!(settings_.t_max > 0.0) || !std::isfinite(settings_.t_max))
            throw std::invalid_argument("t_max must be positive and finite");
        if (!(settings_.abs_tol > 0.0) || !(settings_.rel_tol >= 0.0))
            throw std::invalid_argument("tolerances must be positive");
        if (!(settings_.step_factor > 0.0) || !(settings_.kernel_step_factor > 0.0))
            throw std::invalid_argument("step factors must be positive");
        status_.assign(n_, static_cast<int>(BodyStatus::kActive));
        merged_into_.assign(n_, -1);
        final_time_.assign(n_, 0.0);
        final_mass_ = mass_;
        final_state_ = body_state_;
        rebuild_active();
    }

    ScatteringResult run() {
        if (settings_.stepper == static_cast<int>(StepperType::kDormandPrince5))
            run_with<boost::numeric::odeint::runge_kutta_dopri5<State>>();
        else
            run_with<boost::numeric::odeint::runge_kutta_fehlberg78<State>>();
        return std::move(result_);
    }

private:
    const ExternalPotential& pot_;
    const DynamicalFriction& friction_;
    IntegrationSettings settings_;
    int n_;
    std::vector<double> mass_, soft_;
    std::vector<std::array<double, 6>> body_state_;
    std::vector<int> status_, merged_into_;
    std::vector<double> final_time_, final_mass_;
    std::vector<std::array<double, 6>> final_state_;
    ActiveBodies active_;
    double work_ = 0.0;
    double energy_removed_ = 0.0;
    int n_ejected_ = 0;
    bool unrecorded_event_ = false;  // an event happened since the last sample
    ScatteringResult result_;

    bool mergers_enabled() const { return settings_.collision_distance > 0.0; }
    bool ejections_enabled() const { return settings_.ejection_radius > 0.0; }

    void rebuild_active() {
        active_.clear();
        for (int b = 0; b < n_; ++b)
            if (status_[b] == static_cast<int>(BodyStatus::kActive))
                active_.push_back(b, mass_[b], soft_[b]);
    }

    State pack() const {
        State s(active_.state_size());
        for (std::size_t k = 0; k < active_.size(); ++k)
            for (int c = 0; c < 6; ++c) s[6 * k + c] = body_state_[active_.index[k]][c];
        s.back() = work_;
        return s;
    }

    void unpack(const State& s) {
        for (std::size_t k = 0; k < active_.size(); ++k)
            for (int c = 0; c < 6; ++c) body_state_[active_.index[k]][c] = s[6 * k + c];
        work_ = s.back();
    }

    double energy() const {
        const State s = pack();
        return total_energy(pot_, active_, s.data());
    }

    void record(double t) {
        result_.times.push_back(t);
        for (int b = 0; b < n_; ++b) {
            const bool on = status_[b] == static_cast<int>(BodyStatus::kActive);
            for (int c = 0; c < 6; ++c)
                result_.states.push_back(on ? body_state_[b][c]
                                            : std::numeric_limits<double>::quiet_NaN());
            result_.masses.push_back(on ? mass_[b]
                                        : std::numeric_limits<double>::quiet_NaN());
        }
        result_.energy.push_back(energy());
        result_.energy_removed.push_back(energy_removed_);
        result_.friction_work.push_back(work_);
        unrecorded_event_ = false;
    }

    // Encounter-based cap on the step size (infinite for a single body).
    double step_cap(const State& s) const {
        double cap = std::numeric_limits<double>::infinity();
        const std::size_t n = active_.size();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                double dx2 = 0.0, dv2 = 0.0;
                for (int c = 0; c < 3; ++c) {
                    const double dx = s[6 * i + c] - s[6 * j + c];
                    const double dv = s[6 * i + 3 + c] - s[6 * j + 3 + c];
                    dx2 += dx * dx;
                    dv2 += dv * dv;
                }
                const double r = std::sqrt(dx2);
                const double h = pair_softening(active_.softening[i], active_.softening[j]);
                const double rs = std::max(r, h);
                cap = std::min(cap,
                               settings_.step_factor *
                                   std::sqrt(rs * rs * rs / (active_.mass[i] + active_.mass[j])));
                if (dv2 > 0.0) {
                    const double v = std::sqrt(dv2);
                    cap = std::min(cap, settings_.step_factor * r / v);
                    if (r < kKernelMargin * h)
                        cap = std::min(cap, settings_.kernel_step_factor * h / v);
                }
            }
        }
        return cap;
    }

    // min over active pairs of (separation - collision_distance); positive
    // when no pair is in contact.
    double collision_gap(const State& s) const {
        double gap = std::numeric_limits<double>::infinity();
        const std::size_t n = active_.size();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                double dx2 = 0.0;
                for (int c = 0; c < 3; ++c) {
                    const double dx = s[6 * i + c] - s[6 * j + c];
                    dx2 += dx * dx;
                }
                gap = std::min(gap, std::sqrt(dx2) - settings_.collision_distance);
            }
        }
        return gap;
    }

    // Merge every pair in contact (closest first), repeatedly, at time t.
    // Operates on body_state_. Returns whether anything merged.
    bool handle_mergers(double t) {
        bool any = false;
        while (true) {
            int bi = -1, bj = -1;
            double r_min = settings_.collision_distance;
            for (std::size_t i = 0; i < active_.size(); ++i) {
                for (std::size_t j = i + 1; j < active_.size(); ++j) {
                    const int a = active_.index[i], b = active_.index[j];
                    double dx2 = 0.0;
                    for (int c = 0; c < 3; ++c) {
                        const double dx = body_state_[a][c] - body_state_[b][c];
                        dx2 += dx * dx;
                    }
                    const double r = std::sqrt(dx2);
                    if (r <= r_min) {
                        r_min = r;
                        bi = a;
                        bj = b;
                    }
                }
            }
            if (bi < 0) break;

            const double e_before = energy();
            const bool keep_i =
                mass_[bi] > mass_[bj] || (mass_[bi] == mass_[bj] && bi < bj);
            const int survivor = keep_i ? bi : bj;
            const int absorbed = keep_i ? bj : bi;
            const double m = mass_[bi] + mass_[bj];
            std::array<double, 6> merged{};
            double dv2 = 0.0;
            for (int c = 0; c < 6; ++c)
                merged[c] = (mass_[bi] * body_state_[bi][c] +
                             mass_[bj] * body_state_[bj][c]) / m;
            for (int c = 3; c < 6; ++c) {
                const double dv = body_state_[bi][c] - body_state_[bj][c];
                dv2 += dv * dv;
            }

            status_[absorbed] = static_cast<int>(BodyStatus::kMerged);
            merged_into_[absorbed] = survivor;
            final_time_[absorbed] = t;
            final_mass_[absorbed] = mass_[absorbed];
            final_state_[absorbed] = body_state_[absorbed];
            mass_[survivor] = m;
            soft_[survivor] = std::max(soft_[bi], soft_[bj]);
            body_state_[survivor] = merged;
            rebuild_active();
            energy_removed_ += e_before - energy();

            Event ev;
            ev.time = t;
            ev.type = static_cast<int>(EventType::kMerger);
            ev.body = survivor;
            ev.partner = absorbed;
            ev.mass = m;
            ev.state = merged;
            ev.value = std::sqrt(dv2);
            result_.events.push_back(ev);
            unrecorded_event_ = true;
            any = true;
        }
        return any;
    }

    // Ejection test for slot k of the current (packed) state. On success,
    // `value` is the escape energy used (positive) and `distance` the distance
    // from the reference point.
    bool is_ejected(const State& s, std::size_t k, double& value,
                    double& distance) const {
        const std::size_t n = active_.size();
        const double* xk = s.data() + 6 * k;
        const double mk = active_.mass[k];
        if (settings_.ejection_reference == static_cast<int>(EjectionReference::kCentre)) {
            distance = std::sqrt(xk[0] * xk[0] + xk[1] * xk[1] + xk[2] * xk[2]);
            if (distance <= settings_.ejection_radius) return false;
            if (xk[0] * xk[3] + xk[1] * xk[4] + xk[2] * xk[5] <= 0.0) return false;
            value = specific_energy(pot_, active_, s.data(), k);
            return value > 0.0;
        }

        // kSystem: relative to the centre of mass of the other bodies.
        if (n < 2) return false;
        double m_rest = 0.0;
        double x_rest[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < n; ++j) {
            if (j == k) continue;
            m_rest += active_.mass[j];
            for (int c = 0; c < 6; ++c) x_rest[c] += active_.mass[j] * s[6 * j + c];
        }
        double dx[3], dv[3];
        for (int c = 0; c < 3; ++c) {
            dx[c] = xk[c] - x_rest[c] / m_rest;
            dv[c] = xk[3 + c] - x_rest[3 + c] / m_rest;
        }
        distance = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
        if (distance <= settings_.ejection_radius) return false;
        if (dx[0] * dv[0] + dx[1] * dv[1] + dx[2] * dv[2] <= 0.0) return false;
        value = 0.5 * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]) -
                (mk + m_rest) / distance;
        if (value <= 0.0) return false;
        // Not bound to any single other body (e.g. its partner in a binary).
        for (std::size_t j = 0; j < n; ++j) {
            if (j == k) continue;
            double r2 = 0.0, v2 = 0.0;
            for (int c = 0; c < 3; ++c) {
                const double d = xk[c] - s[6 * j + c];
                const double w = xk[3 + c] - s[6 * j + 3 + c];
                r2 += d * d;
                v2 += w * w;
            }
            if (0.5 * v2 - (mk + active_.mass[j]) / std::sqrt(r2) < 0.0) return false;
        }
        return true;
    }

    // Remove ejected bodies one at a time (farthest first, lighter first on a
    // tie), re-testing the rest after each removal. Operates on body_state_.
    bool handle_ejections(double t) {
        bool any = false;
        while (active_.size() > 0) {
            const State s = pack();
            int best = -1;
            double best_d = -1.0, best_value = 0.0;
            for (std::size_t k = 0; k < active_.size(); ++k) {
                double value = 0.0, d = 0.0;
                if (!is_ejected(s, k, value, d)) continue;
                const bool better =
                    best < 0 || d > best_d ||
                    (d == best_d && active_.mass[k] < mass_[best]);
                if (better) {
                    best = active_.index[k];
                    best_d = d;
                    best_value = value;
                }
            }
            if (best < 0) break;

            const double e_before = energy();
            status_[best] = static_cast<int>(BodyStatus::kEjected);
            final_time_[best] = t;
            final_mass_[best] = mass_[best];
            final_state_[best] = body_state_[best];
            ++n_ejected_;
            rebuild_active();
            energy_removed_ += e_before - energy();

            Event ev;
            ev.time = t;
            ev.type = static_cast<int>(EventType::kEjection);
            ev.body = best;
            ev.mass = mass_[best];
            ev.state = body_state_[best];
            ev.value = best_value;
            result_.events.push_back(ev);
            unrecorded_event_ = true;
            any = true;
        }
        return any;
    }

    // Stop once no interactions remain: every body gone, or (starting from
    // more than one) a single body left.
    bool finished(int& reason) const {
        const std::size_t n = active_.size();
        if (n == 0 || (n_ > 1 && n == 1)) {
            reason = static_cast<int>(n_ejected_ == 0 ? StopReason::kAllMerged
                                                      : StopReason::kBodiesEjected);
            return true;
        }
        return false;
    }

    template <class Stepper>
    void run_with() {
        namespace ode = boost::numeric::odeint;
        const double t_max = settings_.t_max;

        // Output grid (excluding t = 0, which is recorded immediately).
        std::vector<double> t_out;
        if (settings_.record_trajectory && settings_.n_samples >= 2) {
            const int n_out = settings_.n_samples;
            for (int k = 1; k < n_out - 1; ++k)
                t_out.push_back(t_max * static_cast<double>(k) / (n_out - 1));
        }
        t_out.push_back(t_max);
        std::size_t next_out = 0;

        result_.n_bodies = n_;
        record(0.0);

        int reason = static_cast<int>(StopReason::kTimeLimit);
        bool stopped = false;
        double t = 0.0;

        // Bodies already in contact / already escaping at t = 0.
        bool changed = mergers_enabled() && handle_mergers(0.0);
        if (ejections_enabled() && handle_ejections(0.0)) changed = true;
        if (changed && finished(reason)) stopped = true;

        State x = pack();
        EquationsOfMotion sys{pot_, friction_, active_, false, {}};
        auto stepper = ode::make_controlled<Stepper>(settings_.abs_tol, settings_.rel_tol);
        double dt = 1e-3 * std::min(step_cap(x), t_max);
        long n_steps = 0, n_failed = 0;

        while (!stopped) {
            if (t >= t_max) break;
            if (n_steps >= settings_.max_steps) {
                reason = static_cast<int>(StopReason::kStepLimit);
                break;
            }
            const double t_target = t_out[next_out];
            const double cap = step_cap(x);
            const double remaining = t_target - t;
            double h = std::min(dt, cap);
            const bool clipped = h >= remaining;
            if (clipped) h = remaining;

            const State x_prev = x;
            const double t_prev = t;
            const auto res = stepper.try_step(std::ref(sys), x, t, h);
            if (sys.nan_hit) {
                x = x_prev;
                t = t_prev;
                reason = static_cast<int>(StopReason::kIntegrationError);
                break;
            }
            if (res == ode::fail) {
                ++n_failed;
                dt = h;
                if (!(h > 1e-15 * std::max(t, 1e-6 * t_max))) {
                    reason = static_cast<int>(StopReason::kIntegrationError);
                    break;
                }
                continue;
            }
            ++n_steps;
            // try_step returns the suggested next step in h. A step shortened
            // to hit an output time says little about the natural step size,
            // so keep the larger of the two suggestions.
            dt = clipped ? std::max(dt, h) : h;
            if (clipped || t >= t_target) t = t_target;  // no round-off overshoot

            changed = false;
            if (mergers_enabled() && collision_gap(x) <= 0.0) {
                // Bisect for the first contact within [t_prev, t], re-stepping
                // from x_prev with a fresh single-step integrator each time.
                double lo = 0.0, hi = t - t_prev;
                State x_hi = x, x_try(x.size());
                for (int it = 0; it < 64 && hi - lo > 1e-13 * hi; ++it) {
                    const double mid = 0.5 * (lo + hi);
                    Stepper single;
                    single.do_step(std::ref(sys), x_prev, t_prev, x_try, mid);
                    if (collision_gap(x_try) <= 0.0) {
                        hi = mid;
                        x_hi = x_try;
                    } else {
                        lo = mid;
                    }
                }
                x = x_hi;
                t = t_prev + hi;
                unpack(x);
                // An event changes which bodies are active, and so the packed
                // layout: re-pack at once so x always matches active_.
                if (handle_mergers(t)) {
                    changed = true;
                    x = pack();
                }
            }
            if (ejections_enabled()) {
                unpack(x);
                if (handle_ejections(t)) {
                    changed = true;
                    x = pack();
                }
            }
            if (changed) {
                stepper = ode::make_controlled<Stepper>(settings_.abs_tol,
                                                        settings_.rel_tol);
                if (finished(reason)) break;
                dt = std::min(dt, step_cap(x));
            }
            if (t == t_target) {
                unpack(x);
                if (next_out + 1 < t_out.size()) {
                    record(t);
                    ++next_out;
                }
            }
        }

        unpack(x);
        if (result_.times.back() != t || unrecorded_event_) record(t);

        result_.stop_reason = reason;
        result_.t_end = t;
        result_.n_steps = n_steps;
        result_.n_failed_steps = n_failed;
        for (int b = 0; b < n_; ++b) {
            if (status_[b] == static_cast<int>(BodyStatus::kActive)) {
                final_time_[b] = t;
                final_mass_[b] = mass_[b];
                final_state_[b] = body_state_[b];
            }
        }
        result_.status = status_;
        result_.merged_into = merged_into_;
        result_.final_time = final_time_;
        result_.final_mass = final_mass_;
        result_.final_state.clear();
        for (const auto& st : final_state_)
            result_.final_state.insert(result_.final_state.end(), st.begin(), st.end());
    }
};

// Integrate one experiment. See ScatteringIntegrator.
inline ScatteringResult integrate_system(const ExternalPotential& pot,
                                         const DynamicalFriction& friction,
                                         const IntegrationSettings& settings,
                                         const std::vector<double>& mass,
                                         const std::vector<double>& states,
                                         const std::vector<double>& softening) {
    ScatteringIntegrator integrator(pot, friction, settings, mass, states, softening);
    return integrator.run();
}

// Integrate a batch of independent experiments of n_bodies bodies each
// (OpenMP over experiments). `mass` is (n_systems, n_bodies), `states`
// (n_systems, n_bodies, 6), `softening` (n_systems, n_bodies) and `t_max`
// (n_systems,), all row-major; every other setting is shared. An experiment
// whose integration throws is returned with stop reason kIntegrationError and
// no samples.
inline std::vector<ScatteringResult> integrate_batch(
    const ExternalPotential& pot, const DynamicalFriction& friction,
    const IntegrationSettings& settings, std::size_t n_systems, int n_bodies,
    const double* mass, const double* states, const double* softening,
    const double* t_max, bool progress = false) {
    std::vector<ScatteringResult> results(n_systems);
    const std::size_t nb = static_cast<std::size_t>(n_bodies);
    std::atomic<std::size_t> completed{0};
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < n_systems; ++i) {
        IntegrationSettings s = settings;
        s.t_max = t_max[i];
        std::vector<double> m(mass + i * nb, mass + (i + 1) * nb);
        std::vector<double> x(states + 6 * i * nb, states + 6 * (i + 1) * nb);
        std::vector<double> h(softening + i * nb, softening + (i + 1) * nb);
        try {
            results[i] = integrate_system(pot, friction, s, m, x, h);
        } catch (...) {
            results[i] = ScatteringResult{};
            results[i].n_bodies = n_bodies;
            results[i].stop_reason = static_cast<int>(StopReason::kIntegrationError);
        }
        if (progress) {
            const std::size_t done = completed.fetch_add(1) + 1;
            const std::size_t decile = (done * 10) / n_systems;
            if (decile != ((done - 1) * 10) / n_systems) {
                #pragma omp critical(asmodean_progress)
                {
                    std::printf("%zu%% of experiments integrated\n", decile * 10);
                    std::fflush(stdout);
                }
            }
        }
    }
    return results;
}

}  // namespace asmodean
