#pragma once

// Chandrasekhar (1943) dynamical friction from a static, non-rotating,
// isotropic Maxwellian background (Binney & Tremaine 2008, eq. 8.7):
//
//   a_df = -4 pi G^2 M rho(r) lnL / v^3 * [erf(X) - 2X/sqrt(pi) exp(-X^2)] v,
//   X = v / (sqrt(2) sigma(r)),
//
// for a mass M moving with velocity v at distance r from the potential centre
// (the origin), where rho(r) and sigma(r) are the background density and
// one-dimensional velocity dispersion. They are given as tabulated radial
// profiles, interpolated linearly in log r - log rho / log sigma and held at
// their end values outside the table.
//
// Two ways of applying the drag to the bodies are supported:
//
// * kSystem (default): the bodies are treated as one object. The drag force
//   F = M_tot a_df(M_tot, X_com, V_com) on their combined mass, moving with
//   their centre-of-mass velocity at their centre of mass, is distributed
//   among the bodies in proportion to their masses, F_i = (m_i / M_tot) F.
//   Every body therefore feels the same deceleration a_df, which slows the
//   centre of mass without changing the internal (relative) motion.
// * kIndividual: each body feels the drag of its own mass and velocity.
//
// The Coulomb logarithm is either a constant, or computed as
// lnL = 0.5 ln(1 + Lambda^2) with Lambda = b_max (v^2 + sigma^2) / (G M) and
// b_max = r (Binney & Tremaine 2008, eq. 8.1b with a typical encounter speed
// sqrt(v^2 + sigma^2)).
//
// Units follow the rest of the core: G = 1 (the potential's HO units).

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "common.hpp"

namespace asmodean {

enum class FrictionMode : int { kSystem = 0, kIndividual = 1 };

// Velocity-dependent bracket erf(X) - 2X/sqrt(pi) exp(-X^2) of the Chandrasekhar
// formula. For small X the two terms cancel catastrophically, so the leading
// terms of its series 4X^3/(3 sqrt(pi)) (1 - 3X^2/5) are used instead.
inline double chandrasekhar_bracket(double X) {
    const double inv_sqrt_pi = 0.56418958354775628;  // 1/sqrt(pi)
    if (X < 1e-3) return 4.0 / 3.0 * inv_sqrt_pi * X * X * X * (1.0 - 0.6 * X * X);
    return std::erf(X) - 2.0 * inv_sqrt_pi * X * std::exp(-X * X);
}

class DynamicalFriction {
public:
    // Disabled: acceleration() is identically zero.
    DynamicalFriction() = default;

    // Enabled, from tabulated profiles. `radius` must be strictly increasing
    // and positive; `density` and `dispersion` positive and the same length.
    DynamicalFriction(const std::vector<double>& radius,
                      const std::vector<double>& density,
                      const std::vector<double>& dispersion,
                      double coulomb_logarithm, bool variable_coulomb_logarithm,
                      FrictionMode mode)
        : enabled_(true),
          radius_(radius),
          density_(density),
          dispersion_(dispersion),
          coulomb_logarithm_(coulomb_logarithm),
          variable_coulomb_logarithm_(variable_coulomb_logarithm),
          mode_(mode) {
        const std::size_t n = radius.size();
        if (n == 0) throw std::invalid_argument("friction profile must be non-empty");
        if (density.size() != n || dispersion.size() != n)
            throw std::invalid_argument(
                "friction radius, density and dispersion must have equal length");
        for (std::size_t i = 0; i < n; ++i) {
            if (!(radius[i] > 0.0) || !(density[i] > 0.0) || !(dispersion[i] > 0.0))
                throw std::invalid_argument(
                    "friction radius, density and dispersion must be positive");
            if (i > 0 && !(radius[i] > radius[i - 1]))
                throw std::invalid_argument(
                    "friction radius must be strictly increasing");
        }
        if (!variable_coulomb_logarithm && !(coulomb_logarithm >= 0.0))
            throw std::invalid_argument("Coulomb logarithm must be non-negative");
        log_r_.resize(n);
        log_rho_.resize(n);
        log_sigma_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            log_r_[i] = std::log(radius[i]);
            log_rho_[i] = std::log(density[i]);
            log_sigma_[i] = std::log(dispersion[i]);
        }
    }

    bool enabled() const { return enabled_; }
    FrictionMode mode() const { return mode_; }
    double constant_coulomb_logarithm() const { return coulomb_logarithm_; }
    bool variable_coulomb_logarithm() const { return variable_coulomb_logarithm_; }
    const std::vector<double>& radius() const { return radius_; }
    const std::vector<double>& density_table() const { return density_; }
    const std::vector<double>& dispersion_table() const { return dispersion_; }

    // Background density and 1D velocity dispersion at distance r from the
    // centre (log-log interpolation, clamped to the table ends).
    double density(double r) const { return std::exp(interpolate(log_rho_, r)); }
    double dispersion(double r) const {
        return std::exp(interpolate(log_sigma_, r));
    }

    // Coulomb logarithm for a mass `mass` at distance r moving at speed v
    // through a background of dispersion sigma.
    double coulomb_logarithm(double mass, double r, double v, double sigma) const {
        if (!variable_coulomb_logarithm_) return coulomb_logarithm_;
        const double lambda = r * (v * v + sigma * sigma) / mass;
        return 0.5 * std::log1p(lambda * lambda);
    }

    // Chandrasekhar deceleration of a mass `mass` at `pos` moving at `vel`.
    Vec3 acceleration(double mass, const Vec3& pos, const Vec3& vel) const {
        if (!enabled_ || !(mass > 0.0)) return {0.0, 0.0, 0.0};
        const double v = norm(vel);
        if (!(v > 0.0)) return {0.0, 0.0, 0.0};
        const double r = norm(pos);
        const double rho = density(r);
        const double sigma = dispersion(r);
        const double X = v / (std::sqrt(2.0) * sigma);
        const double lnL = coulomb_logarithm(mass, r, v, sigma);
        // For small X the bracket ~ X^3, so bracket / v^3 stays finite.
        const double fac =
            -4.0 * M_PI * mass * rho * lnL * chandrasekhar_bracket(X) / (v * v * v);
        return {fac * vel[0], fac * vel[1], fac * vel[2]};
    }

private:
    bool enabled_ = false;
    std::vector<double> radius_, density_, dispersion_;  // as given (for pickling)
    std::vector<double> log_r_, log_rho_, log_sigma_;
    double coulomb_logarithm_ = 0.0;
    bool variable_coulomb_logarithm_ = false;
    FrictionMode mode_ = FrictionMode::kSystem;

    double interpolate(const std::vector<double>& log_y, double r) const {
        const std::size_t n = log_r_.size();
        if (n == 1 || !(r > 0.0)) return log_y.front();
        const double lr = std::log(r);
        if (lr <= log_r_.front()) return log_y.front();
        if (lr >= log_r_.back()) return log_y.back();
        const auto it = std::upper_bound(log_r_.begin(), log_r_.end(), lr);
        const std::size_t i = static_cast<std::size_t>(it - log_r_.begin());
        const double f = (lr - log_r_[i - 1]) / (log_r_[i] - log_r_[i - 1]);
        return log_y[i - 1] + f * (log_y[i] - log_y[i - 1]);
    }
};

}  // namespace asmodean
