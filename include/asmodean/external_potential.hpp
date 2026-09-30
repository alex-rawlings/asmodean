#pragma once

// The fixed, analytical background potential the bodies move in.
//
// The potential is one of lanfear's analytical potentials (an SCF expansion, a
// Miyamoto-Nagai disc basis, or a per-species composite of the two, each with
// any softened black holes attached), or no potential at all for a pure
// few-body problem. The lanfear classes are compiled straight from lanfear's
// (header-only) C++ core and rebuilt from the state lanfear pickles them with,
// so the forces are bit-for-bit those lanfear itself evaluates.
//
// Units are those of the lanfear potential: G = 1, and the length and mass
// units are the potential's scale radius and field mass (the Hernquist-Ostriker
// "HO" units). The python layer converts the bodies into these units.
//
// The figure never rotates: the potential is static in the inertial frame
// (a pattern speed carried by the lanfear potential is ignored).

#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "lanfear/composite_potential.hpp"
#include "lanfear/disc_potential.hpp"
#include "lanfear/scf_potential.hpp"

namespace asmodean {

// No background potential: the bodies interact only with each other.
struct NullPotential {
    double potential(double, double, double) const { return 0.0; }
    std::array<double, 3> acceleration(double, double, double) const {
        return {0.0, 0.0, 0.0};
    }
};

class ExternalPotential {
public:
    using Variant = std::variant<NullPotential, lanfear::SCFPotential,
                                 lanfear::DiscPotential,
                                 lanfear::CompositePotential>;

    ExternalPotential() : pot_(NullPotential{}) {}
    explicit ExternalPotential(Variant pot) : pot_(std::move(pot)) {}

    double potential(double x, double y, double z) const {
        return std::visit([&](const auto& p) { return p.potential(x, y, z); },
                          pot_);
    }

    std::array<double, 3> acceleration(double x, double y, double z) const {
        return std::visit(
            [&](const auto& p) -> std::array<double, 3> {
                const auto a = p.acceleration(x, y, z);
                return {a[0], a[1], a[2]};
            },
            pot_);
    }

    bool is_null() const { return std::holds_alternative<NullPotential>(pot_); }

    // Short name of the wrapped potential type ("none", "scf", "disc" or
    // "composite"), matching the python-side spec.
    std::string kind() const {
        switch (pot_.index()) {
            case 1: return "scf";
            case 2: return "disc";
            case 3: return "composite";
            default: return "none";
        }
    }

    // Number of softened point masses attached to the potential itself.
    std::size_t num_black_holes() const {
        if (is_null()) return 0;
        return std::visit(
            [](const auto& p) -> std::size_t {
                if constexpr (std::is_same_v<std::decay_t<decltype(p)>,
                                             NullPotential>) {
                    return 0;
                } else {
                    return p.num_black_holes();
                }
            },
            pot_);
    }

private:
    Variant pot_;
};

}  // namespace asmodean
