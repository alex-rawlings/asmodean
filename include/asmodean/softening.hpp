#pragma once

// Gravitational softening between bodies, following Gadget (Springel 2005,
// MNRAS 364, 1105; Springel et al. 2021, MNRAS 506, 2871).
//
// The user-facing softening of each body is the *Plummer-equivalent* length
// epsilon (the value given in a Gadget parameter file). The force itself uses
// the cubic-spline kernel with compact support h = 2.8 epsilon: exactly
// Newtonian for r >= h, and smoothly softened to a finite value inside, where
// the potential at r = 0 equals that of a Plummer sphere of scale epsilon,
// -m / epsilon. Two bodies of different softening interact with the larger of
// their two kernel lengths, h_ij = max(h_i, h_j), as Gadget does.
//
// The kernel is the one lanfear uses for its softened black holes (and which
// Gadget uses for every particle), so it is shared rather than duplicated.

#include <algorithm>

#include "lanfear/spline_softening.hpp"

namespace asmodean {

// Ratio h / epsilon of the spline-kernel length to the Plummer-equivalent
// softening length.
constexpr double kSplineToPlummer = 2.8;

// Spline-kernel length h for a Plummer-equivalent softening epsilon.
inline double spline_length(double plummer_softening) {
    return kSplineToPlummer * plummer_softening;
}

// Kernel length used for the interaction of two bodies (Gadget convention).
inline double pair_softening(double h_i, double h_j) { return std::max(h_i, h_j); }

using lanfear::spline_softened_force_factor;
using lanfear::spline_softened_potential;

}  // namespace asmodean
