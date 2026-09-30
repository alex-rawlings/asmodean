#pragma once

// Small shared definitions: the body-count limit and 3-vector helpers.

#include <array>
#include <cmath>

namespace asmodean {

// Maximum number of bodies in one scattering experiment. Pairwise forces are
// summed directly (O(N^2)), which is only sensible for a few bodies.
constexpr int kMaxBodies = 10;

using Vec3 = std::array<double, 3>;

inline double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

}  // namespace asmodean
