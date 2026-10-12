#pragma once

#include <cstddef>
#include <optional>
#include <vector>
#include <Eigen/Dense>

#include "openmc/vector.h"

namespace openmc {

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────────────────────

// A TT-core is a 3-D array of shape (r_left, n, r_right).
// We store it as a flat std::vector<double> in row-major order
// together with its three dimensions.
struct Core {
    int r_left, n, r_right;
    std::vector<double> data;   // size = r_left * n * r_right

    Core() = default;
    Core(int r_left, int n, int r_right);
    Core(int r_left, int n, int r_right, std::vector<double> data);

    // Element access  (no bounds check in release)
    double& operator()(int i, int j, int k);
    double  operator()(int i, int j, int k) const;

    // Return the slice core[:, j, :] as an Eigen matrix  (r_left x r_right)
    Eigen::MatrixXd mode_slice(int j) const;

    // Fill from an Eigen matrix slice  core[:, j, :] = M
    void set_mode_slice(int j, const Eigen::MatrixXd& M);
};

// ─────────────────────────────────────────────────────────────────────────────
// Free function declarations
// ─────────────────────────────────────────────────────────────────────────────

// Determine truncation rank from singular values and relative threshold eps
int rank_truncation(const Eigen::VectorXd& S, double eps);

// ─────────────────────────────────────────────────────────────────────────────
// TT class
// ─────────────────────────────────────────────────────────────────────────────
class TT {
public:
    std::vector<Core> cores;

    // ── Constructors ──────────────────────────────────────────────────────────
    TT() = default;
    explicit TT(std::vector<Core> cores);

    // ── Shape / rank helpers ──────────────────────────────────────────────────
    std::vector<int> shape()  const;   // physical dimensions
    std::vector<int> ranks()  const;   // TT-ranks  (length d+1)
    int              ndim()   const;   // number of dimensions d

    // ── Arithmetic ────────────────────────────────────────────────────────────
    TT operator+(const TT& other) const;   // TT addition

    // ── Rounding / compression ────────────────────────────────────────────────
    // Mutates cores in-place (mirrors Python behaviour).
    // Pass ranks OR eps, not both.
    void round(std::optional<std::vector<int>> ranks = std::nullopt,
               double eps      = 1e-10,
               int    max_rank = 1'000'000);
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory / decomposition functions
// ─────────────────────────────────────────────────────────────────────────────

// TT-SVD for an OpenMC tally VALUE buffer. Applies normalization and optionally
// squares the normalized values before decomposition. The input buffer is not
// modified.
TT tt_svd_tally_value(const double* value, int size,
  const std::vector<int>& shape, double norm, bool square, double eps = -1.0);

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

std::size_t tt_storage_size(const TT& tt);

std::size_t tt_storage_bytes(const TT& tt);

} // namespace openmc
