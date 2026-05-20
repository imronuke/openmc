#pragma once

#include <cstddef>
#include <vector>
#include <optional>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>

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

    // Reshape helpers
    Eigen::MatrixXd to_matrix_left()  const;  // (r_left*n) x r_right
    Eigen::MatrixXd to_matrix_right() const;  // r_left x (n*r_right)
    void from_matrix_left (const Eigen::MatrixXd& M, int r_left, int n, int r_right);
    void from_matrix_right(const Eigen::MatrixXd& M, int r_left, int n, int r_right);

    int size() const { return r_left * n * r_right; }
};

// ─────────────────────────────────────────────────────────────────────────────
// SVD result
// ─────────────────────────────────────────────────────────────────────────────
struct SVDResult {
    Eigen::MatrixXd U;
    Eigen::VectorXd S;
    Eigen::MatrixXd Vt;
};

// ─────────────────────────────────────────────────────────────────────────────
// Free function declarations
// ─────────────────────────────────────────────────────────────────────────────

// Determine truncation rank from singular values and relative threshold eps
int rank_truncation(const Eigen::VectorXd& S, double eps);

// Thin SVD with QR pre-conditioning for tall/wide matrices
SVDResult fast_svd(const Eigen::MatrixXd& X, double aspect = 1.5);

// Randomized SVD (Halko et al.)
SVDResult randomized_svd(const Eigen::MatrixXd& X,
                          int rank,
                          int n_oversamples = 10,
                          int n_iter        = 2,
                          int random_state  = -1);   // -1 = no seed

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
    std::string      repr()   const;

    // ── Reconstruct full tensor ────────────────────────────────────────────────
    // Returns a flat row-major buffer together with the shape.
    // For small tensors only – exponential memory in d.
    std::pair<std::vector<double>, std::vector<int>> full() const;

    // ── Arithmetic ────────────────────────────────────────────────────────────
    TT operator+(const TT& other) const;   // TT addition
    TT operator*(const TT& other) const;   // Hadamard (element-wise) product

    // ── Single-entry access ───────────────────────────────────────────────────
    // tt.at({i0, i1, ..., i_{d-1}})
    double at(const std::vector<int>& indices) const;

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

// TT-SVD algorithm
// tensor:  flat row-major buffer + shape
// ranks:   fixed ranks (length d-1) – OR –
// eps:     relative error threshold (if ranks is empty)
TT tt_svd(const std::vector<double>& tensor,
          const std::vector<int>&    shape,
          std::vector<int>           ranks    = {},
          double                     eps      = -1.0);

// TT-SVD for an OpenMC tally VALUE buffer. Applies normalization and optionally
// squares the normalized values before decomposition.
TT tt_svd_tally_value(const double* value,
                      int size,
                      const std::vector<int>& shape,
                      double norm,
                      bool square,
                      double eps = -1.0);

// TT decomposition via randomized SVD
TT tt_rand(const std::vector<double>& tensor,
           const std::vector<int>&    shape,
           std::vector<int>           ranks       = {},
           double                     eps         = -1.0,
           int                        max_rank    = 100,
           int                        n_oversamples = 10,
           int                        n_iter      = 2,
           int                        random_state = -1);

// Create zero TT of given shape (rank-1 cores)
TT tt_zeros(const std::vector<int>& shape);

std::vector<int> auto_tt_shape(int n,
                               int site_min = 9,
                               int site_cap = 243,
                               int prefer_order = 3,
                               int min_order = 1,
                               int max_order = 5);

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

double compute_compression_ratio(const TT& tt);

std::size_t tt_storage_size(const TT& tt);

std::size_t tt_storage_bytes(const TT& tt);

// original: flat row-major buffer matching tt.shape()
double compute_relative_error(const std::vector<double>& original, const TT& tt);

} // namespace openmc
