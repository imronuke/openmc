#include "openmc/tt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

namespace openmc {

namespace {

using RowMatrix =
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstCoreMap = Eigen::Map<const RowMatrix>;

ConstCoreMap left_unfolding(const Core& core)
{
  return ConstCoreMap(core.data.data(), core.r_left * core.n, core.r_right);
}

ConstCoreMap right_unfolding(const Core& core)
{
  return ConstCoreMap(core.data.data(), core.r_left, core.n * core.r_right);
}

template<class Derived>
void assign_left(Core& core, const Eigen::MatrixBase<Derived>& matrix,
  int r_left, int n, int r_right)
{
  core.r_left = r_left;
  core.n = n;
  core.r_right = r_right;
  core.data.resize(static_cast<std::size_t>(r_left) * n * r_right);
  Eigen::Map<RowMatrix>(core.data.data(), r_left * n, r_right) = matrix;
}

template<class Derived>
void assign_right(Core& core, const Eigen::MatrixBase<Derived>& matrix,
  int r_left, int n, int r_right)
{
  core.r_left = r_left;
  core.n = n;
  core.r_right = r_right;
  core.data.resize(static_cast<std::size_t>(r_left) * n * r_right);
  Eigen::Map<RowMatrix>(core.data.data(), r_left, n * r_right) = matrix;
}

// Factorizations and scratch matrices are reused across bonds. Factor views
// remain valid until the next compute(), so consumers need not copy U, S or V.
struct SVDWorkspace {
  template<class Derived>
  void compute(const Eigen::MatrixBase<Derived>& matrix,
    std::optional<double> eps = std::nullopt,
    int max_rank = std::numeric_limits<int>::max(), double aspect = 1.5)
  {
    int rows = matrix.rows();
    int cols = matrix.cols();
    tall_ = rows >= aspect * cols;
    wide_ = !tall_ && cols >= aspect * rows;

    if (tall_) {
      qr_.compute(matrix);
      factor_ =
        qr_.matrixQR().topRows(cols).template triangularView<Eigen::Upper>();
    } else if (wide_) {
      qr_.compute(matrix.transpose());
      factor_ =
        qr_.matrixQR().topRows(rows).template triangularView<Eigen::Upper>();
      factor_.transposeInPlace();
    } else {
      factor_ = matrix;
    }

    svd_.compute(factor_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    rank_ = std::min(static_cast<int>(svd_.singularValues().size()), max_rank);
    if (eps) {
      rank_ = std::min(rank_, rank_truncation(svd_.singularValues(), *eps));
    }
    rank_ = std::max(rank_, 1);

    // Only lift the retained vectors through Q. Applying its Householder
    // sequence in place avoids materializing the full thin Q and its product.
    if (tall_) {
      lifted_.setZero(rows, rank_);
      lifted_.topRows(cols) = svd_.matrixU().leftCols(rank_);
      lifted_.applyOnTheLeft(qr_.householderQ());
    } else if (wide_) {
      lifted_.setZero(cols, rank_);
      lifted_.topRows(rows) = svd_.matrixV().leftCols(rank_);
      lifted_.applyOnTheLeft(qr_.householderQ());
    }
  }

  auto matrix_u() const
  {
    return (tall_ ? lifted_ : svd_.matrixU()).leftCols(rank_);
  }

  auto matrix_v() const
  {
    return (wide_ ? lifted_ : svd_.matrixV()).leftCols(rank_);
  }

  auto singular_values() const { return svd_.singularValues().head(rank_); }

  Eigen::HouseholderQR<Eigen::MatrixXd> qr_;
  Eigen::BDCSVD<Eigen::MatrixXd> svd_;
  Eigen::MatrixXd factor_;
  Eigen::MatrixXd lifted_;
  int rank_ {0};
  bool tall_ {false};
  bool wide_ {false};
};

// buf stays in row-major order between successive TT unfoldings.
TT decompose_buffer(vector<double> buf, const vector<int>& shape,
  const vector<int>& ranks, double eps)
{
  SVDWorkspace workspace;
  vector<Core> cores;
  cores.reserve(shape.size());
  int r_prev = 1;

  for (int k = 0; k < static_cast<int>(shape.size()) - 1; ++k) {
    int rows = r_prev * shape[k];
    int cols = buf.size() / rows;
    ConstCoreMap matrix(buf.data(), rows, cols);
    workspace.compute(matrix,
      ranks.empty() ? std::optional<double>(eps) : std::nullopt,
      ranks.empty() ? std::numeric_limits<int>::max() : ranks[k]);
    int rank = workspace.rank_;

    Core core;
    assign_left(core, workspace.matrix_u(), r_prev, shape[k], rank);
    cores.push_back(std::move(core));

    buf.resize(static_cast<std::size_t>(rank) * cols);
    Eigen::Map<RowMatrix>(buf.data(), rank, cols) =
      workspace.singular_values().asDiagonal() *
      workspace.matrix_v().transpose();
    r_prev = rank;
  }

  // Copy only the final values: moving buf would retain the original dense
  // tensor's allocation in this small core after decomposition.
  Core last_core(r_prev, shape.back(), 1);
  std::copy_n(buf.begin(), last_core.data.size(), last_core.data.begin());
  cores.push_back(std::move(last_core));
  return TT(std::move(cores));
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Core implementation
// ─────────────────────────────────────────────────────────────────────────────

Core::Core(int r_left, int n, int r_right)
    : r_left(r_left), n(n), r_right(r_right),
      data(r_left * n * r_right, 0.0) {}

Core::Core(int r_left, int n, int r_right, std::vector<double> data)
    : r_left(r_left), n(n), r_right(r_right), data(std::move(data)) {
    assert(static_cast<int>(this->data.size()) == r_left * n * r_right);
}

double& Core::operator()(int i, int j, int k) {
    return data[i * (n * r_right) + j * r_right + k];
}

double Core::operator()(int i, int j, int k) const {
    return data[i * (n * r_right) + j * r_right + k];
}

// core[:, j, :] → Eigen matrix (r_left × r_right)
Eigen::MatrixXd Core::mode_slice(int j) const {
    Eigen::MatrixXd M(r_left, r_right);
    for (int i = 0; i < r_left; ++i)
        for (int k = 0; k < r_right; ++k)
            M(i, k) = (*this)(i, j, k);
    return M;
}

// core[:, j, :] ← M
void Core::set_mode_slice(int j, const Eigen::MatrixXd& M) {
    assert(M.rows() == r_left && M.cols() == r_right);
    for (int i = 0; i < r_left; ++i)
        for (int k = 0; k < r_right; ++k)
            (*this)(i, j, k) = M(i, k);
}

// Reshape to (r_left * n) × r_right  (left-unfolding)
Eigen::MatrixXd Core::to_matrix_left() const
{
  return left_unfolding(*this);
}

Eigen::MatrixXd Core::to_matrix_right() const
{
  return right_unfolding(*this);
}

void Core::from_matrix_left(
  const Eigen::MatrixXd& matrix, int r_left, int n, int r_right)
{
  assign_left(*this, matrix, r_left, n, r_right);
}

void Core::from_matrix_right(
  const Eigen::MatrixXd& matrix, int r_left, int n, int r_right)
{
  assign_right(*this, matrix, r_left, n, r_right);
}

// ─────────────────────────────────────────────────────────────────────────────
// rank_truncation
// ─────────────────────────────────────────────────────────────────────────────

int rank_truncation(const Eigen::VectorXd& S, double eps) {
    int r = static_cast<int>(S.size());
    Eigen::VectorXd S2 = S.array().square();
    double total = S2.sum();
    if (total == 0.0) return 1;

    // cumulative sum from the tail
    double tail = 0.0;
    for (int i = r - 1; i >= 0; --i) {
        tail += S2(i);
        if (std::sqrt(tail / total) >= eps)
            return i + 1;            // keep singular values 0 … i
    }
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// fast_svd
// ─────────────────────────────────────────────────────────────────────────────

SVDResult fast_svd(const Eigen::MatrixXd& matrix, double aspect)
{
  SVDWorkspace workspace;
  workspace.compute(
    matrix, std::nullopt, std::numeric_limits<int>::max(), aspect);
  return {workspace.matrix_u(), workspace.singular_values(),
    workspace.matrix_v().transpose()};
}

// ─────────────────────────────────────────────────────────────────────────────
// randomized_svd
// ─────────────────────────────────────────────────────────────────────────────

SVDResult randomized_svd(const Eigen::MatrixXd& X,
                          int rank,
                          int n_oversamples,
                          int n_iter,
                          int random_state) {
    int m = static_cast<int>(X.rows());
    int n = static_cast<int>(X.cols());
    int p = std::min(rank + n_oversamples, std::min(m, n));

    // Step 1: Random test matrix
    std::mt19937_64 rng;
    if (random_state >= 0)
        rng.seed(static_cast<uint64_t>(random_state));
    else
        rng.seed(std::random_device{}());

    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Omega(n, p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            Omega(i, j) = dist(rng);

    // Step 2: Y = X @ Omega
    Eigen::MatrixXd Y = X * Omega;

    // Step 3: Power iterations
    for (int it = 0; it < n_iter; ++it)
        Y = X * (X.transpose() * Y);

    // Step 4: QR of Y
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
    Eigen::MatrixXd Q = qr.householderQ() *
                        Eigen::MatrixXd::Identity(m, p);

    // Step 5: B = Q^T X
    Eigen::MatrixXd B = Q.transpose() * X;

    // Step 6: SVD of B
    Eigen::BDCSVD<Eigen::MatrixXd> svd(B,
        Eigen::ComputeThinU | Eigen::ComputeThinV);

    int r_out = std::min(rank, static_cast<int>(svd.singularValues().size()));

    SVDResult res;
    res.U  = (Q * svd.matrixU()).leftCols(r_out);
    res.S  = svd.singularValues().head(r_out);
    res.Vt = svd.matrixV().transpose().topRows(r_out);
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// TT implementation
// ─────────────────────────────────────────────────────────────────────────────

TT::TT(std::vector<Core> cores) : cores(std::move(cores)) {}

std::vector<int> TT::shape() const {
    std::vector<int> s;
    s.reserve(cores.size());
    for (const auto& c : cores) s.push_back(c.n);
    return s;
}

std::vector<int> TT::ranks() const {
    std::vector<int> r;
    r.reserve(cores.size() + 1);
    for (const auto& c : cores) r.push_back(c.r_left);
    r.push_back(cores.back().r_right);
    return r;
}

int TT::ndim() const { return static_cast<int>(cores.size()); }

std::string TT::repr() const {
    std::ostringstream oss;
    oss << "TT(shape=(";
    auto sh = shape();
    for (int i = 0; i < (int)sh.size(); ++i) {
        oss << sh[i];
        if (i + 1 < (int)sh.size()) oss << ", ";
    }
    oss << "), ranks=(";
    auto rk = ranks();
    for (int i = 0; i < (int)rk.size(); ++i) {
        oss << rk[i];
        if (i + 1 < (int)rk.size()) oss << ", ";
    }
    oss << "))";
    return oss.str();
}

// ── full() ────────────────────────────────────────────────────────────────────
// Reconstructs the full tensor by contracting all TT cores.
// Uses mode slices: at each step, for each existing multi-index and each new
// mode index j, we multiply the current vector by core[:, j, :].
//
// We represent the partial result as a 2D matrix:
//   rows  = all combined multi-indices so far  (product n0*...*n_{k-1})
//   cols  = current right rank r_k
//
// For each new core k:
//   new_result[combined_idx * n_k + j, :] = old_result[combined_idx, :] @ core_k[:, j, :]
std::pair<std::vector<double>, std::vector<int>> TT::full() const {
    int d = ndim();
    auto sh = shape();

    // cores[0]: (1, n0, r0)
    // Initialise result as n0 x r0 matrix
    int n0 = sh[0];
    int r0 = cores[0].r_right;
    Eigen::MatrixXd result(n0, r0);
    for (int j = 0; j < n0; ++j)
        result.row(j) = cores[0].mode_slice(j).row(0);  // 1 x r0 -> row j

    for (int k = 1; k < d; ++k) {
        int nk       = sh[k];
        int rk_right = cores[k].r_right;
        int N_prev   = static_cast<int>(result.rows());

        Eigen::MatrixXd new_result(N_prev * nk, rk_right);
        for (int i = 0; i < N_prev; ++i) {
            Eigen::RowVectorXd v = result.row(i);       // 1 x r_{k-1}
            for (int j = 0; j < nk; ++j) {
                // v @ core_k[:, j, :] -> 1 x r_k
                new_result.row(i * nk + j) = v * cores[k].mode_slice(j);
            }
        }
        result = new_result;
    }

    // result: (n0*...*n_{d-1}) x 1
    int total = 1;
    for (int n : sh) total *= n;

    std::vector<double> buf(total);
    for (int i = 0; i < total; ++i)
        buf[i] = result(i, 0);

    return {buf, sh};
}

// ── operator+ ────────────────────────────────────────────────────────────────
TT TT::operator+(const TT& other) const {
    if (shape() != other.shape())
        throw std::invalid_argument("TT addition: shape mismatch");

    int d = ndim();
    std::vector<Core> new_cores;
    new_cores.reserve(d);

    if (d == 1) {
        const Core& G1 = cores[0];
        const Core& G2 = other.cores[0];
        Core sum(1, G1.n, 1);
        for (int j = 0; j < G1.n; ++j) {
            sum(0, j, 0) = G1(0, j, 0) + G2(0, j, 0);
        }
        new_cores.push_back(std::move(sum));
        return TT(std::move(new_cores));
    }

    for (int k = 0; k < d; ++k) {
        const Core& G1 = cores[k];
        const Core& G2 = other.cores[k];
        int n_k = G1.n;

        if (k == 0) {
            // Concatenate along r_right:  [G1, G2]
            // New shape: (1, n_k, r1_right + r2_right)
            Core nc(1, n_k, G1.r_right + G2.r_right);
            for (int j = 0; j < n_k; ++j) {
                auto m1 = G1.mode_slice(j);   // 1 × r1_right
                auto m2 = G2.mode_slice(j);   // 1 × r2_right
                Eigen::MatrixXd cat(1, G1.r_right + G2.r_right);
                cat << m1, m2;
                nc.set_mode_slice(j, cat);
            }
            new_cores.push_back(std::move(nc));

        } else if (k == d - 1) {
            // Concatenate along r_left:
            // New shape: (r1_left + r2_left, n_k, 1)
            Core nc(G1.r_left + G2.r_left, n_k, 1);
            for (int j = 0; j < n_k; ++j) {
                auto m1 = G1.mode_slice(j);   // r1_left × 1
                auto m2 = G2.mode_slice(j);   // r2_left × 1
                Eigen::MatrixXd cat(G1.r_left + G2.r_left, 1);
                cat << m1, m2;
                nc.set_mode_slice(j, cat);
            }
            new_cores.push_back(std::move(nc));

        } else {
            // Block diagonal:
            // [ G1   0  ]
            // [  0  G2  ]
            int rl = G1.r_left  + G2.r_left;
            int rr = G1.r_right + G2.r_right;
            Core nc(rl, n_k, rr);
            for (int j = 0; j < n_k; ++j) {
                Eigen::MatrixXd blk = Eigen::MatrixXd::Zero(rl, rr);
                blk.block(0,          0,          G1.r_left, G1.r_right) = G1.mode_slice(j);
                blk.block(G1.r_left,  G1.r_right, G2.r_left, G2.r_right) = G2.mode_slice(j);
                nc.set_mode_slice(j, blk);
            }
            new_cores.push_back(std::move(nc));
        }
    }
    return TT(std::move(new_cores));
}

// ── operator* (Hadamard) ──────────────────────────────────────────────────────
TT TT::operator*(const TT& other) const {
    if (shape() != other.shape())
        throw std::invalid_argument("TT Hadamard: shape mismatch");

    int d = ndim();
    std::vector<Core> new_cores;
    new_cores.reserve(d);

    for (int k = 0; k < d; ++k) {
        const Core& G1 = cores[k];
        const Core& G2 = other.cores[k];
        int n_k  = G1.n;
        int rl   = G1.r_left  * G2.r_left;
        int rr   = G1.r_right * G2.r_right;

        Core nc(rl, n_k, rr);
        for (int j = 0; j < n_k; ++j) {
            // Kronecker product of the two mode slices
            // A: r1_left x r1_right,  B: r2_left x r2_right
            // kron(A, B): (r1_left*r2_left) x (r1_right*r2_right)
            //   result[i1*r2l + i2, k1*r2r + k2] = A[i1, k1] * B[i2, k2]
            Eigen::MatrixXd A = G1.mode_slice(j);
            Eigen::MatrixXd B = G2.mode_slice(j);

            Eigen::MatrixXd kron(rl, rr);
            for (int i1 = 0; i1 < G1.r_left; ++i1)
                for (int i2 = 0; i2 < G2.r_left; ++i2)
                    for (int k1 = 0; k1 < G1.r_right; ++k1)
                        for (int k2 = 0; k2 < G2.r_right; ++k2)
                            kron(i1 * G2.r_left + i2, k1 * G2.r_right + k2)
                                = A(i1, k1) * B(i2, k2);
            nc.set_mode_slice(j, kron);
        }
        new_cores.push_back(nc);
    }
    return TT(std::move(new_cores));
}

// ── at() ──────────────────────────────────────────────────────────────────────
double TT::at(const std::vector<int>& indices) const {
    int d = ndim();
    auto sh = shape();

    if ((int)indices.size() != d)
        throw std::out_of_range("at(): wrong number of indices");
    for (int k = 0; k < d; ++k)
        if (indices[k] < 0 || indices[k] >= sh[k])
            throw std::out_of_range("at(): index out of bounds");

    // Start: cores[0][0, i0, :]  →  row vector (1 × r0)
    Eigen::RowVectorXd v = cores[0].mode_slice(indices[0]).row(0);

    for (int k = 1; k < d - 1; ++k)
        v = v * cores[k].mode_slice(indices[k]);   // 1 × r_k

    // Last core: r_{d-1} × 1
    Eigen::MatrixXd last = cores[d-1].mode_slice(indices[d-1]);  // r_{d-1} × 1
    return (v * last)(0, 0);
}

// ── round() ──────────────────────────────────────────────────────────────────
void TT::round(std::optional<vector<int>> ranks_opt, double eps, int max_rank)
{
  int d = ndim();
  if (ranks_opt && static_cast<int>(ranks_opt->size()) != d - 1) {
    throw std::invalid_argument("round(): ranks length must be d-1");
  }

  SVDWorkspace workspace;
  Eigen::MatrixXd product;

  // Right-to-left orthogonalization. The maps borrow core storage only until
  // the corresponding contraction has finished, before assign_* resizes it.
  for (int k = d - 1; k > 0; --k) {
    auto matrix = right_unfolding(cores[k]);
    workspace.qr_.compute(matrix.transpose());
    int rank = std::min(matrix.rows(), matrix.cols());
    workspace.lifted_.setIdentity(matrix.cols(), rank);
    workspace.lifted_.applyOnTheLeft(workspace.qr_.householderQ());
    workspace.factor_ =
      workspace.qr_.matrixQR().topRows(rank).triangularView<Eigen::Upper>();

    assign_right(cores[k], workspace.lifted_.transpose(), rank, cores[k].n,
      cores[k].r_right);
    product.noalias() =
      left_unfolding(cores[k - 1]) * workspace.factor_.transpose();
    assign_left(
      cores[k - 1], product, cores[k - 1].r_left, cores[k - 1].n, rank);
  }

  // Left-to-right SVD truncation, keeping the original per-bond tolerance and
  // rank cap. The retained factors are views into the reusable workspace.
  for (int k = 0; k < d - 1; ++k) {
    workspace.compute(left_unfolding(cores[k]),
      ranks_opt ? std::nullopt : std::optional<double>(eps),
      ranks_opt ? std::min((*ranks_opt)[k], max_rank) : max_rank);
    int rank = workspace.rank_;
    assign_left(
      cores[k], workspace.matrix_u(), cores[k].r_left, cores[k].n, rank);

    product.noalias() =
      workspace.matrix_v().transpose() * right_unfolding(cores[k + 1]);
    for (int i = 0; i < rank; ++i) {
      product.row(i) *= workspace.singular_values()(i);
    }
    assign_right(
      cores[k + 1], product, rank, cores[k + 1].n, cores[k + 1].r_right);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// tt_svd
// ─────────────────────────────────────────────────────────────────────────────

TT tt_svd(const std::vector<double>& tensor,
          const std::vector<int>&    shape,
          std::vector<int>           ranks,
          double                     eps) {
    int d = static_cast<int>(shape.size());

    if (ranks.empty() && eps < 0.0) eps = 1e-10;
    if (!ranks.empty() && eps > 0.0)
        throw std::invalid_argument("Specify either ranks or eps, not both");
    if (!ranks.empty() && (int)ranks.size() != d - 1)
        throw std::invalid_argument("ranks must have length d-1");

    if (shape.empty())
      throw std::invalid_argument("tt_svd: shape must not be empty");
    return decompose_buffer(tensor, shape, ranks, eps);
}

TT tt_svd_tally_value(const double* value, int size, const vector<int>& shape,
  double norm, bool square, double eps)
{
  if (value == nullptr) {
    throw std::invalid_argument("tt_svd_tally_value: value pointer is null");
  }
  if (shape.empty() || size <= 0) {
    throw std::invalid_argument("tt_svd_tally_value: invalid shape");
  }
  int expected_size = 1;
  for (int n : shape) {
    if (n <= 0) {
      throw std::invalid_argument("tt_svd_tally_value: invalid shape");
    }
    if (expected_size > size / n) {
      throw std::invalid_argument("tt_svd_tally_value: shape/size mismatch");
    }
    expected_size *= n;
  }
  if (expected_size != size) {
    throw std::invalid_argument("tt_svd_tally_value: shape/size mismatch");
  }
  if (eps < 0.0)
    eps = 1e-10;

  vector<double> buf(size);
  for (int i = 0; i < size; ++i) {
    double v = value[i] * norm;
    buf[i] = square ? v * v : v;
  }

  return decompose_buffer(std::move(buf), shape, {}, eps);
}

TT tt_rand(const std::vector<double>& tensor,
           const std::vector<int>&    shape,
           std::vector<int>           ranks,
           double                     eps,
           int                        max_rank,
           int                        n_oversamples,
           int                        n_iter,
           int                        random_state) {
    int d = static_cast<int>(shape.size());

    if (ranks.empty() && eps < 0.0) eps = 1e-10;
    if (!ranks.empty() && eps > 0.0)
        throw std::invalid_argument("Specify either ranks or eps, not both");
    if (!ranks.empty() && (int)ranks.size() != d - 1)
        throw std::invalid_argument("ranks must have length d-1");

    std::vector<double> buf(tensor);

    std::vector<Core> cores;
    cores.reserve(d);
    int r_prev = 1;

    for (int k = 0; k < d - 1; ++k) {
        int n_k = shape[k];
        int rows = r_prev * n_k;
        int cols = static_cast<int>(buf.size()) / rows;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            mat_rm = Eigen::Map<const Eigen::Matrix<double,
                                                    Eigen::Dynamic, Eigen::Dynamic,
                                                    Eigen::RowMajor>>(buf.data(), rows, cols);
        Eigen::MatrixXd mat = mat_rm;

        int init_rank = std::min(rows, cols);
        int r_k;
        SVDResult svd;

        if (!ranks.empty()) {
            r_k = std::min({init_rank, ranks[k], max_rank});
            svd = randomized_svd(mat, r_k, n_oversamples, n_iter, random_state);
        } else {
            r_k = std::min(init_rank, max_rank);
            svd = randomized_svd(mat, r_k, n_oversamples, n_iter, random_state);
            r_k = std::min(rank_truncation(svd.S, eps), r_k);
            r_k = std::max(r_k, 1);
            svd.U  = svd.U.leftCols(r_k);
            svd.S  = svd.S.head(r_k);
            svd.Vt = svd.Vt.topRows(r_k);
        }

        Core core(r_prev, n_k, r_k);
        for (int i = 0; i < r_prev; ++i)
            for (int j = 0; j < n_k; ++j)
                for (int l = 0; l < r_k; ++l)
                    core(i, j, l) = svd.U(i * n_k + j, l);
        cores.push_back(core);

        buf.resize(r_k * cols);
        for (int i = 0; i < r_k; ++i)
            for (int j = 0; j < cols; ++j)
                buf[i * cols + j] = svd.S(i) * svd.Vt(i, j);

        r_prev = r_k;
    }

    int n_last = shape[d-1];
    Core last_core(r_prev, n_last, 1);
    for (int i = 0; i < r_prev; ++i)
        for (int j = 0; j < n_last; ++j)
            last_core(i, j, 0) = buf[i * n_last + j];
    cores.push_back(last_core);

    return TT(std::move(cores));
}


TT tt_zeros(const std::vector<int>& shape) {
    std::vector<Core> cores;
    cores.reserve(shape.size());
    for (int n : shape)
        cores.emplace_back(1, n, 1);   // all-zero rank-1 cores
    return TT(std::move(cores));
}

std::vector<int> prime_factors(int n) {
    std::vector<int> factors;
    if (n <= 1) return factors;
    while (n % 2 == 0) {
        factors.push_back(2);
        n /= 2;
    }
    for (int p = 3; p * p <= n; p += 2) {
        while (n % p == 0) {
            factors.push_back(p);
            n /= p;
        }
    }
    if (n > 1) factors.push_back(n);
    return factors;
}

std::vector<int> balanced_groups_no_empty(int n, int order) {
    if (order <= 0)
        throw std::invalid_argument("auto_tt_shape: order must be positive");
    if (n <= 1) return std::vector<int>(order, 1);

    auto primes = prime_factors(n);
    std::sort(primes.rbegin(), primes.rend());

    std::vector<int> groups(order, 1);
    std::vector<double> logs(order, 0.0);
    for (int p : primes) {
        int j = static_cast<int>(
            std::min_element(logs.begin(), logs.end()) - logs.begin());
        groups[j] *= p;
        logs[j] += std::log(static_cast<double>(p));
    }
    std::sort(groups.begin(), groups.end());
    return groups;
}

double score_dims_list(
    const std::vector<int>& dims, int site_min, int site_cap) {
    double penalty = 0.0;
    std::vector<double> logs;
    logs.reserve(dims.size());

    for (int n : dims) {
        if (n <= 0)
            throw std::invalid_argument("auto_tt_shape: dimensions must be positive");
        if (n == 1) {
            penalty += 2.0;
        } else {
            if (n < site_min) penalty += static_cast<double>(site_min - n) / site_min;
            if (n > site_cap) penalty += static_cast<double>(n - site_cap) / site_cap;
            logs.push_back(std::log(static_cast<double>(n)));
        }
    }

    if (logs.size() >= 2) {
        double mu = std::accumulate(logs.begin(), logs.end(), 0.0) / logs.size();
        double var = 0.0;
        for (double x : logs) var += (x - mu) * (x - mu);
        penalty += var / logs.size();
    }
    return penalty;
}

std::vector<int> auto_tt_shape(int n,
                               int site_min,
                               int site_cap,
                               int prefer_order,
                               int min_order,
                               int max_order) {
    if (n <= 0)
        throw std::invalid_argument("auto_tt_shape: n must be positive");
    if (site_min <= 1 || site_cap < site_min || prefer_order <= 0 ||
        min_order <= 0 || max_order < min_order)
        throw std::invalid_argument("auto_tt_shape: invalid parameters");
    if (n <= 1) return {1};
    if (n <= site_cap) return {n};

    double best_score = std::numeric_limits<double>::infinity();
    std::vector<int> best_dims;
    for (int d = min_order; d <= max_order; ++d) {
        auto dims = balanced_groups_no_empty(n, d);
        double s = score_dims_list(dims, site_min, site_cap) +
                   0.1 * std::abs(d - prefer_order);
        if (s < best_score) {
            best_score = s;
            best_dims = std::move(dims);
        }
    }
    return best_dims;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

double compute_compression_ratio(const TT& tt) {
    int original_size = 1;
    for (int n : tt.shape()) original_size *= n;
    auto tt_size = tt_storage_size(tt);
    return static_cast<double>(original_size) / tt_size;
}

std::size_t tt_storage_size(const TT& tt) {
    std::size_t size = 0;
    for (const auto& c : tt.cores) size += c.data.size();
    return size;
}

std::size_t tt_storage_bytes(const TT& tt) {
    return tt_storage_size(tt) * sizeof(double);
}

double compute_relative_error(const std::vector<double>& original, const TT& tt) {
    auto [reconstructed, sh] = tt.full();

    double diff_sq = 0.0, orig_sq = 0.0;
    for (int i = 0; i < (int)original.size(); ++i) {
        double d = original[i] - reconstructed[i];
        diff_sq += d * d;
        orig_sq += original[i] * original[i];
    }
    return std::sqrt(diff_sq) / std::sqrt(orig_sq);
}

} // namespace openmc
