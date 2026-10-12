#include "openmc/tt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

namespace openmc {

namespace {

using RowMatrix =
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstCoreMap = Eigen::Map<const RowMatrix>;
using ThinBDCSVD = Eigen::BDCSVD<Eigen::MatrixXd,
  Eigen::ComputeThinU | Eigen::ComputeThinV>;

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
    int max_rank = std::numeric_limits<int>::max())
  {
    int rows = matrix.rows();
    int cols = matrix.cols();
    tall_ = rows >= 1.5 * cols;
    wide_ = !tall_ && cols >= 1.5 * rows;

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

    svd_.compute(factor_);
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
  ThinBDCSVD svd_;
  Eigen::MatrixXd factor_;
  Eigen::MatrixXd lifted_;
  int rank_ {0};
  bool tall_ {false};
  bool wide_ {false};
};

// buf stays in row-major order between successive TT unfoldings.
TT decompose_buffer(vector<double> buf, const vector<int>& shape, double eps)
{
  SVDWorkspace workspace;
  vector<Core> cores;
  cores.reserve(shape.size());
  int r_prev = 1;

  for (int k = 0; k < static_cast<int>(shape.size()) - 1; ++k) {
    int rows = r_prev * shape[k];
    int cols = buf.size() / rows;
    ConstCoreMap matrix(buf.data(), rows, cols);
    workspace.compute(matrix, eps);
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

  return decompose_buffer(std::move(buf), shape, eps);
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

std::size_t tt_storage_size(const TT& tt) {
    std::size_t size = 0;
    for (const auto& c : tt.cores) size += c.data.size();
    return size;
}

std::size_t tt_storage_bytes(const TT& tt) {
    return tt_storage_size(tt) * sizeof(double);
}

} // namespace openmc
