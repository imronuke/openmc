#include "openmc/tt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <Eigen/Dense>

namespace openmc {

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
Eigen::MatrixXd Core::to_matrix_left() const {
    Eigen::MatrixXd M(r_left * n, r_right);
    for (int i = 0; i < r_left; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < r_right; ++k)
                M(i * n + j, k) = (*this)(i, j, k);
    return M;
}

// Reshape to r_left × (n * r_right)  (right-unfolding)
Eigen::MatrixXd Core::to_matrix_right() const {
    Eigen::MatrixXd M(r_left, n * r_right);
    for (int i = 0; i < r_left; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < r_right; ++k)
                M(i, j * r_right + k) = (*this)(i, j, k);
    return M;
}

// Fill core from (r_left*n) × r_right matrix
void Core::from_matrix_left(const Eigen::MatrixXd& M,
                             int r_left_, int n_, int r_right_) {
    r_left = r_left_; n = n_; r_right = r_right_;
    data.resize(r_left * n * r_right);
    for (int i = 0; i < r_left; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < r_right; ++k)
                (*this)(i, j, k) = M(i * n + j, k);
}

// Fill core from r_left × (n*r_right) matrix
void Core::from_matrix_right(const Eigen::MatrixXd& M,
                              int r_left_, int n_, int r_right_) {
    r_left = r_left_; n = n_; r_right = r_right_;
    data.resize(r_left * n * r_right);
    for (int i = 0; i < r_left; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < r_right; ++k)
                (*this)(i, j, k) = M(i, j * r_right + k);
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

SVDResult fast_svd(const Eigen::MatrixXd& X, double aspect) {
    int m = static_cast<int>(X.rows());
    int n = static_cast<int>(X.cols());

    SVDResult res;

    // Tall case: X = Q R, then SVD(R)
    if (m >= aspect * n) {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);
        Eigen::MatrixXd Q = qr.householderQ() *
                            Eigen::MatrixXd::Identity(m, n);   // thin Q
        Eigen::MatrixXd R = Q.transpose() * X;

        Eigen::BDCSVD<Eigen::MatrixXd> svd(R,
            Eigen::ComputeThinU | Eigen::ComputeThinV);
        res.U  = Q * svd.matrixU();
        res.S  = svd.singularValues();
        res.Vt = svd.matrixV().transpose();
        return res;
    }

    // Wide case: X^T = Q R, then SVD(R^T)
    if (n >= aspect * m) {
        Eigen::MatrixXd Xt = X.transpose();
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Xt);
        Eigen::MatrixXd Q = qr.householderQ() *
                            Eigen::MatrixXd::Identity(n, m);
        Eigen::MatrixXd R = Q.transpose() * Xt;

        Eigen::BDCSVD<Eigen::MatrixXd> svd(R.transpose(),
            Eigen::ComputeThinU | Eigen::ComputeThinV);
        res.U  = svd.matrixU();
        res.S  = svd.singularValues();
        res.Vt = (svd.matrixV().transpose()) * Q.transpose();
        return res;
    }

    // Moderately shaped
    Eigen::BDCSVD<Eigen::MatrixXd> svd(X,
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    res.U  = svd.matrixU();
    res.S  = svd.singularValues();
    res.Vt = svd.matrixV().transpose();
    return res;
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
            new_cores.push_back(nc);

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
            new_cores.push_back(nc);

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
            new_cores.push_back(nc);
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
void TT::round(std::optional<std::vector<int>> ranks_opt,
               double eps,
               int    max_rank) {
    int d = ndim();
    auto sh = shape();

    std::vector<int> ranks_list;
    bool use_ranks = ranks_opt.has_value();
    if (use_ranks) {
        ranks_list = ranks_opt.value();
        if ((int)ranks_list.size() != d - 1)
            throw std::invalid_argument("round(): ranks length must be d-1");
    }

    // Helper: build (r_left * n) x r_right matrix from core (row outer = r_left, then n)
    auto core_to_left_mat = [](const Core& c) -> Eigen::MatrixXd {
        Eigen::MatrixXd M(c.r_left * c.n, c.r_right);
        for (int i = 0; i < c.r_left; ++i)
            for (int j = 0; j < c.n; ++j)
                M.row(i * c.n + j) = c.mode_slice(j).row(i);
        return M;
    };

    // Helper: build r_left x (n * r_right) matrix (n outer, r_right inner)
    auto core_to_right_mat = [](const Core& c) -> Eigen::MatrixXd {
        Eigen::MatrixXd M(c.r_left, c.n * c.r_right);
        for (int i = 0; i < c.r_left; ++i)
            for (int j = 0; j < c.n; ++j)
                M.block(i, j * c.r_right, 1, c.r_right) = c.mode_slice(j).row(i);
        return M;
    };

    // Helper: fill core from (r_left * n) x r_right matrix
    auto left_mat_to_core = [](Core& c, const Eigen::MatrixXd& M,
                                int rl, int n, int rr) {
        c.r_left = rl; c.n = n; c.r_right = rr;
        c.data.resize(rl * n * rr);
        for (int i = 0; i < rl; ++i)
            for (int j = 0; j < n; ++j)
                for (int l = 0; l < rr; ++l)
                    c(i, j, l) = M(i * n + j, l);
    };

    // Helper: fill core from r_left x (n * r_right) matrix
    auto right_mat_to_core = [](Core& c, const Eigen::MatrixXd& M,
                                 int rl, int n, int rr) {
        c.r_left = rl; c.n = n; c.r_right = rr;
        c.data.resize(rl * n * rr);
        for (int i = 0; i < rl; ++i)
            for (int j = 0; j < n; ++j)
                for (int l = 0; l < rr; ++l)
                    c(i, j, l) = M(i, j * rr + l);
    };

    // ── Step 1: Right-to-left orthogonalisation ──────────────────────────────
    for (int k = d - 1; k > 0; --k) {
        // mat: r_left x (n * r_right)  (right unfolding)
        Eigen::MatrixXd mat = core_to_right_mat(cores[k]);

        // QR of mat^T: (n*r_right) x r_left = Q * R
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(mat.transpose());
        int min_dim = std::min(mat.rows(), mat.cols());
        Eigen::MatrixXd Q = qr.householderQ() *
                            Eigen::MatrixXd::Identity(mat.cols(), min_dim);
        Eigen::MatrixXd R = Q.transpose() * mat.transpose();  // min_dim x r_left

        // cores[k] <- Q^T reshaped as (min_dim, n_k, r_right)
        right_mat_to_core(cores[k], Q.transpose(), min_dim, sh[k], cores[k].r_right);

        // Merge R^T into cores[k-1]:  core[k-1] @ R^T along right rank
        // prev_mat: (r_prev_left * n_{k-1}) x old_r_right
        Eigen::MatrixXd prev_mat = core_to_left_mat(cores[k-1]);
        // R: min_dim x old_r_right  ->  R^T: old_r_right x min_dim
        Eigen::MatrixXd new_prev = prev_mat * R.transpose();
        left_mat_to_core(cores[k-1], new_prev, cores[k-1].r_left, sh[k-1], min_dim);
    }

    // ── Step 2: Left-to-right SVD truncation ─────────────────────────────────
    for (int k = 0; k < d - 1; ++k) {
        Eigen::MatrixXd mat = core_to_left_mat(cores[k]);  // (r_left*n) x r_right
        int init_rank = std::min(mat.rows(), mat.cols());

        SVDResult svd = fast_svd(mat);

        int r_new;
        if (use_ranks) {
            r_new = std::min({init_rank, ranks_list[k], max_rank});
        } else {
            r_new = std::min(init_rank, max_rank);
            r_new = std::min(rank_truncation(svd.S, eps), r_new);
        }
        r_new = std::max(r_new, 1);

        Eigen::MatrixXd U  = svd.U.leftCols(r_new);
        Eigen::VectorXd S  = svd.S.head(r_new);
        Eigen::MatrixXd Vt = svd.Vt.topRows(r_new);

        // Update cores[k] with U*S
        Eigen::MatrixXd US = U * S.asDiagonal();
        left_mat_to_core(cores[k], US, cores[k].r_left, sh[k], r_new);

        // Push Vt into cores[k+1]
        // cores[k+1] currently: old_r x (n_{k+1} * r_{k+2})  (right unfolding)
        // Vt: r_new x old_r   ->  new cores[k+1] mat: r_new x (n_{k+1} * r_{k+2})
        Eigen::MatrixXd next_right = core_to_right_mat(cores[k+1]);
        Eigen::MatrixXd new_next   = Vt * next_right;
        right_mat_to_core(cores[k+1], new_next, r_new, sh[k+1], cores[k+1].r_right);
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

    // Keep data as a flat row-major buffer throughout.
    // At step k, buf has logical shape (r_prev, n_k, n_{k+1}, ..., n_{d-1}).
    std::vector<double> buf(tensor);

    std::vector<Core> cores;
    cores.reserve(d);
    int r_prev = 1;

    for (int k = 0; k < d - 1; ++k) {
        int n_k = shape[k];
        int rows = r_prev * n_k;
        int cols = static_cast<int>(buf.size()) / rows;

        // Map as RowMajor so reshape matches numpy semantics
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            mat_rm = Eigen::Map<const Eigen::Matrix<double,
                                                    Eigen::Dynamic, Eigen::Dynamic,
                                                    Eigen::RowMajor>>(buf.data(), rows, cols);
        Eigen::MatrixXd mat = mat_rm;   // convert to ColMajor for SVD

        SVDResult svd = fast_svd(mat);

        int r_k;
        if (!ranks.empty()) {
            r_k = std::min((int)svd.S.size(), ranks[k]);
        } else {
            r_k = rank_truncation(svd.S, eps);
            r_k = std::max(1, std::min(r_k, (int)svd.S.size()));
        }

        Eigen::MatrixXd U  = svd.U.leftCols(r_k);
        Eigen::VectorXd S  = svd.S.head(r_k);
        Eigen::MatrixXd Vt = svd.Vt.topRows(r_k);

        // Store core k: from US matrix (r_prev*n_k) x r_k, row-major
        // core(i, j, l) = US(i*n_k + j, l)
        Eigen::MatrixXd US = U * S.asDiagonal();
        Core core(r_prev, n_k, r_k);
        for (int i = 0; i < r_prev; ++i)
            for (int j = 0; j < n_k; ++j)
                for (int l = 0; l < r_k; ++l)
                    core(i, j, l) = US(i * n_k + j, l);
        cores.push_back(core);

        // Store Vt as row-major buffer for next iteration
        // Vt: r_k x cols (ColMajor). Convert to row-major buffer.
        buf.resize(r_k * cols);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            buf.data(), r_k, cols) = Vt;

        r_prev = r_k;
    }

    // Last core: r_prev x n_{d-1} x 1
    // buf is row-major r_prev x n_{d-1}
    int n_last = shape[d-1];
    Core last_core(r_prev, n_last, 1);
    for (int i = 0; i < r_prev; ++i)
        for (int j = 0; j < n_last; ++j)
            last_core(i, j, 0) = buf[i * n_last + j];
    cores.push_back(last_core);

    return TT(std::move(cores));
}

TT tt_svd_tally_value(const double* value,
                      int size,
                      const std::vector<int>& shape,
                      double norm,
                      bool square,
                      double eps) {
    if (value == nullptr)
        throw std::invalid_argument("tt_svd_tally_value: value pointer is null");

    int expected_size = 1;
    for (int n : shape) {
        if (n <= 0)
            throw std::invalid_argument("tt_svd_tally_value: invalid shape");
        expected_size *= n;
    }
    if (expected_size != size)
        throw std::invalid_argument("tt_svd_tally_value: shape/size mismatch");

    int d = static_cast<int>(shape.size());
    if (eps < 0.0) eps = 1e-10;

    std::vector<double> buf(size);
    for (int i = 0; i < size; ++i) {
        double v = value[i] * norm;
        buf[i] = square ? v * v : v;
    }

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

        SVDResult svd = fast_svd(mat);

        int r_k = rank_truncation(svd.S, eps);
        r_k = std::max(1, std::min(r_k, static_cast<int>(svd.S.size())));

        Eigen::MatrixXd U  = svd.U.leftCols(r_k);
        Eigen::VectorXd S  = svd.S.head(r_k);
        Eigen::MatrixXd Vt = svd.Vt.topRows(r_k);

        Eigen::MatrixXd US = U * S.asDiagonal();
        Core core(r_prev, n_k, r_k);
        for (int i = 0; i < r_prev; ++i)
            for (int j = 0; j < n_k; ++j)
                for (int l = 0; l < r_k; ++l)
                    core(i, j, l) = US(i * n_k + j, l);
        cores.push_back(core);

        buf.resize(r_k * cols);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            buf.data(), r_k, cols) = Vt;

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

        Eigen::MatrixXd US = svd.U * svd.S.asDiagonal();
        Core core(r_prev, n_k, r_k);
        for (int i = 0; i < r_prev; ++i)
            for (int j = 0; j < n_k; ++j)
                for (int l = 0; l < r_k; ++l)
                    core(i, j, l) = US(i * n_k + j, l);
        cores.push_back(core);

        buf.resize(r_k * cols);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            buf.data(), r_k, cols) = svd.Vt;

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

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

double compute_compression_ratio(const TT& tt) {
    int original_size = 1;
    for (int n : tt.shape()) original_size *= n;
    int tt_size = 0;
    for (const auto& c : tt.cores) tt_size += c.size();
    return static_cast<double>(original_size) / tt_size;
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
