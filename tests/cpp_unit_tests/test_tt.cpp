#include "openmc/tt.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "openmc/vector.h"

using namespace openmc;

namespace {

void check_reconstruction(const TT& tt, const vector<double>& expected,
  const vector<int>& shape, double tolerance = 1.0e-12)
{
  const auto [actual, actual_shape] = tt.full();
  REQUIRE(actual_shape == shape);
  REQUIRE(actual.size() == expected.size());
  REQUIRE(tt.cores.front().r_left == 1);
  REQUIRE(tt.cores.back().r_right == 1);
  for (std::size_t i = 1; i < tt.cores.size(); ++i) {
    REQUIRE(tt.cores[i - 1].r_right == tt.cores[i].r_left);
  }

  double norm_squared = 0.0;
  double error_squared = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    norm_squared += expected[i] * expected[i];
    const double error = actual[i] - expected[i];
    error_squared += error * error;
  }
  REQUIRE(std::sqrt(error_squared) <=
          tolerance * std::max(1.0, std::sqrt(norm_squared)));
}

vector<double> matrix_values(const Eigen::MatrixXd& matrix)
{
  vector<double> values(matrix.size());
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      values[i * matrix.cols() + j] = matrix(i, j);
    }
  }
  return values;
}

vector<double> dense_values(const vector<int>& shape)
{
  const int size =
    std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  vector<double> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = std::sin(0.13 * (i + 1)) + 0.3 * std::cos(0.07 * i * i);
  }
  return values;
}

} // namespace

TEST_CASE("TT SVD reconstructs tall, wide, and square matrices")
{
  for (const auto& shape :
    vector<vector<int>> {{9, 3}, {3, 9}, {5, 5}, {1, 7}, {7, 1}}) {
    INFO("matrix shape: " << shape[0] << " x " << shape[1]);
    Eigen::MatrixXd matrix(shape[0], shape[1]);
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        matrix(i, j) = std::sin((i + 1.0) * (j + 2.0)) + 0.1 * (i - j);
      }
    }

    const auto svd = fast_svd(matrix);
    const int rank = std::min(shape[0], shape[1]);
    REQUIRE(svd.S.size() == rank);
    REQUIRE(svd.U.rows() == shape[0]);
    REQUIRE(svd.U.cols() == rank);
    REQUIRE(svd.Vt.rows() == rank);
    REQUIRE(svd.Vt.cols() == shape[1]);
    REQUIRE((matrix - svd.U * svd.S.asDiagonal() * svd.Vt).norm() <=
            1.0e-12 * matrix.norm());
    REQUIRE((svd.U.transpose() * svd.U - Eigen::MatrixXd::Identity(rank, rank))
              .norm() <= 1.0e-12);
    REQUIRE(
      (svd.Vt * svd.Vt.transpose() - Eigen::MatrixXd::Identity(rank, rank))
        .norm() <= 1.0e-12);

    const auto values = matrix_values(matrix);
    check_reconstruction(tt_svd(values, shape, {}, 0.0), values, shape);
  }
}

TEST_CASE("TT decomposition and rounding preserve nonuniform tensors")
{
  const vector<int> shape {2, 3, 4, 2};
  const auto first = dense_values(shape);
  vector<double> second(first.size());
  vector<double> expected(first.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    second[i] = std::cos(0.31 * i) - 0.2 * std::sin(0.11 * i * i);
    expected[i] = first[i] + second[i];
  }

  const auto first_tt = tt_svd(first, shape, {}, 1.0e-14);
  const auto second_tt = tt_svd(second, shape, {}, 1.0e-14);
  check_reconstruction(first_tt, first, shape);
  check_reconstruction(second_tt, second, shape);
  auto sum = first_tt + second_tt;
  sum.round(std::nullopt, 1.0e-12);
  check_reconstruction(sum, expected, shape, 2.0e-12);
}

TEST_CASE("TT zero and rank-one tensors retain minimal ranks")
{
  const vector<int> shape {2, 3, 5};
  vector<double> values(30, 0.0);

  SECTION("zero tensor")
  {
    auto tt = tt_svd(values, shape, {}, 1.0e-12);
    tt.round(std::nullopt, 1.0e-12);
    REQUIRE(tt.ranks() == vector<int> {1, 1, 1, 1});
    REQUIRE(tt.full().first == values);

    auto zero = tt_zeros(shape);
    zero.round(std::nullopt, 1.0e-12);
    REQUIRE(zero.ranks() == vector<int> {1, 1, 1, 1});
    REQUIRE(zero.full().first == values);
  }

  SECTION("separable tensor")
  {
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        for (int k = 0; k < shape[2]; ++k) {
          values[(i * shape[1] + j) * shape[2] + k] =
            (i + 1.0) * (j + 2.0) * (k - 2.0);
        }
      }
    }
    auto tt = tt_svd(values, shape, {}, 1.0e-12);
    REQUIRE(tt.ranks() == vector<int> {1, 1, 1, 1});
    tt.round(std::nullopt, 1.0e-12);
    REQUIRE(tt.ranks() == vector<int> {1, 1, 1, 1});
    check_reconstruction(tt, values, shape);
  }
}

TEST_CASE("TT decomposition and rounding support a single mode")
{
  const vector<int> shape {5};
  const vector<double> values {1.0, -3.0, 0.0, 2.5, 1.0e-8};
  auto tt = tt_svd(values, shape, {}, 1.0e-12);
  tt.round(std::nullopt, 1.0e-12);
  REQUIRE(tt.ranks() == vector<int> {1, 1});
  check_reconstruction(tt, values, shape);
}

TEST_CASE("TT addition preserves boundary ranks for a single mode")
{
  const vector<int> shape {5};
  const vector<double> values_a {1.0, -3.0, 0.0, 2.5, 1.0e-8};
  const vector<double> values_b {2.0, 1.0, -4.0, 0.5, 3.0};
  auto tt_a = tt_svd(values_a, shape);
  auto tt_b = tt_svd(values_b, shape);

  auto sum = tt_a + tt_b;

  REQUIRE(sum.ranks() == vector<int> {1, 1});
  vector<double> expected(values_a.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = values_a[i] + values_b[i];
  }
  check_reconstruction(sum, expected, shape);
}

TEST_CASE("TT prescribed ranks and rank caps retain leading singular modes")
{
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(5, 5);
  matrix.diagonal() << 4.0, 2.0, 0.5, 0.1, 0.01;
  const vector<int> shape {5, 5};
  const auto values = matrix_values(matrix);
  auto tt = tt_svd(values, shape, {}, 0.0);
  int expected_rank = 2;

  SECTION("fixed decomposition rank")
  {
    tt = tt_svd(values, shape, {2});
  }
  SECTION("fixed rounding rank")
  {
    tt.round(vector<int> {2});
  }
  SECTION("rounding rank cap")
  {
    tt.round(std::nullopt, 0.0, 2);
  }
  SECTION("cap also limits a prescribed rank")
  {
    tt.round(vector<int> {3}, 0.0, 1);
    expected_rank = 1;
  }

  REQUIRE(tt.ranks() == vector<int> {1, expected_rank, 1});
  for (int i = expected_rank; i < matrix.rows(); ++i) {
    matrix(i, i) = 0.0;
  }
  check_reconstruction(tt, matrix_values(matrix), shape);
}

TEST_CASE("TT truncation preserves small resolved singular values")
{
  Eigen::Matrix4d rotation;
  rotation << 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
    1.0, -1.0, -1.0, 1.0;
  rotation *= 0.5;
  Eigen::Vector4d spectrum;
  spectrum << 1.0, 1.0e-4, 1.0e-8, 1.0e-12;
  const Eigen::MatrixXd matrix =
    rotation * spectrum.asDiagonal() * rotation.transpose();
  const auto values = matrix_values(matrix);
  const vector<int> shape {4, 4};

  for (const double eps : {1.0e-6, 1.0e-10}) {
    INFO("relative tolerance: " << eps);
    const int expected_rank = eps == 1.0e-6 ? 2 : 3;
    auto retained = spectrum;
    retained.tail(4 - expected_rank).setZero();
    const Eigen::MatrixXd expected =
      rotation * retained.asDiagonal() * rotation.transpose();

    auto decomposed = tt_svd(values, shape, {}, eps);
    auto tally =
      tt_svd_tally_value(values.data(), values.size(), shape, 1.0, false, eps);
    auto rounded = tt_svd(values, shape, {}, 0.0);
    rounded.round(std::nullopt, eps);
    for (const auto* tt : {&decomposed, &tally, &rounded}) {
      REQUIRE(tt->ranks() == vector<int> {1, expected_rank, 1});
      check_reconstruction(*tt, matrix_values(expected), shape, 1.0e-13);
      check_reconstruction(*tt, values, shape, eps);
    }
  }
}

TEST_CASE("TT tally decomposition normalizes and squares immutable input")
{
  for (const auto& shape : vector<vector<int>> {{24}, {3, 8}, {2, 3, 4}}) {
    const auto values = dense_values(shape);
    const auto original = values;
    for (const bool square : {false, true}) {
      vector<double> expected(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double normalized = values[i] * 0.25;
        expected[i] = square ? normalized * normalized : normalized;
      }
      const auto tt = tt_svd_tally_value(
        values.data(), values.size(), shape, 0.25, square, 1.0e-13);
      check_reconstruction(tt, expected, shape);
      REQUIRE(values == original);
    }

    const auto zero = tt_svd_tally_value(
      values.data(), values.size(), shape, 0.0, false, 1.0e-13);
    REQUIRE(zero.full().first == vector<double>(values.size(), 0.0));
    REQUIRE(values == original);
  }
}

TEST_CASE("TT tally truncates low-rank tall and wide unfoldings")
{
  for (const auto& shape : vector<vector<int>> {{9, 3}, {3, 9}, {5, 5}}) {
    vector<double> values(shape[0] * shape[1]);
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < shape[1]; ++j) {
        values[i * shape[1] + j] = (i - 2.0) * (j + 0.5);
      }
    }
    for (const bool square : {false, true}) {
      vector<double> expected = values;
      for (double& value : expected) {
        value *= 0.25;
        if (square) {
          value *= value;
        }
      }
      const auto tt = tt_svd_tally_value(
        values.data(), values.size(), shape, 0.25, square, 1.0e-12);
      REQUIRE(tt.ranks() == vector<int> {1, 1, 1});
      check_reconstruction(tt, expected, shape);
    }
  }
}
