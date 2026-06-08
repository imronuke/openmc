#include <cstddef>
#include <stdexcept>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/particle.h"
#include "openmc/settings.h"
#include "openmc/tt_density.h"

using namespace openmc;

namespace {

class TTTestState {
public:
  TTTestState()
    : event_based_ {settings::event_based}, run_CE_ {settings::run_CE},
      n_procs_ {mpi::n_procs}, n_coord_levels_ {model::n_coord_levels}
  {
    settings::event_based = false;
    settings::run_CE = true;
    mpi::n_procs = 1;
    model::n_coord_levels = 1;
    model::atom_density_tt.clear();
    model::materials.clear();
    model::material_map.clear();
  }

  ~TTTestState()
  {
    model::atom_density_tt.clear();
    model::materials.clear();
    model::material_map.clear();
    settings::event_based = event_based_;
    settings::run_CE = run_CE_;
    mpi::n_procs = n_procs_;
    model::n_coord_levels = n_coord_levels_;
  }

private:
  bool event_based_;
  bool run_CE_;
  int n_procs_;
  int n_coord_levels_;
};

} // namespace

TEST_CASE("AtomDensityTT reconstructs one material row")
{
  TTTestState state;

  vector<double> dense {1.0e-24, 2.0e-24, 3.0e-24,
    4.0e-24, 5.0e-24, 6.0e-24};
  TT density_tt = tt_svd(dense, {2, 3}, {}, 0.0);

  model::atom_density_tt.set_data({5, 7}, {"U235", "Xe135", "I135"},
    {10, C_NONE, 12}, {2}, {3}, std::move(density_tt));

  vector<double> density;
  model::atom_density_tt.reconstruct_material(7, density);

  vector<double> expected {4.0e-24, 5.0e-24, 6.0e-24};
  REQUIRE_THAT(density, Catch::Matchers::Approx(expected));
  REQUIRE(model::atom_density_tt.material_position(5) == 0);
  REQUIRE(model::atom_density_tt.contains_material(7));
  REQUIRE_FALSE(model::atom_density_tt.contains_material(9));
  REQUIRE(model::atom_density_tt.nuclide_position(12) == 2);
}

TEST_CASE("AtomDensityTT uses particle-local material density cache")
{
  TTTestState state;

  vector<double> dense {1.0e-24, 2.0e-24, 3.0e-24,
    4.0e-24, 5.0e-24, 6.0e-24};
  TT density_tt = tt_svd(dense, {2, 3}, {}, 0.0);

  model::atom_density_tt.set_data({5, 7}, {"U235", "Xe135", "I135"},
    {10, C_NONE, 12}, {2}, {3}, std::move(density_tt));

  Particle p;
  p.material() = 5;

  const auto& first_density = model::atom_density_tt.material_densities(p);
  vector<double> expected_first {1.0e-24, 2.0e-24, 3.0e-24};
  REQUIRE_THAT(first_density, Catch::Matchers::Approx(expected_first));
  REQUIRE(model::atom_density_tt.contains_material(p));

  double density {};
  REQUIRE(model::atom_density_tt.atom_density(p, 10, density));
  REQUIRE(density == Catch::Approx(1.0e-24));
  REQUIRE(model::atom_density_tt.atom_density(p, 12, density, 2.0));
  REQUIRE(density == Catch::Approx(6.0e-24));
  REQUIRE_FALSE(model::atom_density_tt.atom_density(p, 11, density));
  REQUIRE(density == 0.0);
  REQUIRE_FALSE(model::atom_density_tt.contains_nuclide(C_NONE));

  p.material() = 7;
  const auto& second_density = model::atom_density_tt.material_densities(p);
  vector<double> expected_second {4.0e-24, 5.0e-24, 6.0e-24};
  REQUIRE_THAT(second_density, Catch::Matchers::Approx(expected_second));

  vector<double> updated_dense {7.0e-24, 8.0e-24, 9.0e-24,
    10.0e-24, 11.0e-24, 12.0e-24};
  TT updated_density_tt = tt_svd(updated_dense, {2, 3}, {}, 0.0);
  model::atom_density_tt.set_data({5, 7}, {"U235", "Xe135", "I135"},
    {10, C_NONE, 12}, {2}, {3}, std::move(updated_density_tt));

  SourceSite site;
  site.particle = ParticleType::neutron();
  p.from_source(&site);
  p.material() = 7;
  const auto& updated_density = model::atom_density_tt.material_densities(p);
  vector<double> expected_updated {10.0e-24, 11.0e-24, 12.0e-24};
  REQUIRE_THAT(updated_density, Catch::Matchers::Approx(expected_updated));

  p.material() = 9;
  REQUIRE_FALSE(model::atom_density_tt.contains_material(p));
  REQUIRE_FALSE(model::atom_density_tt.atom_density(p, 10, density));
  REQUIRE_THROWS_AS(model::atom_density_tt.material_densities(p),
    std::out_of_range);
}

TEST_CASE("AtomDensityTT releases dense storage for TT-backed materials")
{
  TTTestState state;

  model::materials.push_back(make_unique<Material>());
  model::materials.push_back(make_unique<Material>());
  model::materials[0]->atom_density_ = tensor::Tensor<double>({2}, 0.0);
  model::materials[1]->atom_density_ = tensor::Tensor<double>({2}, 0.0);

  vector<double> dense {1.0e-24, 2.0e-24};
  TT density_tt = tt_svd(dense, {1, 2}, {}, 0.0);

  model::atom_density_tt.set_data(
    {0}, {"U235", "U238"}, {10, 11}, {1}, {2}, std::move(density_tt));

  REQUIRE(model::materials[0]->atom_density_.empty());
  REQUIRE_FALSE(model::materials[1]->atom_density_.empty());

  const int* nuclides {nullptr};
  const double* densities {nullptr};
  int n {};
  REQUIRE(openmc_material_get_densities(0, &nuclides, &densities, &n) ==
          OPENMC_E_ALLOCATE);

  vector<double> density;
  model::atom_density_tt.reconstruct_material(0, density);
  REQUIRE_THAT(density, Catch::Matchers::Approx(dense));
}

TEST_CASE("AtomDensityTT accepts C API transfer")
{
  TTTestState state;

  vector<double> dense {1.0e-24, 2.0e-24, 3.0e-24,
    4.0e-24, 5.0e-24, 6.0e-24};
  TT density_tt = tt_svd(dense, {2, 3}, {}, 0.0);

  vector<int> core_shapes;
  vector<size_t> core_offsets;
  vector<double> core_data;
  for (const auto& core : density_tt.cores) {
    core_shapes.insert(
      core_shapes.end(), {core.r_left, core.n, core.r_right});
    core_offsets.push_back(core_data.size());
    core_data.insert(core_data.end(), core.data.begin(), core.data.end());
  }

  vector<int32_t> material_indices {5, 7};
  const char* nuclide_names[] {"U235", "Xe135", "I135"};
  vector<int> nuclide_indices {10, C_NONE, 12};
  vector<int> mat_shape {2};
  vector<int> nuc_shape {3};

  int err = openmc_atom_density_tt_set(
    static_cast<int>(material_indices.size()), material_indices.data(),
    static_cast<int>(nuclide_indices.size()), nuclide_names,
    nuclide_indices.data(), static_cast<int>(mat_shape.size()),
    mat_shape.data(), static_cast<int>(nuc_shape.size()), nuc_shape.data(),
    static_cast<int>(density_tt.cores.size()), core_shapes.data(),
    core_offsets.data(), core_data.size(), core_data.data());
  REQUIRE(err == 0);

  REQUIRE(model::atom_density_tt.enabled());
  REQUIRE(model::atom_density_tt.nuclide_names()[1] == "Xe135");
  REQUIRE(model::atom_density_tt.nuclide_indices()[1] == C_NONE);
  REQUIRE(model::atom_density_tt.material_position(7) == 1);

  vector<double> density;
  model::atom_density_tt.reconstruct_material(5, density);

  vector<double> expected {1.0e-24, 2.0e-24, 3.0e-24};
  REQUIRE_THAT(density, Catch::Matchers::Approx(expected));
}
