#include <cstddef>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/message_passing.h"
#include "openmc/settings.h"
#include "openmc/tt_density.h"

using namespace openmc;

TEST_CASE("AtomDensityTT reconstructs one material row")
{
  settings::event_based = false;
  settings::run_CE = true;
  mpi::n_procs = 1;

  model::atom_density_tt.clear();

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

  model::atom_density_tt.clear();
}

TEST_CASE("AtomDensityTT accepts C API transfer")
{
  settings::event_based = false;
  settings::run_CE = true;
  mpi::n_procs = 1;

  model::atom_density_tt.clear();

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

  model::atom_density_tt.clear();
}
