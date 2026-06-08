#include "openmc/tt_density.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/settings.h"

namespace openmc {

namespace {

constexpr double NEGATIVE_DENSITY_THRESHOLD {-1.0e-21};

// Return the product of a TT shape while checking for invalid dimensions and
// overflow beyond OpenMC's current integer scale.
int32_t checked_product(const vector<int>& shape, const char* name)
{
  int64_t product {1};
  for (auto n : shape) {
    if (n <= 0) {
      throw std::invalid_argument(
        std::string {name} + " dimensions must be positive.");
    }
    if (product > std::numeric_limits<int32_t>::max() / n) {
      throw std::overflow_error(
        std::string {name} + " dimensions are too large.");
    }
    product *= n;
  }
  return static_cast<int32_t>(product);
}

// Convert a flat row-major material/nuclide index into its factored TT index.
vector<int> flat_to_tt_index(int flat_index, const vector<int>& shape)
{
  vector<int> index(shape.size());
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    index[i] = flat_index % shape[i];
    flat_index /= shape[i];
  }
  return index;
}

// Validate one TT core shape and storage size.
void validate_core(const Core& core, int core_index)
{
  if (core.r_left <= 0 || core.n <= 0 || core.r_right <= 0) {
    throw std::invalid_argument(
      "Tensor-train core dimensions must be positive.");
  }

  const auto expected_size =
    static_cast<size_t>(core.r_left) * core.n * core.r_right;
  if (core.data.size() != expected_size) {
    throw std::invalid_argument(
      "Tensor-train core data size is incompatible with its shape.");
  }

  if (core_index == 0 && core.r_left != 1) {
    throw std::invalid_argument(
      "The first tensor-train core must have left rank 1.");
  }
}

// Validate that TT cores form one consistent tensor train.
void validate_tt(const TT& tt)
{
  if (tt.cores.empty()) {
    throw std::invalid_argument(
      "Tensor-train atom density storage requires at least one core.");
  }

  const int n_cores = static_cast<int>(tt.cores.size());
  for (int i = 0; i < n_cores; ++i) {
    validate_core(tt.cores[i], i);
    if (i > 0 && tt.cores[i].r_left != tt.cores[i - 1].r_right) {
      throw std::invalid_argument(
        "Tensor-train core ranks are not chain-consistent.");
    }
  }

  if (tt.cores.back().r_right != 1) {
    throw std::invalid_argument(
      "The last tensor-train core must have right rank 1.");
  }
}

// Build a lookup from a material or nuclide index to its flat TT-axis
// position.
template<typename T>
std::unordered_map<T, int> build_position_map(
  const vector<T>& values, const char* name)
{
  std::unordered_map<T, int> map;
  map.reserve(values.size());
  const int n_values = static_cast<int>(values.size());
  for (int i = 0; i < n_values; ++i) {
    auto inserted = map.emplace(values[i], i);
    if (!inserted.second) {
      throw std::invalid_argument(std::string {"Duplicate "} + name +
                                  " in tensor-train atom density metadata.");
    }
  }
  return map;
}

// Build a lookup for nuclides with loaded transport data. Nuclides that are
// present only in depletion metadata use C_NONE and are skipped.
std::unordered_map<int, int> build_nuclide_position_map(
  const vector<int>& values)
{
  std::unordered_map<int, int> map;
  map.reserve(values.size());
  const int n_values = static_cast<int>(values.size());
  for (int i = 0; i < n_values; ++i) {
    if (values[i] == C_NONE)
      continue;
    auto inserted = map.emplace(values[i], i);
    if (!inserted.second) {
      throw std::invalid_argument(
        "Duplicate nuclide index in tensor-train atom density metadata.");
    }
  }
  return map;
}

// Contract the material-axis cores at one material index, leaving the TT state
// vector at the material/nuclide axis boundary.
void contract_material_prefix(
  const TT& tt, const vector<int>& mat_prefix, vector<double>& state)
{
  state.assign(1, 1.0);
  const int n_mat_cores = static_cast<int>(mat_prefix.size());
  for (int i_core = 0; i_core < n_mat_cores; ++i_core) {
    const auto& core {tt.cores[i_core]};
    const int mode {mat_prefix[i_core]};
    vector<double> next(core.r_right, 0.0);
    for (int i = 0; i < core.r_left; ++i) {
      for (int k = 0; k < core.r_right; ++k) {
        next[k] += state[i] * core(i, mode, k);
      }
    }
    state.swap(next);
  }
}

// Expand the remaining nuclide-axis cores into a dense material density row.
void expand_nuclide_suffix(
  const TT& tt, int first_core, vector<double>& state, vector<double>& density)
{
  int n_rows {1};
  int rank {static_cast<int>(state.size())};

  const int n_cores = static_cast<int>(tt.cores.size());
  for (int i_core = first_core; i_core < n_cores; ++i_core) {
    const auto& core {tt.cores[i_core]};
    vector<double> next(
      static_cast<size_t>(n_rows) * core.n * core.r_right, 0.0);

    for (int row = 0; row < n_rows; ++row) {
      for (int mode = 0; mode < core.n; ++mode) {
        for (int k = 0; k < core.r_right; ++k) {
          double value {0.0};
          for (int i = 0; i < rank; ++i) {
            value += state[row * rank + i] * core(i, mode, k);
          }
          next[(row * core.n + mode) * core.r_right + k] = value;
        }
      }
    }

    n_rows *= core.n;
    rank = core.r_right;
    state.swap(next);
  }

  if (rank != 1) {
    throw std::runtime_error(
      "Tensor-train material row reconstruction ended with non-unit rank.");
  }

  density.resize(n_rows);
  for (int i = 0; i < n_rows; ++i) {
    density[i] = state[i];
  }
}

} // namespace

namespace model {

AtomDensityTT atom_density_tt;

} // namespace model

// Reset all TT atom-density state so subsequent OpenMC initializations start
// from ordinary material density storage.
void AtomDensityTT::clear()
{
  enabled_ = false;
  material_indices_.clear();
  nuclide_names_.clear();
  nuclide_indices_.clear();
  mat_shape_.clear();
  nuc_shape_.clear();
  density_tt_ = TT {};
  material_pos_.clear();
  nuclide_pos_.clear();
}

// Store a complete TT density tensor and its material/nuclide metadata after
// validating shape compatibility and unsupported execution modes.
void AtomDensityTT::set_data(const vector<int32_t>& material_indices,
  const vector<std::string>& nuclide_names,
  const vector<int>& nuclide_indices, const vector<int>& mat_shape,
  const vector<int>& nuc_shape, TT density_tt)
{
  validate_supported_mode();
  validate_tt(density_tt);

  if (mat_shape.empty() || nuc_shape.empty()) {
    throw std::invalid_argument(
      "Tensor-train atom density material and nuclide shapes are required.");
  }

  const auto n_materials = checked_product(mat_shape, "Material TT shape");
  if (n_materials != static_cast<int64_t>(material_indices.size())) {
    throw std::invalid_argument(
      "Material TT shape is incompatible with material metadata.");
  }

  const auto n_nuclides = checked_product(nuc_shape, "Nuclide TT shape");
  if (n_nuclides != static_cast<int64_t>(nuclide_indices.size())) {
    throw std::invalid_argument(
      "Nuclide TT shape is incompatible with nuclide metadata.");
  }
  if (nuclide_names.size() != nuclide_indices.size()) {
    throw std::invalid_argument(
      "Nuclide names and indices have incompatible sizes.");
  }
  for (auto nuc_index : nuclide_indices) {
    if (nuc_index < C_NONE) {
      throw std::invalid_argument(
        "Nuclide indices must be non-negative or C_NONE.");
    }
  }

  vector<int> expected_shape;
  expected_shape.reserve(mat_shape.size() + nuc_shape.size());
  expected_shape.insert(
    expected_shape.end(), mat_shape.begin(), mat_shape.end());
  expected_shape.insert(
    expected_shape.end(), nuc_shape.begin(), nuc_shape.end());
  if (density_tt.shape() != expected_shape) {
    throw std::invalid_argument(
      "Tensor-train atom density shape is incompatible with metadata.");
  }

  auto material_pos =
    build_position_map<int32_t>(material_indices, "material index");
  auto nuclide_pos = build_nuclide_position_map(nuclide_indices);

  material_indices_ = material_indices;
  nuclide_names_ = nuclide_names;
  nuclide_indices_ = nuclide_indices;
  mat_shape_ = mat_shape;
  nuc_shape_ = nuc_shape;
  density_tt_ = std::move(density_tt);
  material_pos_ = std::move(material_pos);
  nuclide_pos_ = std::move(nuclide_pos);
  enabled_ = true;
}

// Reject TT depletion in modes that do not yet have a correct TT-density path.
void AtomDensityTT::validate_supported_mode() const
{
  if (settings::event_based) {
    throw std::runtime_error(
      "Tensor-train depletion is not supported with event-based mode.");
  }

  if (!settings::run_CE) {
    throw std::runtime_error(
      "Tensor-train depletion is only supported in continuous-energy mode.");
  }

  if (mpi::n_procs != 1) {
    throw std::runtime_error(
      "Tensor-train depletion is not supported with multiple MPI ranks.");
  }
}

// Return the flat TT material-axis position for a C++ material index.
int AtomDensityTT::material_position(int32_t material_index) const
{
  auto it = material_pos_.find(material_index);
  if (it == material_pos_.end()) {
    throw std::out_of_range(
      "Material index is not present in tensor-train atom density metadata.");
  }
  return it->second;
}

// Return whether a C++ material index is backed by the TT density storage.
bool AtomDensityTT::contains_material(int32_t material_index) const
{
  return material_pos_.find(material_index) != material_pos_.end();
}

// Return the flat TT nuclide-axis position for a C++ nuclide index.
int AtomDensityTT::nuclide_position(int nuclide_index) const
{
  auto it = nuclide_pos_.find(nuclide_index);
  if (it == nuclide_pos_.end()) {
    throw std::out_of_range(
      "Nuclide index is not present in tensor-train atom density metadata.");
  }
  return it->second;
}

// Reconstruct the dense atom-density row for one material without expanding
// densities for other materials.
void AtomDensityTT::reconstruct_material(
  int32_t material_index, vector<double>& density) const
{
  if (!enabled_) {
    throw std::runtime_error(
      "Tensor-train atom density storage has not been enabled.");
  }

  const auto mat_pos = material_position(material_index);
  const auto mat_prefix = flat_to_tt_index(mat_pos, mat_shape_);

  vector<double> state;
  contract_material_prefix(density_tt_, mat_prefix, state);
  expand_nuclide_suffix(
    density_tt_, static_cast<int>(mat_shape_.size()), state, density);

  if (!density.empty() &&
      *std::min_element(density.begin(), density.end()) <
        NEGATIVE_DENSITY_THRESHOLD) {
    for (auto& value : density) {
      if (value < NEGATIVE_DENSITY_THRESHOLD)
        value = 0.0;
    }
  }
}

namespace {

// Copy a C API integer array into OpenMC storage.
template<typename T>
vector<T> copy_array(const T* data, int n, const char* name)
{
  if (n <= 0) {
    throw std::invalid_argument(
      std::string {name} + " count must be positive.");
  }
  if (data == nullptr) {
    throw std::invalid_argument(std::string {name} + " pointer is null.");
  }
  return {data, data + n};
}

// Copy C strings into C++ storage so metadata remains valid after the C API
// call returns.
vector<std::string> copy_nuclide_names(const char** names, int n)
{
  if (n <= 0) {
    throw std::invalid_argument("Nuclide count must be positive.");
  }
  if (names == nullptr) {
    throw std::invalid_argument("Nuclide-name pointer is null.");
  }

  vector<std::string> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (names[i] == nullptr) {
      throw std::invalid_argument("Nuclide-name entry is null.");
    }
    result.emplace_back(names[i]);
  }
  return result;
}

// Return the number of doubles in one core with size_t overflow protection.
size_t checked_core_size(int r_left, int n, int r_right)
{
  auto max_size = std::numeric_limits<size_t>::max();
  size_t size = static_cast<size_t>(r_left);
  if (size > max_size / static_cast<size_t>(n)) {
    throw std::overflow_error("Tensor-train core size is too large.");
  }
  size *= static_cast<size_t>(n);
  if (size > max_size / static_cast<size_t>(r_right)) {
    throw std::overflow_error("Tensor-train core size is too large.");
  }
  return size * static_cast<size_t>(r_right);
}

// Copy flattened core data from the C API into owning TT cores.
std::vector<Core> copy_cores(int n_cores, const int* core_shapes,
  const size_t* core_offsets, size_t core_data_size, const double* core_data)
{
  if (n_cores <= 0) {
    throw std::invalid_argument("Tensor-train core count must be positive.");
  }
  if (core_shapes == nullptr || core_offsets == nullptr ||
      core_data == nullptr) {
    throw std::invalid_argument("Tensor-train core pointer is null.");
  }

  std::vector<Core> cores;
  cores.reserve(n_cores);
  for (int i = 0; i < n_cores; ++i) {
    const int r_left = core_shapes[3 * i];
    const int n = core_shapes[3 * i + 1];
    const int r_right = core_shapes[3 * i + 2];
    if (r_left <= 0 || n <= 0 || r_right <= 0) {
      throw std::invalid_argument(
        "Tensor-train core dimensions must be positive.");
    }

    const size_t core_size = checked_core_size(r_left, n, r_right);
    const size_t offset = core_offsets[i];
    if (offset > core_data_size || core_size > core_data_size - offset) {
      throw std::invalid_argument(
        "Tensor-train core offset is incompatible with data storage.");
    }

    std::vector<double> data(
      core_data + offset, core_data + offset + core_size);
    cores.emplace_back(r_left, n, r_right, std::move(data));
  }
  return cores;
}

} // namespace

extern "C" int openmc_atom_density_tt_clear()
{
  model::atom_density_tt.clear();
  return 0;
}

extern "C" int openmc_atom_density_tt_set(int n_materials,
  const int32_t* material_indices, int n_nuclides,
  const char** nuclide_names, const int* nuclide_indices, int mat_ndim,
  const int* mat_shape, int nuc_ndim, const int* nuc_shape, int n_cores,
  const int* core_shapes, const size_t* core_offsets, size_t core_data_size,
  const double* core_data)
{
  try {
    auto materials =
      copy_array<int32_t>(material_indices, n_materials, "Material");
    auto names = copy_nuclide_names(nuclide_names, n_nuclides);
    auto nuclides = copy_array<int>(nuclide_indices, n_nuclides, "Nuclide");
    auto mat_dims = copy_array<int>(mat_shape, mat_ndim, "Material shape");
    auto nuc_dims = copy_array<int>(nuc_shape, nuc_ndim, "Nuclide shape");
    auto cores = copy_cores(
      n_cores, core_shapes, core_offsets, core_data_size, core_data);

    model::atom_density_tt.set_data(materials, names, nuclides, mat_dims,
      nuc_dims, TT {std::move(cores)});
  } catch (const std::bad_alloc& e) {
    set_errmsg(e.what());
    return OPENMC_E_ALLOCATE;
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_INVALID_ARGUMENT;
  }

  return 0;
}

} // namespace openmc
