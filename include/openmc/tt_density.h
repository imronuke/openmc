#ifndef OPENMC_TT_DENSITY_H
#define OPENMC_TT_DENSITY_H

#include <cstdint>
#include <string>
#include <unordered_map>

#include "openmc/tt.h"
#include "openmc/vector.h"

namespace openmc {

//! Tensor-train storage for material atom densities.
class AtomDensityTT {
public:
  //! Clear tensor-train atom density storage.
  void clear();

  //! Whether tensor-train atom density storage is active.
  bool enabled() const { return enabled_; }

  //! Set complete tensor-train atom density storage.
  void set_data(const vector<int32_t>& material_indices,
    const vector<std::string>& nuclide_names,
    const vector<int>& nuclide_indices, const vector<int>& mat_shape,
    const vector<int>& nuc_shape, TT density_tt);

  //! Validate that the current OpenMC execution mode supports TT densities.
  void validate_supported_mode() const;

  //! Get flat material position in the tensor-train material axis.
  int material_position(int32_t material_index) const;

  //! Whether a material index is represented by the TT material axis.
  bool contains_material(int32_t material_index) const;

  //! Get flat nuclide position in the tensor-train nuclide axis.
  int nuclide_position(int nuclide_index) const;

  //! Reconstruct atom densities in [atom/b-cm] for one material.
  void reconstruct_material(
    int32_t material_index, vector<double>& density) const;

  //! Material indices represented by the TT material axis.
  const vector<int32_t>& material_indices() const { return material_indices_; }

  //! Nuclide indices represented by the TT nuclide axis.
  const vector<int>& nuclide_indices() const { return nuclide_indices_; }

  //! Nuclide names represented by the TT nuclide axis.
  const vector<std::string>& nuclide_names() const { return nuclide_names_; }

  //! TT dimensions corresponding to the material axis.
  const vector<int>& mat_shape() const { return mat_shape_; }

  //! TT dimensions corresponding to the nuclide axis.
  const vector<int>& nuc_shape() const { return nuc_shape_; }

  //! Tensor-train representation of atom densities.
  const TT& density_tt() const { return density_tt_; }

private:
  bool enabled_ {false};
  vector<int32_t> material_indices_;
  vector<std::string> nuclide_names_;
  vector<int> nuclide_indices_;
  vector<int> mat_shape_;
  vector<int> nuc_shape_;
  TT density_tt_;
  std::unordered_map<int32_t, int> material_pos_;
  std::unordered_map<int, int> nuclide_pos_;
};

namespace model {

extern AtomDensityTT atom_density_tt;

} // namespace model

} // namespace openmc

#endif // OPENMC_TT_DENSITY_H
