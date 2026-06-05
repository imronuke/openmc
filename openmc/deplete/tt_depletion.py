"""Tensor-train depletion helper functions."""

import time
import warnings
from itertools import repeat

import h5py
import numpy as np

from openmc import IDWarning, Material
from openmc.tt import TT, _flat_to_tt_index, tt_svd

from .reaction_rates import ReactionRates


__all__ = ["TTAtomDensities", "TTDepletionAtoms", "TTDepletionRates"]


_TT_DEPLETION_ATOM_NUMBERS = "tt_depletion_atom_numbers"
_TT_DEPLETION_RATES = "tt_depletion_reaction_rates"
_NEGATIVE_ATOM_DENSITY_THRESHOLD = -1.0e-21


def _tt_filter_split(tt_shape, score_size):
    product = 1
    for i in range(len(tt_shape) - 1, -1, -1):
        product *= tt_shape[i]
        if product == score_size:
            return i
    raise RuntimeError("Tensor-train shape is incompatible with tally scores.")


class _TTReactionRates:
    """Normalization data for streamed TT reaction-rate slices."""

    _tt_reaction_rates = True

    def __init__(
            self, normalization_factor, zero_source=False,
            reaction_rate_mask=None):
        self.normalization_factor = normalization_factor
        self.zero_source = zero_source
        self.reaction_rate_mask = reaction_rate_mask


class TTAtomDensities:
    """Stores local material atom densities in tensor-train format.

    This class mirrors the read side of :class:`openmc.deplete.AtomNumber`.
    The represented logical tensor is ``rho[material, nuclide]`` in
    [atom/b-cm]. Atom-number accessors multiply by material volume to preserve
    the existing depletion convention at Bateman and results boundaries.

    Parameters
    ----------
    local_mats : list of str
        Material IDs
    nuclides : list of str
        Nuclides to be tracked
    volume : dict
        Volume of each material in [cm^3]
    n_nuc_burn : int
        Number of nuclides to be burned
    density_tt : openmc.TT
        Tensor-train representation of the material density tensor
    mat_shape : tuple of int
        TT dimensions corresponding to the material axis
    nuc_shape : tuple of int
        TT dimensions corresponding to the nuclide axis
    tt_eps : float, optional
        TT-SVD truncation tolerance used when refreshing ``density_tt``.

    """

    def __init__(self, local_mats, nuclides, volume, n_nuc_burn, density_tt,
                 mat_shape, nuc_shape, tt_eps=None):
        self.index_mat = {mat: i for i, mat in enumerate(local_mats)}
        self.index_nuc = {nuc: i for i, nuc in enumerate(nuclides)}

        self.volume = np.ones(len(local_mats))
        for mat, val in volume.items():
            if mat in self.index_mat:
                self.volume[self.index_mat[mat]] = val

        self.n_nuc_burn = n_nuc_burn
        self.density_tt = density_tt
        self.mat_shape = tuple(int(n) for n in mat_shape)
        self.nuc_shape = tuple(int(n) for n in nuc_shape)
        self.tt_eps = tt_eps

        if int(np.prod(self.mat_shape)) != len(local_mats):
            raise ValueError("Material TT shape is incompatible with local_mats.")
        if int(np.prod(self.nuc_shape)) != len(nuclides):
            raise ValueError("Nuclide TT shape is incompatible with nuclides.")
        if self.density_tt is None and len(local_mats) != 0:
            raise ValueError("Density TT is required for local materials.")
        if (self.density_tt is not None and
                self.density_tt.shape != self.mat_shape + self.nuc_shape):
            raise ValueError("Density TT shape is incompatible with metadata.")

    @staticmethod
    def _density_to_atoms(density, volume):
        """Convert density in [atom/b-cm] to total atoms."""
        return np.asarray(density, dtype=float) * volume * 1.0e24

    @staticmethod
    def _atoms_to_density(atoms, volume):
        """Convert total atoms to density in [atom/b-cm]."""
        return np.asarray(atoms, dtype=float) / volume * 1.0e-24

    def _density_to_tt(self, density):
        density = np.asarray(density, dtype=float)
        expected_shape = (len(self.index_mat), len(self.index_nuc))
        if density.shape != expected_shape:
            raise ValueError(
                "Dense atom-density array shape is incompatible with metadata.")
        if len(self.index_mat) == 0:
            return None

        logical_shape = self.mat_shape + self.nuc_shape
        return tt_svd(density.reshape(logical_shape, order='C'),
                      eps=self.tt_eps)

    def compress_from_density(self, density):
        """Replace the TT representation from dense atom densities."""
        self.density_tt = self._density_to_tt(density)

    def dense_to_tt_density(self, density):
        """Return a new TTAtomDensities object from dense atom densities."""
        density_tt = self._density_to_tt(density)
        volume = {
            mat: self.volume[i]
            for mat, i in self.index_mat.items()
        }
        return type(self)(
            list(self.index_mat), list(self.index_nuc), volume,
            self.n_nuc_burn, density_tt, self.mat_shape, self.nuc_shape,
            self.tt_eps)

    def _get_mat_index(self, mat):
        if isinstance(mat, Material):
            mat = str(mat.id)
        return self.index_mat[mat] if isinstance(mat, str) else mat

    def _get_nuc_index(self, nuc):
        return self.index_nuc[nuc] if isinstance(nuc, str) else nuc

    def _mat_indices_from_slice(self, mat):
        return list(range(len(self.index_mat)))[mat]

    def _density_vector(self, mat_index):
        if self.density_tt is None:
            raise IndexError("Material index is out of bounds.")
        prefix = _flat_to_tt_index(mat_index, self.mat_shape)
        density = self.density_tt._contract_slice(prefix).reshape(-1, order='C')
        _clip_significant_negative_densities(density)
        return density

    def __getitem__(self, pos):
        """Retrieve atom density in [atom/b-cm] from TT density storage."""
        mat, nuc = pos
        mat = self._get_mat_index(mat)
        nuc = self._get_nuc_index(nuc)

        if isinstance(mat, slice):
            return self.get_mat_full_density_slice(mat)[:, nuc]
        return self._density_vector(mat)[nuc]

    @property
    def materials(self):
        return self.index_mat.keys()

    @property
    def nuclides(self):
        return self.index_nuc.keys()

    @property
    def n_nuc(self):
        return len(self.index_nuc)

    @property
    def burnable_nuclides(self):
        return [nuc for nuc, ind in self.index_nuc.items()
                if ind < self.n_nuc_burn]

    def get_mat_volume(self, mat):
        """Return material volume in [cm^3]."""
        mat = self._get_mat_index(mat)
        return self.volume[mat]

    def get_mat_full_density_slice(self, mat):
        """Return densities in [atom/b-cm] for all tracked nuclides."""
        mat = self._get_mat_index(mat)
        if isinstance(mat, slice):
            indices = self._mat_indices_from_slice(mat)
            if not indices:
                return np.empty((0, self.n_nuc))
            return np.vstack([self._density_vector(i) for i in indices])
        return self._density_vector(mat)

    def get_mat_density_slice(self, mat):
        """Return densities in [atom/b-cm] for all burned nuclides."""
        return self.get_mat_full_density_slice(mat)[..., :self.n_nuc_burn]

    def get_mat_full_atom_slice(self, mat):
        """Return total atoms for all tracked nuclides in one material."""
        mat = self._get_mat_index(mat)
        density = self.get_mat_full_density_slice(mat)
        volumes = self.volume[mat]
        if isinstance(mat, slice) and np.ndim(density) == 2:
            return self._density_to_atoms(density, volumes[:, None])
        return self._density_to_atoms(density, volumes)

    def get_mat_atom_slice(self, mat):
        """Return total atoms for all burned nuclides in one material."""
        return self.get_mat_full_atom_slice(mat)[..., :self.n_nuc_burn]

    def get_atom_density(self, mat, nuc):
        """Return atom density of given material and nuclide in [atom/cm^3]."""
        return self[mat, nuc] * 1.0e24

    def get_atom_densities(self, mat, units='atom/b-cm'):
        """Return atom densities for a given material."""
        mat = self._get_mat_index(mat)
        normalization = 1.0 if units == 'atom/b-cm' else 1.0e24
        full_slice = self.get_mat_full_density_slice(mat)
        return {
            name: normalization * full_slice[..., nuc]
            for name, nuc in self.index_nuc.items()
        }


def _copy_tt_mean(tt, n_realizations):
    """Return a TT copy representing the tally mean."""
    cores = [core.copy() for core in tt.cores]
    if n_realizations > 0 and cores:
        cores[0] /= n_realizations
    return TT(cores)


def _clip_significant_negative_atoms(atom_number, volume):
    """Clip atom numbers below the material-density warning threshold."""
    threshold = _NEGATIVE_ATOM_DENSITY_THRESHOLD * volume * 1.0e24
    if atom_number.size and np.min(atom_number) < threshold:
        atom_number[atom_number < threshold] = 0.0


def _clip_significant_negative_densities(density):
    """Clip atom densities below the material-density warning threshold."""
    if density.size and np.min(density) < _NEGATIVE_ATOM_DENSITY_THRESHOLD:
        density[density < _NEGATIVE_ATOM_DENSITY_THRESHOLD] = 0.0


def _write_tt_hdf5(group, name, tt):
    """Write a TT object to an HDF5 group."""
    tt_group = group.create_group(name)
    tt_group.create_dataset("n_cores", data=len(tt.cores))
    for i, core in enumerate(tt.cores):
        tt_group.create_dataset(f"core_{i}_shape", data=np.asarray(core.shape))
        tt_group.create_dataset(f"core_{i}", data=core.ravel())


def _get_tt_atom_number_write_data(tt_eps, atom_number, volume=None):
    """Return TT atom-number data for HDF5 output."""
    atom_number = np.asarray(atom_number, dtype=float)
    if volume is not None:
        volume = np.asarray(volume, dtype=float)
        if atom_number.shape[0] != volume.size:
            raise ValueError(
                "Atom-number volumes are incompatible with atom-number data.")
        clipped = False
        for i, mat_volume in enumerate(volume):
            threshold = (
                _NEGATIVE_ATOM_DENSITY_THRESHOLD * mat_volume * 1.0e24)
            atom_slice = atom_number[i]
            if atom_slice.size and np.min(atom_slice) < threshold:
                if not clipped:
                    atom_number = atom_number.copy()
                    clipped = True
                    atom_slice = atom_number[i]
                atom_slice[atom_slice < threshold] = 0.0
    return tt_svd(atom_number, eps=tt_eps)


def _write_tt_atom_numbers(handle, step, block_index, atom_tt):
    """Write per-step TT atom-number data to HDF5."""
    group = handle.require_group(_TT_DEPLETION_ATOM_NUMBERS)
    steps = group.require_group('steps')
    step_group = steps.require_group(str(step))
    blocks = step_group.require_group('blocks')
    block_name = str(block_index)
    if block_name in blocks:
        del blocks[block_name]
    block_group = blocks.create_group(block_name)
    _write_tt_hdf5(block_group, 'atom_number', atom_tt)


def _write_tt_atom_number_result(result, handle, index, block_index):
    """Write atom-number data in tensor-train format."""
    volume = np.empty(result.n_mat)
    for mat, mat_index in result.index_mat.items():
        volume[mat_index] = result.volume[mat]
    atom_tt = _get_tt_atom_number_write_data(
        result.tt_eps, result.data, volume)
    _write_tt_atom_numbers(handle, index, block_index, atom_tt)


def _set_tt_step_result_atom_numbers(result, operator, burn_list):
    """Populate StepResult atom numbers from TT-backed operator storage."""
    for mat_i, mat in enumerate(burn_list):
        result[mat_i, :] = operator.number.get_mat_full_atom_slice(mat)


def _load_step_result_metadata(result_cls, handle, step, has_stages=False):
    """Load depletion step metadata without dense atom numbers."""
    result = result_cls()

    eigenvalues_dset = handle["/eigenvalues"]
    time_dset = handle["/time"]
    if "source_rate" in handle:
        source_rate_dset = handle["/source_rate"]
    else:
        # Older versions used "power" instead of "source_rate"
        source_rate_dset = handle["/power"]

    if has_stages:
        result.k = eigenvalues_dset[step, 0, :]
        # source_rate had shape (n_steps, n_stages) in old format
        result.source_rate = source_rate_dset[step, 0]
    else:
        result.k = eigenvalues_dset[step, :]
        result.source_rate = source_rate_dset[step]

    result.time = time_dset[step, :]

    if "depletion time" in handle:
        proc_time_dset = handle["/depletion time"]
        if step < proc_time_dset.shape[0]:
            result.proc_time = proc_time_dset[step]

    if "keff_search_root" in handle:
        keff_search_root_dset = handle["/keff_search_root"]
        result.keff_search_root = keff_search_root_dset[step]

    if result.proc_time is None:
        result.proc_time = np.array([np.nan])

    result.volume = {}
    result.index_mat = {}
    result.index_nuc = {}
    result.mat_to_name = {}
    rxn_nuc_to_ind = {}
    rxn_to_ind = {}

    for mat, mat_group in handle["/materials"].items():
        result.volume[mat] = mat_group.attrs["volume"]
        result.index_mat[mat] = mat_group.attrs["index"]
        if "name" in mat_group.attrs:
            result.mat_to_name[mat] = mat_group.attrs["name"]

    for nuc, nuc_group in handle["/nuclides"].items():
        result.index_nuc[nuc] = nuc_group.attrs["atom number index"]

        if ("reaction rates" in handle and
                "reaction rate index" in nuc_group.attrs):
            rxn_nuc_to_ind[nuc] = nuc_group.attrs["reaction rate index"]

    if "reaction rates" in handle and "reactions" in handle:
        for rxn, rxn_group in handle["/reactions"].items():
            rxn_to_ind[rxn] = rxn_group.attrs["index"]

    rate = ReactionRates(result.index_mat, rxn_nuc_to_ind, rxn_to_ind, True)
    if "reaction rates" in handle:
        if has_stages:
            # Old format: (n_steps, n_stages, n_mats, n_nucs, n_rxns)
            rate[:] = handle["/reaction rates"][step, 0, :, :, :]
        else:
            # New format: (n_steps, n_mats, n_nucs, n_rxns)
            rate[:] = handle["/reaction rates"][step, :, :, :]
    result.rates = rate

    return result


def _load_results_data(handle, result_cls):
    """Load dense results or metadata-only TT atom-number results."""
    data = []
    if "number" in handle:
        for i in range(handle["number"].shape[0]):
            data.append(result_cls.from_hdf5(handle, i))
        return data

    if _TT_DEPLETION_ATOM_NUMBERS in handle:
        for i in range(handle["time"].shape[0]):
            data.append(_load_step_result_metadata(result_cls, handle, i))
        return data

    raise RuntimeError(
        "Depletion results file does not contain atom-number data.")


def _raise_dense_atom_number_error():
    """Raise when a dense atom-number API is used with TT atom numbers."""
    raise RuntimeError(
        "This depletion results file stores atom numbers in "
        "tensor-train format. Use openmc.deplete.TTDepletionAtoms "
        "to read atom numbers from this file.")


def _require_dense_atom_numbers(result):
    """Raise if a dense atom-number API is used with TT atom numbers."""
    if result.data is None:
        _raise_dense_atom_number_error()


class TTDepletionAtoms:
    """Reader for tensor-train depletion atom numbers.

    Parameters
    ----------
    filename : str, optional
        Path to depletion results file with tensor-train atom numbers.

    """

    def __init__(self, filename='depletion_results.h5'):
        self.filename = filename
        with h5py.File(filename, 'r') as handle:
            if _TT_DEPLETION_ATOM_NUMBERS not in handle:
                raise RuntimeError(
                    "Depletion results file does not contain tensor-train "
                    "atom numbers.")

            self.index_mat = {}
            self.volume = {}
            self.mat_to_name = {}
            self.index_nuc = {}

            for mat, mat_group in handle['materials'].items():
                self.index_mat[mat] = int(mat_group.attrs['index'])
                self.volume[mat] = mat_group.attrs['volume']
                if "name" in mat_group.attrs:
                    self.mat_to_name[mat] = mat_group.attrs["name"]

            for nuc, nuc_group in handle['nuclides'].items():
                self.index_nuc[nuc] = int(nuc_group.attrs[
                    "atom number index"])

            self.nuclides = [None] * len(self.index_nuc)
            for nuc, i_nuc in self.index_nuc.items():
                self.nuclides[i_nuc] = nuc

            self.time = handle['time'][()]
            self.n_steps = self.time.shape[0]

    def _normalize_step(self, step):
        if step < 0:
            step += self.n_steps
        if step < 0 or step >= self.n_steps:
            raise IndexError("Depletion step index is out of bounds.")
        return step

    def _normalize_mat(self, mat):
        if isinstance(mat, Material):
            mat = str(mat.id)
        else:
            mat = str(mat)
        if mat not in self.index_mat:
            raise KeyError(f"Material {mat} is not present in the results.")
        return mat

    def _get_material_atom_vector(self, handle, step, mat):
        mat = self._normalize_mat(mat)
        mat_index = self.index_mat[mat]
        blocks = handle[
            f'{_TT_DEPLETION_ATOM_NUMBERS}/steps/{step}/blocks']
        block_starts = sorted(int(block) for block in blocks)
        block_pos = np.searchsorted(block_starts, mat_index, side='right') - 1
        if block_pos < 0:
            raise IndexError("Material index is out of bounds.")

        block_start = block_starts[block_pos]
        atom_tt = TT.from_hdf5(
            blocks[str(block_start)]['atom_number'])
        if len(atom_tt.shape) != 2 or atom_tt.shape[1] != len(self.index_nuc):
            raise RuntimeError(
                "Tensor-train atom-number block shape is incompatible with "
                "depletion metadata.")

        local_mat_index = mat_index - block_start
        if local_mat_index < 0 or local_mat_index >= atom_tt.shape[0]:
            raise IndexError("Material index is out of bounds.")

        atom_number = atom_tt._contract_slice(
            (local_mat_index,)).reshape(-1, order='C')
        _clip_significant_negative_atoms(atom_number, self.volume[mat])
        return atom_number

    def get_material_atom_vector(self, step, mat):
        """Return total atom numbers for one material at one depletion step."""
        step = self._normalize_step(step)
        with h5py.File(self.filename, 'r') as handle:
            return self._get_material_atom_vector(handle, step, mat)

    def get_material_atoms(self, step, mat):
        """Return total atom numbers by nuclide for one material."""
        atom_vector = self.get_material_atom_vector(step, mat)
        return {
            nuc: atom_vector[i]
            for i, nuc in enumerate(self.nuclides)
        }

    def get_atoms(
            self, mat, nuc, nuc_units="atoms", time_units="s"):
        """Get number of nuclides over time from a single material."""
        if time_units not in {"s", "d", "min", "h", "a"}:
            raise ValueError(f"Invalid time_units '{time_units}'.")
        if nuc_units not in {"atoms", "atom/b-cm", "atom/cm3"}:
            raise ValueError(f"Invalid nuc_units '{nuc_units}'.")

        mat = self._normalize_mat(mat)
        if nuc not in self.index_nuc:
            raise KeyError(f"Nuclide {nuc} is not present in the results.")
        i_nuc = self.index_nuc[nuc]

        times = self.time[:, 0].copy()
        concentrations = np.empty(self.n_steps)

        with h5py.File(self.filename, 'r') as handle:
            for step in range(self.n_steps):
                atoms = self._get_material_atom_vector(handle, step, mat)
                concentrations[step] = atoms[i_nuc]

        if time_units == "a":
            times /= 365.25*24*60*60
        elif time_units == "d":
            times /= 24*60*60
        elif time_units == "h":
            times /= 60*60
        elif time_units == "min":
            times /= 60

        if nuc_units != "atoms":
            concentrations /= self.volume[mat]
            if nuc_units == "atom/b-cm":
                concentrations *= 1.0e-24

        return times, concentrations

    def get_material(self, step, mat):
        """Return material object for one depleted composition."""
        step = self._normalize_step(step)
        mat_id = self._normalize_mat(mat)
        atoms = self.get_material_atom_vector(step, mat_id)
        volume = self.volume[mat_id]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IDWarning)
            material = Material(material_id=int(mat_id))

        if mat_id in self.mat_to_name:
            material.name = self.mat_to_name[mat_id]
        for nuc, atom in zip(self.nuclides, atoms):
            if atom <= 0.0:
                continue
            atom_per_bcm = atom / volume * 1.0e-24
            material.add_nuclide(nuc, atom_per_bcm)
        material.volume = volume
        return material


def _get_tt_depletion_rate_write_data(operator, rates):
    """Return TT depletion rate data for HDF5 output."""
    rate_helper = vars(operator).get('_rate_helper')
    tt_eps = None if rate_helper is None else vars(rate_helper).get('_tt_eps')
    reaction_rate_mask = getattr(rates, 'reaction_rate_mask', None)
    if reaction_rate_mask is None:
        reaction_rate_mask = vars(operator).get('_tt_reaction_rate_mask')
    if reaction_rate_mask is None:
        reaction_rate_mask = _tt_reaction_rate_mask(
            operator.chain, operator.reaction_rates)

    if getattr(rates, 'zero_source', False):
        return {
            'zero_source': True,
            'tt_eps': tt_eps,
            'reaction_rate_mask': reaction_rate_mask,
            'normalization_factor': 0.0,
            'n_realizations': 0,
            'tt_shape': tuple(),
            'tt_mean': None,
        }

    tally = operator._rate_helper._rate_tally
    n_realizations = tally.num_realizations
    tt_sum = tally.tt_sum
    return {
        'zero_source': False,
        'tt_eps': tt_eps,
        'reaction_rate_mask': reaction_rate_mask,
        'normalization_factor': rates.normalization_factor,
        'n_realizations': n_realizations,
        'tt_shape': tt_sum.shape,
        'tt_mean': _copy_tt_mean(tt_sum, n_realizations),
    }


def _write_tt_depletion_rates(handle, step, data):
    """Write per-step TT depletion reaction-rate data to HDF5."""
    group = handle.require_group(_TT_DEPLETION_RATES)
    group.attrs['format_version'] = 1
    group.attrs['stored_values'] = np.bytes_('tally_mean')
    group.attrs['normalization_mode'] = np.bytes_('fission-q')
    if data['tt_eps'] is not None:
        group.attrs['tt_eps'] = data['tt_eps']
    reaction_rate_mask = np.asarray(data['reaction_rate_mask'], dtype=bool)
    if 'reaction_rate_mask' in group:
        if not np.array_equal(group['reaction_rate_mask'][()], reaction_rate_mask):
            raise ValueError(
                "Tensor-train depletion reaction-rate mask changed between "
                "depletion steps.")
    else:
        group.create_dataset('reaction_rate_mask', data=reaction_rate_mask)

    steps = group.require_group('steps')
    step_name = str(step)
    if step_name in steps:
        del steps[step_name]
    step_group = steps.create_group(step_name)
    step_group.attrs['zero_source'] = data['zero_source']
    step_group.create_dataset(
        'normalization_factor', data=data['normalization_factor'])
    step_group.create_dataset(
        'n_realizations', data=data['n_realizations'])
    step_group.create_dataset(
        'tt_shape', data=np.asarray(data['tt_shape'], dtype=np.int64))

    if not data['zero_source']:
        _write_tt_hdf5(step_group, 'tt_mean', data['tt_mean'])


class TTDepletionRates:
    """Reader for tensor-train depletion reaction rates.

    Parameters
    ----------
    filename : str, optional
        Path to depletion results file with tensor-train depletion reaction
        rates.

    """

    def __init__(self, filename='depletion_results.h5'):
        self.filename = filename
        with h5py.File(filename, 'r') as handle:
            if _TT_DEPLETION_RATES not in handle:
                raise RuntimeError(
                    "Depletion results file does not contain tensor-train "
                    "depletion reaction rates.")

            tt_group = handle[_TT_DEPLETION_RATES]
            self.tt_eps = tt_group.attrs.get('tt_eps')
            if self.tt_eps is not None:
                self.tt_eps = float(self.tt_eps)

            self.index_mat = {}
            self.volume = {}
            self.index_nuc = {}
            self.index_rx = {}

            for mat, mat_group in handle['materials'].items():
                self.index_mat[mat] = mat_group.attrs['index']
                self.volume[mat] = mat_group.attrs['volume']

            for nuc, nuc_group in handle['nuclides'].items():
                if "reaction rate index" in nuc_group.attrs:
                    self.index_nuc[nuc] = nuc_group.attrs[
                        "reaction rate index"]

            if "reactions" in handle:
                for rx, rx_group in handle['reactions'].items():
                    self.index_rx[rx] = rx_group.attrs['index']

            if not self.index_nuc or not self.index_rx:
                raise RuntimeError(
                    "Tensor-train depletion reaction-rate metadata is "
                    "incomplete.")

            if 'reaction_rate_mask' not in tt_group:
                raise RuntimeError(
                    "Tensor-train depletion reaction-rate mask is missing.")
            self.reaction_rate_mask = tt_group['reaction_rate_mask'][()]
            expected_shape = (len(self.index_nuc), len(self.index_rx))
            if self.reaction_rate_mask.shape != expected_shape:
                raise RuntimeError(
                    "Tensor-train depletion reaction-rate mask shape is "
                    "incompatible with reaction-rate metadata.")

            self.n_steps = handle['time'].shape[0]

    def _normalize_step(self, step):
        if step < 0:
            step += self.n_steps
        if step < 0 or step >= self.n_steps:
            raise IndexError("Depletion step index is out of bounds.")
        return step

    def _empty_material_rates(self, mat):
        rates = ReactionRates(
            {mat: 0}, self.index_nuc, self.index_rx, from_results=True)[0]
        rates.fill(0.0)
        return rates

    def get_material_rates(self, step, mat):
        """Return reaction rates for one material at one depletion step.

        Parameters
        ----------
        step : int
            Depletion step index.
        mat : str or int
            Material ID.

        Returns
        -------
        numpy.ndarray
            Reaction rates with shape ``(n_nuclides, n_reactions)``.

        """
        step = self._normalize_step(step)
        mat = str(mat)
        mat_index = self.index_mat[mat]

        with h5py.File(self.filename, 'r') as handle:
            step_group = handle[
                f'{_TT_DEPLETION_RATES}/steps/{step}']
            if step_group.attrs['zero_source']:
                return self._empty_material_rates(mat)

            tt_shape = tuple(int(x) for x in step_group['tt_shape'][()])
            score_size = len(self.index_nuc) * len(self.index_rx)
            split = _tt_filter_split(tt_shape, score_size)
            filter_shape = tt_shape[:split]
            n_filter_bins = int(np.prod(filter_shape))
            if mat_index < 0 or mat_index >= n_filter_bins:
                raise IndexError("Material index is out of bounds.")

            prefix_index = _flat_to_tt_index(mat_index, filter_shape)
            tt_mean = TT.from_hdf5(step_group['tt_mean'])
            data = tt_mean._contract_slice(prefix_index)
            data = data.reshape((len(self.index_nuc), len(self.index_rx)))

            material_rates = self._empty_material_rates(mat)
            material_rates[:] = data
            material_rates *= (
                step_group['normalization_factor'][()] /
                (1e24 * self.volume[mat]))
            return material_rates

    def get_rate(self, step, mat, nuc, rx):
        """Return one reaction rate at one depletion step."""
        rates = self.get_material_rates(step, mat)
        return rates[self.index_nuc[nuc], self.index_rx[rx]]

    def to_reaction_rates(self, step):
        """Reconstruct all reaction rates for one depletion step."""
        step = self._normalize_step(step)
        rates = ReactionRates(
            self.index_mat, self.index_nuc, self.index_rx, from_results=True)
        for mat, i_mat in self.index_mat.items():
            rates[i_mat] = self.get_material_rates(step, mat)
        return rates


def _tt_rate_indices(operator):
    rates = operator.reaction_rates
    rxn_nuclides = operator._rate_helper.nuclides
    nuc_ind = [rates.index_nuc[nuc] for nuc in rxn_nuclides]
    rx_ind = [rates.index_rx[react] for react in operator.chain.reactions]
    return rxn_nuclides, nuc_ind, rx_ind


def _tt_reaction_rate_mask(chain, rates):
    """Return mask for nuclide/reaction pairs present in the depletion chain."""
    mask = np.zeros((rates.n_nuc, rates.n_react), dtype=bool)
    for nuc in chain.nuclides:
        i_nuc = rates.index_nuc.get(nuc.name)
        if i_nuc is None:
            continue
        for reaction in nuc.reactions:
            i_rx = rates.index_rx.get(reaction.type)
            if i_rx is not None:
                mask[i_nuc, i_rx] = True
    return mask


def _prepare_tt_reaction_rates(operator, source_rate):
    """Prepare fission-Q normalization for TT material rate slices."""
    rates = operator.reaction_rates
    rxn_nuclides, nuc_ind, rx_ind = _tt_rate_indices(operator)
    fission_ind = rates.index_rx.get("fission")
    reaction_rate_mask = _tt_reaction_rate_mask(operator.chain, rates)
    operator._tt_reaction_rate_mask = reaction_rate_mask

    operator._normalization_helper.reset()
    operator._yield_helper.unpack()
    operator._rate_helper.reset_tally_means()

    fission_yields = []
    density = np.zeros(rates.n_nuc)
    fission_rates = np.empty(rates.n_nuc) if fission_ind is not None else None

    for i, mat in enumerate(operator.local_mats):
        mat_index = operator._mat_index_map[mat]
        mat_density = operator.number.get_mat_full_density_slice(mat)
        density.fill(0.0)
        for nuc, i_nuc_results in zip(rxn_nuclides, nuc_ind):
            density[i_nuc_results] = mat_density[operator.number.index_nuc[nuc]]

        tally_rates = operator._rate_helper.get_material_rates(
            mat_index, nuc_ind, rx_ind)
        fission_yields.append(operator._yield_helper.weighted_yields(i))

        if fission_ind is not None:
            np.multiply(
                tally_rates[:, fission_ind], density,
                out=fission_rates)
            np.multiply(
                fission_rates, reaction_rate_mask[:, fission_ind],
                out=fission_rates)
            operator._normalization_helper.update(fission_rates)

    operator.chain.fission_yields = fission_yields
    return operator._normalization_helper.factor(source_rate)


def _get_tt_reaction_rates(operator, mat, normalization_factor):
    """Return normalized TT reaction rates for one material."""
    rates = operator.reaction_rates
    _, nuc_ind, rx_ind = _tt_rate_indices(operator)
    mat = str(mat)
    mat_index = operator._mat_index_map[mat]

    tally_rates = operator._rate_helper.get_material_rates(
        mat_index, nuc_ind, rx_ind)
    material_rates = rates[0].copy()
    volume_b_cm = 1e24 * operator.number.get_mat_volume(mat)
    np.multiply(
        tally_rates, normalization_factor / volume_b_cm,
        out=material_rates)
    reaction_rate_mask = vars(operator).get('_tt_reaction_rate_mask')
    if reaction_rate_mask is None:
        reaction_rate_mask = _tt_reaction_rate_mask(operator.chain, rates)
        operator._tt_reaction_rate_mask = reaction_rate_mask
    np.multiply(material_rates, reaction_rate_mask, out=material_rates)
    return material_rates


def timed_tt_deplete(
        solver, chain, operator, n, rates, dt, current_timestep=None,
        substeps=1, transfer_rates=None, external_source_rates=None):
    """Deplete materials with streamed tensor-train reaction-rate slices."""
    if transfer_rates is not None:
        raise ValueError(
            "Transfer rates are not supported when tensor-train "
            "depletion is enabled.")
    if (external_source_rates is not None and
            current_timestep in external_source_rates.external_timesteps):
        raise ValueError(
            "External source rates are not supported when tensor-train "
            "depletion is enabled.")

    start = time.time()
    fission_yields = chain.fission_yields
    if len(fission_yields) == 1:
        fission_yields = repeat(fission_yields[0])
    elif len(fission_yields) != len(operator.local_mats):
        raise ValueError(
            "Number of material fission yield distributions {} is not "
            "equal to the number of compositions {}".format(
                len(fission_yields), len(operator.local_mats)))

    zero_rates = None
    reaction_rate_mask = getattr(rates, 'reaction_rate_mask', None)
    if reaction_rate_mask is not None:
        operator._tt_reaction_rate_mask = reaction_rate_mask
    if getattr(rates, 'zero_source', False):
        zero_rates = operator.reaction_rates[0].copy()
        zero_rates.fill(0.0)

    number = operator.number
    density_end = number.get_mat_full_density_slice(np.s_[:])

    for mat, yields in zip(operator.local_mats, fission_yields):
        n_mat = number.get_mat_atom_slice(mat)
        if zero_rates is None:
            mat_rates = _get_tt_reaction_rates(
                operator, mat, rates.normalization_factor)
        else:
            mat_rates = zero_rates
        matrix = chain.form_matrix(mat_rates, yields)
        n_result = solver(matrix, n_mat, dt, substeps)
        n_result.clip(min=0.0, out=n_result)
        mat_index = number._get_mat_index(mat)
        density_end[mat_index, :number.n_nuc_burn] = number._atoms_to_density(
            n_result, number.get_mat_volume(mat))

    return time.time() - start, number.dense_to_tt_density(density_end)
