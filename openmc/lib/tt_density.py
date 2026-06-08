from ctypes import c_char_p, c_double, c_int, c_int32, c_size_t, POINTER

import numpy as np

from . import _dll
from .error import _error_handler
from .material import materials
from .nuclide import nuclides


__all__ = [
    'clear_atom_density_tt', 'set_atom_density_tt',
    'set_atom_density_tt_data',
]


C_NONE = -1


_dll.openmc_atom_density_tt_clear.argtypes = []
_dll.openmc_atom_density_tt_clear.restype = c_int
_dll.openmc_atom_density_tt_clear.errcheck = _error_handler
_dll.openmc_atom_density_tt_set.argtypes = [
    c_int, POINTER(c_int32), c_int, POINTER(c_char_p), POINTER(c_int),
    c_int, POINTER(c_int), c_int, POINTER(c_int), c_int, POINTER(c_int),
    POINTER(c_size_t), c_size_t, POINTER(c_double)
]
_dll.openmc_atom_density_tt_set.restype = c_int
_dll.openmc_atom_density_tt_set.errcheck = _error_handler


def clear_atom_density_tt():
    """Clear tensor-train atom-density storage on the C side."""
    _dll.openmc_atom_density_tt_clear()


def _ordered_keys(index):
    """Return dictionary keys ordered by their integer metadata position."""
    keys = [None] * len(index)
    for key, i in index.items():
        keys[i] = key
    return keys


def _core_transfer_arrays(density_tt):
    """Return shape, offset, and flattened data arrays for TT transfer."""
    cores = [np.ascontiguousarray(core, dtype=np.float64)
             for core in density_tt.cores]
    if not cores:
        raise ValueError("Tensor-train atom-density storage needs cores.")

    core_shapes = np.ascontiguousarray(
        [core.shape for core in cores], dtype=np.intc).reshape(-1)
    core_offsets = np.empty(len(cores), dtype=np.uintp)
    total_size = sum(core.size for core in cores)
    core_data = np.empty(total_size, dtype=np.float64)

    offset = 0
    for i, core in enumerate(cores):
        core_offsets[i] = offset
        flat = core.reshape(-1, order='C')
        core_data[offset:offset + core.size] = flat
        offset += core.size

    return core_shapes, core_offsets, core_data


def set_atom_density_tt_data(
        material_indices, nuclide_names, nuclide_indices, mat_shape,
        nuc_shape, density_tt):
    """Set C-side tensor-train atom-density storage from explicit metadata."""
    if density_tt is None:
        clear_atom_density_tt()
        return

    material_indices = np.ascontiguousarray(material_indices, dtype=np.int32)
    nuclide_names = [str(name) for name in nuclide_names]
    nuclide_indices = np.ascontiguousarray(nuclide_indices, dtype=np.intc)
    mat_shape = np.ascontiguousarray(mat_shape, dtype=np.intc)
    nuc_shape = np.ascontiguousarray(nuc_shape, dtype=np.intc)
    core_shapes, core_offsets, core_data = _core_transfer_arrays(density_tt)

    encoded_names = [name.encode() for name in nuclide_names]
    name_array = (c_char_p * len(encoded_names))(*encoded_names)

    _dll.openmc_atom_density_tt_set(
        len(material_indices),
        material_indices.ctypes.data_as(POINTER(c_int32)),
        len(nuclide_names),
        name_array,
        nuclide_indices.ctypes.data_as(POINTER(c_int)),
        len(mat_shape),
        mat_shape.ctypes.data_as(POINTER(c_int)),
        len(nuc_shape),
        nuc_shape.ctypes.data_as(POINTER(c_int)),
        len(density_tt.cores),
        core_shapes.ctypes.data_as(POINTER(c_int)),
        core_offsets.ctypes.data_as(POINTER(c_size_t)),
        core_data.size,
        core_data.ctypes.data_as(POINTER(c_double)))


def set_atom_density_tt(number):
    """Set C-side TT atom densities from a TTAtomDensities object."""
    if number.density_tt is None:
        clear_atom_density_tt()
        return

    material_ids = _ordered_keys(number.index_mat)
    material_indices = [
        materials[int(mat)]._index
        for mat in material_ids
    ]

    nuclide_names = _ordered_keys(number.index_nuc)
    nuclide_indices = [
        nuclides[nuc]._index if nuc in nuclides else C_NONE
        for nuc in nuclide_names
    ]

    set_atom_density_tt_data(material_indices, nuclide_names, nuclide_indices,
                             number.mat_shape, number.nuc_shape,
                             number.density_tt)
