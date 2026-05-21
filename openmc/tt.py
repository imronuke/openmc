from __future__ import annotations

from collections.abc import Iterable

import h5py
import numpy as np


class TT:
    """Tensor-train data represented by a sequence of cores.

    Parameters
    ----------
    cores : iterable of numpy.ndarray
        Tensor-train cores with shape ``(r_left, n, r_right)``.

    """

    def __init__(self, cores: Iterable[np.ndarray]):
        self.cores = [np.asarray(core) for core in cores]
        for core in self.cores:
            if core.ndim != 3:
                raise ValueError("Each tensor-train core must be three-dimensional.")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(core.shape[1] for core in self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        if not self.cores:
            return tuple()
        return (self.cores[0].shape[0],) + tuple(core.shape[2] for core in self.cores)

    @classmethod
    def from_hdf5(cls, group: h5py.Group):
        """Read a tensor train from an HDF5 group."""
        n_cores = int(group["n_cores"][()])
        cores = []
        for i in range(n_cores):
            shape = tuple(int(x) for x in group[f"core_{i}_shape"][()])
            data = group[f"core_{i}"][()]
            cores.append(data.reshape(shape))
        return cls(cores)

    def at(self, indices: Iterable[int]) -> float:
        """Return one tensor entry."""
        indices = tuple(int(i) for i in indices)
        if len(indices) != len(self.cores):
            raise ValueError("Number of indices must match tensor-train order.")

        value = np.array([[1.0]])
        for core, i in zip(self.cores, indices):
            if i < 0 or i >= core.shape[1]:
                raise IndexError("Tensor-train index is out of bounds.")
            value = value @ core[:, i, :]
        return float(value[0, 0])

    def values(self, indices: Iterable[Iterable[int]]) -> np.ndarray:
        """Return tensor entries for multiple indices."""
        return np.array([self.at(index) for index in indices])
