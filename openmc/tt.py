from __future__ import annotations

from collections.abc import Iterable

import h5py
import numpy as np


def _flat_to_tt_index(flat_index, tt_shape):
    index = []
    for n in reversed(tt_shape):
        index.append(flat_index % n)
        flat_index //= n
    return tuple(reversed(index))


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

    def _contract_slice(self, prefix_index):
        state = np.array([1.0])
        for core, i in zip(self.cores[:len(prefix_index)], prefix_index):
            if i < 0 or i >= core.shape[1]:
                raise IndexError("Tensor-train index is out of bounds.")
            state = state @ core[:, i, :]

        for core in self.cores[len(prefix_index):]:
            state = np.tensordot(state, core, axes=([-1], [0]))

        return state[..., 0]


def _rank_truncation(singular_values, eps):
    singular_values = np.asarray(singular_values)
    total = np.sum(singular_values**2)
    if total == 0.0:
        return 1

    tail = 0.0
    for i in range(singular_values.size - 1, -1, -1):
        tail += singular_values[i]**2
        if np.sqrt(tail / total) >= eps:
            return i + 1
    return singular_values.size


def tt_svd(tensor, shape=None, ranks=None, eps=None):
    """Construct a tensor train using TT-SVD.

    Parameters
    ----------
    tensor : numpy.ndarray or iterable of float
        Dense tensor data in row-major order.
    shape : iterable of int, optional
        Logical tensor shape. If omitted, ``tensor.shape`` is used.
    ranks : iterable of int, optional
        Fixed TT ranks with length ``len(shape) - 1``. Mutually exclusive with
        ``eps``.
    eps : float, optional
        Relative singular-value tail tolerance used to truncate ranks. If not
        specified and ``ranks`` is not specified, a default of ``1.0e-10`` is
        used.

    Returns
    -------
    TT
        Tensor-train representation of ``tensor``.

    """
    dense = np.asarray(tensor, dtype=float)
    if shape is None:
        shape = dense.shape
    shape = tuple(int(n) for n in shape)
    if not shape:
        raise ValueError("TT-SVD requires a non-empty shape.")
    if any(n <= 0 for n in shape):
        raise ValueError("TT-SVD shape dimensions must be positive.")
    if dense.size != int(np.prod(shape)):
        raise ValueError("TT-SVD shape is incompatible with tensor size.")

    if ranks is None:
        ranks = ()
    else:
        ranks = tuple(int(r) for r in ranks)
    if ranks and eps is not None:
        raise ValueError("Specify either ranks or eps, not both.")
    if ranks and len(ranks) != len(shape) - 1:
        raise ValueError("ranks must have length len(shape) - 1.")
    if not ranks and eps is None:
        eps = 1.0e-10
    if eps is not None and eps < 0.0:
        raise ValueError("eps must be non-negative.")

    buf = dense.reshape(-1, order='C').copy()
    cores = []
    r_prev = 1

    for k, n_k in enumerate(shape[:-1]):
        rows = r_prev * n_k
        cols = buf.size // rows
        matrix = buf.reshape((rows, cols), order='C')
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)

        if ranks:
            r_k = min(s.size, ranks[k])
            if r_k <= 0:
                raise ValueError("TT ranks must be positive.")
        else:
            r_k = _rank_truncation(s, eps)
            r_k = max(1, min(r_k, s.size))

        cores.append(u[:, :r_k].reshape((r_prev, n_k, r_k), order='C'))
        buf = (s[:r_k, None] * vh[:r_k, :]).reshape(-1, order='C')
        r_prev = r_k

    cores.append(buf.reshape((r_prev, shape[-1], 1), order='C'))
    return TT(cores)
