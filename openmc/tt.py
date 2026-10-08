from __future__ import annotations

from collections.abc import Iterable
from math import prod

import h5py
import numpy as np


__all__ = ["TT"]

_TT_LAYOUT_VERSION = 1


def _read_packed_tt_metadata(group, shape, n_channels):
    """Read and validate the packed channel/core tables from an HDF5 group."""
    shape = tuple(int(n) for n in shape)
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("Tensor-train shape must be nonempty and positive.")

    channel_offsets = np.asarray(group["channel_core_offsets"][()],
                                 dtype=np.int64)
    core_shapes = np.asarray(group["core_shapes"][()], dtype=np.int64)
    data_offsets = np.asarray(group["core_data_offsets"][()],
                               dtype=np.int64)
    core_data = group["core_data"]
    if core_data.ndim != 1:
        raise ValueError("Tensor-train core data must be one-dimensional.")
    if channel_offsets.ndim != 1 or len(channel_offsets) != n_channels + 1:
        raise ValueError("Tensor-train channel core offsets have an invalid shape.")
    if (core_shapes.ndim != 1 or len(core_shapes) % 3 != 0):
        raise ValueError("Tensor-train core shape table has an invalid shape.")

    n_cores = len(core_shapes) // 3
    core_shapes = core_shapes.reshape((n_cores, 3))
    if data_offsets.ndim != 1 or len(data_offsets) != n_cores + 1:
        raise ValueError("Tensor-train core data offsets have an invalid shape.")
    if (channel_offsets[0] != 0 or channel_offsets[-1] != n_cores or
            np.any(np.diff(channel_offsets) < 0)):
        raise ValueError("Tensor-train channel core offsets are inconsistent.")
    if (data_offsets[0] != 0 or data_offsets[-1] != core_data.shape[0] or
            np.any(np.diff(data_offsets) < 0)):
        raise ValueError("Tensor-train core data offsets are inconsistent.")

    for channel in range(n_channels):
        first_core = int(channel_offsets[channel])
        end_core = int(channel_offsets[channel + 1])
        if first_core == end_core:
            continue
        if end_core - first_core != len(shape):
            raise ValueError(
                "Tensor-train channel core count does not match its shape.")
        previous_rank = 1
        for i in range(first_core, end_core):
            r_left, mode, r_right = (int(n) for n in core_shapes[i])
            if min(r_left, mode, r_right) <= 0:
                raise ValueError("Tensor-train core dimensions must be positive.")
            if mode != shape[i - first_core]:
                raise ValueError(
                    "Tensor-train core modes do not match the tensor shape.")
            if r_left != previous_rank:
                raise ValueError("Tensor-train channel ranks are incompatible.")
            data_size = int(data_offsets[i + 1] - data_offsets[i])
            if data_size != prod((r_left, mode, r_right)):
                raise ValueError(
                    "Tensor-train core data size does not match its shape.")
            previous_rank = r_right
        if previous_rank != 1:
            raise ValueError("Tensor-train boundary ranks must be one.")

    return channel_offsets, core_shapes, data_offsets


def _read_packed_tt_ranks(group, shape, n_channels):
    """Return one rank tuple for each channel in a packed TT group."""
    shape = tuple(int(n) for n in shape)
    channel_offsets, core_shapes, _ = _read_packed_tt_metadata(
        group, shape, n_channels)
    ranks_by_channel = []
    for channel in range(n_channels):
        first = int(channel_offsets[channel])
        end = int(channel_offsets[channel + 1])
        if first == end:
            ranks_by_channel.append((1,) * (len(shape) + 1))
        else:
            ranks = [int(core_shapes[first, 0])]
            ranks.extend(int(core_shapes[i, 2]) for i in range(first, end))
            ranks_by_channel.append(tuple(ranks))
    return ranks_by_channel


def _write_packed_tt_channels(group, channels):
    """Write TT channels using the packed statepoint core representation."""
    channels = tuple(channels)
    channel_core_offsets = [0]
    core_shapes = []
    core_data_offsets = [0]
    expected_shape = None
    n_data = sum(core.size for channel in channels for core in channel.cores)
    core_data = np.empty(n_data, dtype=np.float64)
    data_index = 0

    for channel in channels:
        if expected_shape is None:
            expected_shape = channel.shape
        elif channel.shape != expected_shape:
            raise ValueError("Tensor-train channels must share one shape.")

        if channel.cores and len(channel.cores) != len(channel.shape):
            raise ValueError(
                "Tensor-train channel core count does not match its shape.")
        previous_rank = 1
        for i, core in enumerate(channel.cores):
            r_left, mode, r_right = core.shape
            if min(r_left, mode, r_right) <= 0:
                raise ValueError(
                    "Tensor-train core dimensions must be positive.")
            if mode != channel.shape[i] or r_left != previous_rank:
                raise ValueError(
                    "Tensor-train channel cores are incompatible.")
            core_shapes.extend((r_left, mode, r_right))
            flat_core = np.asarray(core, dtype=np.float64).ravel()
            next_data_index = data_index + flat_core.size
            core_data[data_index:next_data_index] = flat_core
            data_index = next_data_index
            core_data_offsets.append(data_index)
            previous_rank = r_right
        if channel.cores and previous_rank != 1:
            raise ValueError("Tensor-train boundary ranks must be one.")
        channel_core_offsets.append(len(core_data_offsets) - 1)

    group.create_dataset(
        "channel_core_offsets", data=np.asarray(channel_core_offsets,
                                                  dtype=np.int64))
    group.create_dataset(
        "core_shapes", data=np.asarray(core_shapes, dtype=np.int32))
    group.create_dataset(
        "core_data_offsets", data=np.asarray(core_data_offsets,
                                              dtype=np.int64))
    group.create_dataset("core_data", data=core_data)


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
    shape : iterable of int, optional
        Logical tensor shape. Required when ``cores`` is empty, which
        represents an all-zero tensor-train channel.

    """

    def __init__(self, cores: Iterable[np.ndarray], shape=None):
        self.cores = [np.asarray(core) for core in cores]
        for core in self.cores:
            if core.ndim != 3:
                raise ValueError("Each tensor-train core must be three-dimensional.")
        if self.cores:
            self._shape = tuple(core.shape[1] for core in self.cores)
            if shape is not None and tuple(shape) != self._shape:
                raise ValueError("Provided tensor-train shape does not match its cores.")
        elif shape is None:
            self._shape = None
        else:
            self._shape = tuple(int(n) for n in shape)
            if not self._shape or any(n <= 0 for n in self._shape):
                raise ValueError("Empty tensor-train shape must be nonempty and positive.")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape or tuple()

    @property
    def ranks(self) -> tuple[int, ...]:
        if not self.cores:
            return (1,) * (len(self.shape) + 1) if self.shape else tuple()
        return (self.cores[0].shape[0],) + tuple(core.shape[2] for core in self.cores)

    @classmethod
    def from_hdf5(cls, group: h5py.Group, shape=None):
        """Read a tensor train from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            Group containing the tensor-train cores.
        shape : iterable of int, optional
            Logical tensor shape, required for an empty all-zero tensor train.
        """
        n_cores = int(group["n_cores"][()])
        cores = []
        for i in range(n_cores):
            core_shape = tuple(int(x) for x in group[f"core_{i}_shape"][()])
            data = group[f"core_{i}"][()]
            cores.append(data.reshape(core_shape))
        return cls(cores, shape=shape)

    @classmethod
    def from_hdf5_channels(cls, group, shape, n_channels):
        """Read all channels from the packed TT HDF5 representation.

        Parameters
        ----------
        group : h5py.Group
            Group containing the packed channel/core tables.
        shape : iterable of int
            Spatial shape shared by the channel tensor trains.
        n_channels : int
            Expected number of nuclide-score channels.
        """
        channel_offsets, core_shapes, data_offsets = _read_packed_tt_metadata(
            group, shape, n_channels)
        core_data = group["core_data"][()]
        shape = tuple(int(n) for n in shape)
        channels = []
        for channel in range(n_channels):
            cores = []
            first = int(channel_offsets[channel])
            end = int(channel_offsets[channel + 1])
            for i in range(first, end):
                core_shape = tuple(int(n) for n in core_shapes[i])
                start = int(data_offsets[i])
                stop = int(data_offsets[i + 1])
                cores.append(core_data[start:stop].reshape(core_shape))
            channels.append(cls(cores, shape=shape))
        return tuple(channels)

    def at(self, indices: Iterable[int]) -> float:
        """Return one tensor entry."""
        indices = tuple(int(i) for i in indices)
        if not self.cores:
            if self._shape is None:
                raise ValueError("An empty tensor train requires a known shape.")
            if len(indices) != len(self._shape):
                raise ValueError("Number of indices must match tensor-train order.")
            if any(i < 0 or i >= n for i, n in zip(indices, self._shape)):
                raise IndexError("Tensor-train index is out of bounds.")
            return 0.0
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
        prefix_index = tuple(int(i) for i in prefix_index)
        if not self.cores:
            if self._shape is None:
                raise ValueError("An empty tensor train requires a known shape.")
            if len(prefix_index) > len(self._shape):
                raise ValueError("Slice index has more dimensions than the tensor.")
            if any(i < 0 or i >= n for i, n in
                   zip(prefix_index, self._shape)):
                raise IndexError("Tensor-train index is out of bounds.")
            remaining_shape = self._shape[len(prefix_index):]
            return np.zeros(remaining_shape) if remaining_shape else np.asarray(0.0)

        if len(prefix_index) > len(self.cores):
            raise ValueError("Slice index has more dimensions than the tensor.")
        state = np.array([1.0])
        for core, i in zip(self.cores[:len(prefix_index)], prefix_index):
            if i < 0 or i >= core.shape[1]:
                raise IndexError("Tensor-train index is out of bounds.")
            state = state @ core[:, i, :]

        for core in self.cores[len(prefix_index):]:
            state = np.tensordot(state, core, axes=([-1], [0]))

        return state[..., 0]
