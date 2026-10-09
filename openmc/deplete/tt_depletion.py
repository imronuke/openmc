"""Tensor-train depletion reaction-rate helper functions."""

import time
from itertools import repeat
from math import prod

import h5py
import numpy as np

from openmc.tt import (
    TT, _TT_LAYOUT_VERSION, _flat_to_tt_index, _write_packed_tt_channels)

from .pool import _solve_matrices
from .reaction_rates import ReactionRates


__all__ = ["TTDepletionRates"]


_TT_DEPLETION_RATES = "tt_depletion_reaction_rates"


class _TTReactionRates:
    """Normalization data for streamed TT reaction-rate slices."""

    _tt_reaction_rates = True

    def __init__(
            self, normalization_factor, zero_source=False,
            reaction_rate_mask=None):
        self.normalization_factor = normalization_factor
        self.zero_source = zero_source
        self.reaction_rate_mask = reaction_rate_mask


def _copy_tt_mean(tt, n_realizations):
    """Return a TT copy representing the tally mean."""
    cores = [core.copy() for core in tt.cores]
    if n_realizations > 0 and cores:
        cores[0] /= n_realizations
    return TT(cores, shape=tt.shape)


def _get_tt_depletion_rate_write_data(operator, rates):
    """Return TT depletion reaction-rate data for HDF5 output."""
    rate_helper = vars(operator).get('_rate_helper')
    tt_eps = None if rate_helper is None else vars(rate_helper).get('_tt_eps')
    tt_shape = None if rate_helper is None else vars(rate_helper).get('_tt_shape')
    reaction_rate_mask = getattr(rates, 'reaction_rate_mask', None)
    if reaction_rate_mask is None:
        reaction_rate_mask = vars(operator).get('_tt_reaction_rate_mask')
    if reaction_rate_mask is None:
        reaction_rate_mask = _tt_reaction_rate_mask(
            operator.chain, operator.reaction_rates)
    _, nuc_indices, reaction_indices = _tt_rate_indices(operator)
    channel_indices = np.asarray(
        [(i_nuc, i_rx) for i_nuc in nuc_indices
         for i_rx in reaction_indices], dtype=np.int64).reshape((-1, 2))

    if getattr(rates, 'zero_source', False):
        return {
            'zero_source': True,
            'tt_eps': tt_eps,
            'reaction_rate_mask': reaction_rate_mask,
            'normalization_factor': 0.0,
            'n_realizations': 0,
            'tt_shape': tuple() if tt_shape is None else tuple(tt_shape),
            'channel_indices': channel_indices,
            'tt_means': None,
        }

    tally = operator._rate_helper._rate_tally
    n_realizations = tally.num_realizations
    tt_shape = tally.tt_shape
    tt_sum = tuple(tally.tt_sum)
    if any(tt.shape != tt_shape for tt in tt_sum):
        raise RuntimeError(
            "Tensor-train channel shape does not match depletion filter "
            "shape.")
    if len(tt_sum) != len(channel_indices):
        raise RuntimeError(
            "Tensor-train channel count does not match depletion rate "
            "indices.")
    return {
        'zero_source': False,
        'tt_eps': tt_eps,
        'reaction_rate_mask': reaction_rate_mask,
        'normalization_factor': rates.normalization_factor,
        'n_realizations': n_realizations,
        'tt_shape': tt_shape,
        'channel_indices': channel_indices,
        'tt_means': tuple(
            _copy_tt_mean(tt, n_realizations) for tt in tt_sum),
    }


def _write_tt_depletion_rates(handle, step, data):
    """Write per-step TT depletion reaction-rate data to HDF5."""
    group = handle.require_group(_TT_DEPLETION_RATES)
    group.attrs['format_version'] = 2
    group.attrs['tt_layout_version'] = _TT_LAYOUT_VERSION
    group.attrs['stored_values'] = np.bytes_('tally_mean')
    group.attrs['normalization_mode'] = np.bytes_('fission-q')
    if data['tt_eps'] is not None:
        group.attrs['tt_eps'] = data['tt_eps']
    reaction_rate_mask = np.asarray(data['reaction_rate_mask'], dtype=bool)
    if 'reaction_rate_mask' in group:
        if not np.array_equal(group['reaction_rate_mask'][()], reaction_rate_mask):
            raise ValueError(
                "Tensor-train reaction-rate mask changed between "
                "depletion steps.")
    else:
        group.create_dataset('reaction_rate_mask', data=reaction_rate_mask)

    channel_indices = np.asarray(data['channel_indices'], dtype=np.int64)
    if 'channel_indices' in group:
        if not np.array_equal(group['channel_indices'][()], channel_indices):
            raise ValueError(
                "Tensor-train channel indices changed between depletion "
                "steps.")
    else:
        group.create_dataset('channel_indices', data=channel_indices)

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
        tt_mean_group = step_group.create_group('tt_mean')
        _write_packed_tt_channels(tt_mean_group, data['tt_means'])


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
            format_version = int(tt_group.attrs.get('format_version', -1))
            if format_version != 2:
                raise RuntimeError(
                    f"Unsupported TT depletion results format version "
                    f"{format_version}.")
            layout_version = int(
                tt_group.attrs.get('tt_layout_version', -1))
            if layout_version != _TT_LAYOUT_VERSION:
                raise RuntimeError(
                    f"Unsupported tensor-train layout version "
                    f"{layout_version}.")
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
                    "Tensor-train reaction-rate metadata is "
                    "incomplete.")

            if 'reaction_rate_mask' not in tt_group:
                raise RuntimeError(
                    "Tensor-train reaction-rate mask is missing.")
            self.reaction_rate_mask = tt_group['reaction_rate_mask'][()]
            expected_shape = (len(self.index_nuc), len(self.index_rx))
            if self.reaction_rate_mask.shape != expected_shape:
                raise RuntimeError(
                    "Tensor-train reaction-rate mask shape is "
                    "incompatible with reaction-rate metadata.")

            if 'channel_indices' not in tt_group:
                raise RuntimeError(
                    "Tensor-train channel indices are missing.")
            self.channel_indices = np.asarray(
                tt_group['channel_indices'][()], dtype=np.int64)
            if (self.channel_indices.ndim != 2 or
                    self.channel_indices.shape[1] != 2):
                raise RuntimeError(
                    "Tensor-train channel indices have an invalid shape.")
            if (np.any(self.channel_indices[:, 0] < 0) or
                    np.any(self.channel_indices[:, 0] >= len(self.index_nuc)) or
                    np.any(self.channel_indices[:, 1] < 0) or
                    np.any(self.channel_indices[:, 1] >= len(self.index_rx))):
                raise RuntimeError(
                    "Tensor-train channel indices are out of bounds.")
            if len({tuple(pair) for pair in self.channel_indices}) != len(
                    self.channel_indices):
                raise RuntimeError(
                    "Tensor-train channel indices contain duplicates.")

            self.n_steps = handle['number'].shape[0]

        self._cached_step = None
        self._cached_tt_means = None

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
            filter_shape = tt_shape
            n_filter_bins = prod(filter_shape)
            if mat_index < 0 or mat_index >= n_filter_bins:
                raise IndexError("Material index is out of bounds.")

            prefix_index = _flat_to_tt_index(mat_index, filter_shape)
            if self._cached_step != step:
                if 'tt_mean' not in step_group:
                    raise RuntimeError(
                        "Tensor-train means are missing from depletion step.")
                self._cached_tt_means = TT.from_hdf5_channels(
                    step_group['tt_mean'], tt_shape,
                    len(self.channel_indices))
                self._cached_step = step

            material_rates = self._empty_material_rates(mat)
            factor = (
                step_group['normalization_factor'][()] /
                (1e24 * self.volume[mat]))
            for (i_nuc, i_rx), tt_mean in zip(
                    self.channel_indices, self._cached_tt_means):
                material_rates[i_nuc, i_rx] = (
                    float(tt_mean._contract_slice(prefix_index)) * factor)
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
    number = np.zeros(rates.n_nuc)
    fission_rates = np.empty(rates.n_nuc) if fission_ind is not None else None
    fission_tally_index = (
        rx_ind.index(fission_ind) if fission_ind is not None else None)

    for i, mat in enumerate(operator.local_mats):
        mat_index = operator._mat_index_map[mat]
        number.fill(0.0)
        for nuc, i_nuc_results in zip(rxn_nuclides, nuc_ind):
            number[i_nuc_results] = operator.number[mat, nuc]

        fission_yields.append(operator._yield_helper.weighted_yields(i))

        if fission_ind is not None:
            tally_rates = operator._rate_helper.get_material_rates(
                mat_index, nuc_ind, rx_ind,
                tally_score_index=fission_tally_index)
            volume_b_cm = 1e24 * operator.number.get_mat_volume(mat)
            np.multiply(number, 1.0 / volume_b_cm, out=fission_rates)
            np.multiply(
                tally_rates[:, fission_ind], fission_rates,
                out=fission_rates)
            np.multiply(
                fission_rates, reaction_rate_mask[:, fission_ind],
                out=fission_rates)
            operator._normalization_helper.update(fission_rates)

    operator.chain.fission_yields = fission_yields
    return operator._normalization_helper.factor(source_rate)


def _get_tt_reaction_rates(
        operator, mat, normalization_factor, nuc_ind, rx_ind,
        tally_channel_indices):
    """Return normalized TT reaction rates for one material."""
    rates = operator.reaction_rates
    mat = str(mat)
    mat_index = operator._mat_index_map[mat]

    tally_rates = operator._rate_helper.get_material_rates(
        mat_index, nuc_ind, rx_ind,
        tally_channel_indices=tally_channel_indices)
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
    if len(n) != len(operator.local_mats):
        raise ValueError(
            "Number of compositions {} is not equal to the number of local "
            "materials {}.".format(len(n), len(operator.local_mats)))

    if transfer_rates is not None:
        raise ValueError(
            "Transfer rates are not supported when tensor-train "
            "reaction-rate mode is enabled.")
    if (external_source_rates is not None and
            current_timestep in external_source_rates.external_timesteps):
        raise ValueError(
            "External source rates are not supported when tensor-train "
            "reaction-rate mode is enabled.")

    start = time.time()
    fission_yields = chain.fission_yields
    if len(fission_yields) == 1:
        fission_yields = repeat(fission_yields[0])
    elif len(fission_yields) != len(n):
        raise ValueError(
            "Number of material fission yield distributions {} is not "
            "equal to the number of compositions {}".format(
                len(fission_yields), len(n)))

    if getattr(rates, 'zero_source', False):
        zero_rates = operator.reaction_rates[0].copy()
        zero_rates.fill(0.0)
        nuc_ind = rx_ind = active_tally_channels = None
    else:
        zero_rates = None
        reaction_rate_mask = getattr(rates, 'reaction_rate_mask', None)
        if reaction_rate_mask is None:
            reaction_rate_mask = vars(operator).get(
                '_tt_reaction_rate_mask')
        if reaction_rate_mask is None:
            reaction_rate_mask = _tt_reaction_rate_mask(
                chain, operator.reaction_rates)
        operator._tt_reaction_rate_mask = reaction_rate_mask
        _, nuc_ind, rx_ind = _tt_rate_indices(operator)
        active_tally_channels = np.flatnonzero(
            reaction_rate_mask[np.ix_(nuc_ind, rx_ind)].ravel()).tolist()

    def material_inputs():
        for mat, n_mat, yields in zip(
                operator.local_mats, n, fission_yields):
            if zero_rates is None:
                mat_rates = _get_tt_reaction_rates(
                    operator, mat, rates.normalization_factor, nuc_ind,
                    rx_ind, active_tally_channels)
            else:
                mat_rates = zero_rates
            matrix = chain.form_matrix(mat_rates, yields)
            yield matrix, n_mat, dt, substeps

    n_results = _solve_matrices(
        solver, material_inputs(), batch_size=128,
        parallel=len(operator.local_mats) > 1)
    for n_result in n_results:
        n_result.clip(min=0.0, out=n_result)

    return time.time() - start, n_results
