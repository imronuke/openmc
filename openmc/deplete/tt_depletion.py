"""Tensor-train depletion reaction-rate helper functions."""

import time
from itertools import repeat

import h5py
import numpy as np

from openmc.tt import TT, _flat_to_tt_index

from .reaction_rates import ReactionRates


__all__ = ["TTDepletionRates"]


_TT_DEPLETION_RATES = "tt_depletion_reaction_rates"


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


def _copy_tt_mean(tt, n_realizations):
    """Return a TT copy representing the tally mean."""
    cores = [core.copy() for core in tt.cores]
    if n_realizations > 0 and cores:
        cores[0] /= n_realizations
    return TT(cores)


def _write_tt_hdf5(group, name, tt):
    """Write a TT object to an HDF5 group."""
    tt_group = group.create_group(name)
    tt_group.create_dataset("n_cores", data=len(tt.cores))
    for i, core in enumerate(tt.cores):
        tt_group.create_dataset(f"core_{i}_shape", data=np.asarray(core.shape))
        tt_group.create_dataset(f"core_{i}", data=core.ravel())


def _get_tt_depletion_rate_write_data(operator, rates):
    """Return TT depletion reaction-rate data for HDF5 output."""
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
                "Tensor-train reaction-rate mask changed between "
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

            self.n_steps = handle['number'].shape[0]

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
    number = np.zeros(rates.n_nuc)
    fission_rates = np.empty(rates.n_nuc) if fission_ind is not None else None

    for i, mat in enumerate(operator.local_mats):
        mat_index = operator._mat_index_map[mat]
        number.fill(0.0)
        for nuc, i_nuc_results in zip(rxn_nuclides, nuc_ind):
            number[i_nuc_results] = operator.number[mat, nuc]

        tally_rates = operator._rate_helper.get_material_rates(
            mat_index, nuc_ind, rx_ind)
        fission_yields.append(operator._yield_helper.weighted_yields(i))

        if fission_ind is not None:
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

    results = []
    zero_rates = None
    reaction_rate_mask = getattr(rates, 'reaction_rate_mask', None)
    if reaction_rate_mask is not None:
        operator._tt_reaction_rate_mask = reaction_rate_mask
    if getattr(rates, 'zero_source', False):
        zero_rates = operator.reaction_rates[0].copy()
        zero_rates.fill(0.0)

    for mat, n_mat, yields in zip(operator.local_mats, n, fission_yields):
        if zero_rates is None:
            mat_rates = _get_tt_reaction_rates(
                operator, mat, rates.normalization_factor)
        else:
            mat_rates = zero_rates
        matrix = chain.form_matrix(mat_rates, yields)
        n_result = solver(matrix, n_mat, dt, substeps)
        n_result.clip(min=0.0, out=n_result)
        results.append(n_result)

    return time.time() - start, results
