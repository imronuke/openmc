"""Tensor-train depletion helper functions."""

import time
from itertools import repeat

import numpy as np


class _TTReactionRates:
    """Normalization data for streamed TT reaction-rate slices."""

    _tt_reaction_rates = True

    def __init__(self, normalization_factor):
        self.normalization_factor = normalization_factor


def _tt_rate_indices(operator):
    rates = operator.reaction_rates
    rxn_nuclides = operator._rate_helper.nuclides
    nuc_ind = [rates.index_nuc[nuc] for nuc in rxn_nuclides]
    rx_ind = [rates.index_rx[react] for react in operator.chain.reactions]
    return rxn_nuclides, nuc_ind, rx_ind


def _prepare_tt_reaction_rates(operator, source_rate):
    """Prepare fission-Q normalization for TT material rate slices."""
    rates = operator.reaction_rates
    rxn_nuclides, nuc_ind, rx_ind = _tt_rate_indices(operator)
    fission_ind = rates.index_rx.get("fission")

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
    elif len(fission_yields) != len(n):
        raise ValueError(
            "Number of material fission yield distributions {} is not "
            "equal to the number of compositions {}".format(
                len(fission_yields), len(n)))

    results = []
    for mat, n_mat, yields in zip(operator.local_mats, n, fission_yields):
        mat_rates = _get_tt_reaction_rates(
            operator, mat, rates.normalization_factor)
        matrix = chain.form_matrix(mat_rates, yields)
        n_result = solver(matrix, n_mat, dt, substeps)
        n_result.clip(min=0.0, out=n_result)
        results.append(n_result)

    return time.time() - start, results
