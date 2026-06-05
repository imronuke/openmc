#!/usr/bin/env python3
"""Compare dense and tensor-train depletion reaction rates for 3D pin examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


EXAMPLES = Path(__file__).resolve().parent
DEFAULT_DENSE_RESULTS = EXAMPLES / "3d_pin_depletion" / "depletion_results_old.h5"
DEFAULT_TT_RESULTS = EXAMPLES / "3d_pin_depletion_tt" / "depletion_results_old.h5"


@dataclass
class Difference:
    abs_diff: float = 0.0
    rel_diff: float = -1.0
    step: int | None = None
    material: str | None = None
    nuclide: str | None = None
    reaction: str | None = None
    dense: float = 0.0
    tt: float = 0.0


def _check_results_file(path, required_group, regenerate_hint):
    if not path.is_file():
        raise SystemExit(
            f"Missing results file: {path}\n"
            f"Run the corresponding example first with write_rates=True.")

    with h5py.File(path, "r") as handle:
        if required_group not in handle:
            raise SystemExit(
                f"{path} does not contain /{required_group}.\n"
                f"{regenerate_hint}")


def _ordered_keys(index):
    return sorted(index, key=lambda key: index[key])


def _normalize_steps(step, n_steps):
    if step is None:
        return range(n_steps)
    if step < 0:
        step += n_steps
    if step < 0 or step >= n_steps:
        raise SystemExit(f"Step index {step} is out of bounds for {n_steps} steps.")
    return [step]


def _compare_step(step, dense_result, tt_reader, nucs, reactions, rtol, atol):
    n_values = 0
    n_fail = 0
    max_abs = 0.0
    max_rel = 0.0
    n_dense_zero = 0
    n_dense_zero_tt_nonzero = 0
    max_zero_ref_abs = 0.0
    worst = Difference()

    for mat in _ordered_keys(dense_result.index_mat):
        dense_mat_index = dense_result.index_mat[mat]
        dense_rates = dense_result.rates[dense_mat_index]
        tt_rates = tt_reader.get_material_rates(step, mat)

        for nuc in nucs:
            dense_nuc_index = dense_result.rates.index_nuc[nuc]
            tt_nuc_index = tt_reader.index_nuc[nuc]

            for reaction in reactions:
                dense_rx_index = dense_result.rates.index_rx[reaction]
                tt_rx_index = tt_reader.index_rx[reaction]

                dense_value = dense_rates[dense_nuc_index, dense_rx_index]
                tt_value = tt_rates[tt_nuc_index, tt_rx_index]
                if dense_value != 0.0:
                    abs_diff = abs(tt_value - dense_value)
                    rel_diff = abs_diff / abs(dense_value)
                    max_rel = max(max_rel, rel_diff)
                else:
                    abs_diff = 0.0
                    rel_diff = 0.0
                    n_dense_zero += 1
                    if tt_value != 0.0:
                        n_dense_zero_tt_nonzero += 1
                    max_zero_ref_abs = max(max_zero_ref_abs, abs(tt_value))

                max_abs = max(max_abs, abs_diff)
                n_values += 1
                if abs_diff > atol + rtol * abs(dense_value):
                    n_fail += 1

                if (rel_diff, abs_diff) > (worst.rel_diff, worst.abs_diff):
                    worst = Difference(
                        abs_diff=abs_diff,
                        rel_diff=rel_diff,
                        step=step,
                        material=mat,
                        nuclide=nuc,
                        reaction=reaction,
                        dense=dense_value,
                        tt=tt_value,
                    )

    return (
        n_values, n_fail, max_abs, max_rel, n_dense_zero,
        n_dense_zero_tt_nonzero, max_zero_ref_abs, worst)


def _keff_value(keff):
    """Return the central value from stored depletion eigenvalue data."""
    if hasattr(keff, "nominal_value"):
        return keff.nominal_value
    keff = np.asarray(keff)
    if keff.ndim == 0:
        return float(keff)
    return float(keff[0])


def compare(args):
    _check_results_file(
        args.dense_results,
        "reaction rates",
        "Re-run examples/3d_pin_depletion/3d_depletion.py with "
        "integrator.integrate(write_rates=True).",
    )
    _check_results_file(
        args.tt_results,
        "tt_depletion_reaction_rates",
        "Re-run examples/3d_pin_depletion_tt/3d_depletion.py with "
        "integrator.integrate(write_rates=True).",
    )

    import openmc.deplete

    dense_results = openmc.deplete.Results(args.dense_results)
    tt_reader = openmc.deplete.TTDepletionRates(args.tt_results)
    with h5py.File(args.tt_results, "r") as handle:
        tt_keff = handle["eigenvalues"][:, 0]

    if len(dense_results) != tt_reader.n_steps:
        raise SystemExit(
            "The dense and TT result files contain different numbers of steps: "
            f"{len(dense_results)} != {tt_reader.n_steps}.")

    first_dense = dense_results[0]
    dense_mats = set(first_dense.index_mat)
    tt_mats = set(tt_reader.index_mat)
    if dense_mats != tt_mats:
        raise SystemExit(
            "The dense and TT result files contain different material IDs.")

    dense_nucs = set(first_dense.rates.index_nuc)
    tt_nucs = set(tt_reader.index_nuc)
    if dense_nucs != tt_nucs:
        raise SystemExit(
            "The dense and TT result files contain different reaction-rate "
            "nuclides.")

    dense_reactions = set(first_dense.rates.index_rx)
    tt_reactions = set(tt_reader.index_rx)
    if dense_reactions != tt_reactions:
        raise SystemExit(
            "The dense and TT result files contain different reaction names.")

    nucs = _ordered_keys(first_dense.rates.index_nuc)
    reactions = _ordered_keys(first_dense.rates.index_rx)
    steps = _normalize_steps(args.step, len(dense_results))

    total_values = 0
    total_fail = 0
    global_max_abs = 0.0
    global_max_rel = 0.0
    global_dense_zero = 0
    global_dense_zero_tt_nonzero = 0
    global_max_zero_ref_abs = 0.0
    global_worst = Difference()

    print(f"Dense results: {args.dense_results}")
    print(f"TT results:    {args.tt_results}")
    print(f"Materials:     {len(dense_mats)}")
    print(f"Nuclides:      {len(nucs)}")
    print(f"Reactions:     {len(reactions)}")
    if tt_reader.tt_eps is not None:
        print(f"TT eps:        {tt_reader.tt_eps:g}")
    else:
        print("TT eps:        not stored")
    print(f"Tolerance:     rtol={args.rtol:g}, atol={args.atol:g}")

    for step in steps:
        result = _compare_step(
            step, dense_results[step], tt_reader, nucs, reactions,
            args.rtol, args.atol)
        (
            n_values, n_fail, max_abs, max_rel, n_dense_zero,
            n_dense_zero_tt_nonzero, max_zero_ref_abs, worst) = result

        total_values += n_values
        total_fail += n_fail
        global_max_abs = max(global_max_abs, max_abs)
        global_max_rel = max(global_max_rel, max_rel)
        global_dense_zero += n_dense_zero
        global_dense_zero_tt_nonzero += n_dense_zero_tt_nonzero
        global_max_zero_ref_abs = max(global_max_zero_ref_abs, max_zero_ref_abs)
        if (worst.rel_diff, worst.abs_diff) > (
                global_worst.rel_diff, global_worst.abs_diff):
            global_worst = worst

        print()
        print(f"Step {step}:")
        print(
            f"  keff                         "
            f"dense={_keff_value(dense_results[step].k):.5f}, "
            f"TT={_keff_value(tt_keff[step]):.5f}")
        print(f"  compared values              {n_values}")
        print(f"  values outside tolerance      {n_fail}")
        print(f"  max abs difference            {max_abs:.16e}")
        print(f"  max rel difference            {max_rel:.16e}")
        print(f"  dense-zero values             {n_dense_zero}")
        print(f"  dense-zero with nonzero TT     {n_dense_zero_tt_nonzero}")
        print(f"  max abs TT value where dense=0 {max_zero_ref_abs:.16e}")

    print()
    print("Global:")
    print(f"  compared values              {total_values}")
    print(f"  values outside tolerance      {total_fail}")
    print(f"  max abs difference            {global_max_abs:.16e}")
    print(f"  max rel difference            {global_max_rel:.16e}")
    print(f"  dense-zero values             {global_dense_zero}")
    print(f"  dense-zero with nonzero TT     {global_dense_zero_tt_nonzero}")
    print(f"  max abs TT value where dense=0 {global_max_zero_ref_abs:.16e}")
    print("  worst relative-difference entry")
    print(f"    step                        {global_worst.step}")
    print(f"    material                    {global_worst.material}")
    print(f"    nuclide                     {global_worst.nuclide}")
    print(f"    reaction                    {global_worst.reaction}")
    print(f"    dense                       {global_worst.dense:.16e}")
    print(f"    TT                          {global_worst.tt:.16e}")
    print(f"    abs difference              {global_worst.abs_diff:.16e}")
    print(f"    rel difference              {global_worst.rel_diff:.16e}")

    if total_fail > 0:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare reaction rates from the non-TT and TT 3D pin depletion "
            "examples. Both depletion_results.h5 files must be generated with "
            "write_rates=True."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dense-results", type=Path, default=DEFAULT_DENSE_RESULTS,
        help="Non-TT depletion_results.h5 file.")
    parser.add_argument(
        "--tt-results", type=Path, default=DEFAULT_TT_RESULTS,
        help="TT depletion_results.h5 file.")
    parser.add_argument(
        "--step", type=int,
        help="Compare only one depletion step. By default all steps are compared.")
    parser.add_argument(
        "--rtol", type=float, default=1.0e-2,
        help="Relative tolerance used to count values outside tolerance.")
    parser.add_argument(
        "--atol", type=float, default=1.0e-30,
        help="Absolute tolerance used to count values outside tolerance.")
    args = parser.parse_args()

    compare(args)


if __name__ == "__main__":
    main()
