"""Dedicated module containing depletion function

Provided to avoid some circular imports
"""
import multiprocessing
import os
import sys
from itertools import islice, repeat, starmap
from multiprocessing import Pool
from threading import Event, Thread

import numpy as np
from scipy.sparse import hstack

from openmc.mpi import comm
from .._sparse_compat import block_array

# Configurable switch that enables / disables the use of
# multiprocessing routines during depletion
USE_MULTIPROCESSING = True

# Allow user to override the number of worker processes to use for depletion
# calculations
NUM_PROCESSES = None

_MEMORY_SAMPLE_INTERVAL = 0.25
_BYTES_PER_MIB = 1024 * 1024


def _read_process_pss(pid):
    """Return a process's proportional set size in bytes on Linux."""
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as stat:
            for line in stat:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_process_rss(pid):
    """Return a process's resident set size in bytes on Linux."""
    try:
        with open(f"/proc/{pid}/statm", encoding="ascii") as stat:
            resident_pages = int(stat.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class _ProcessMemorySampler:
    """Sample aggregate memory for the Python process and its pool workers."""

    def __init__(self, pids):
        self.pids = tuple(pids)
        self.memory_kind = "PSS"
        if not all(_read_process_pss(pid) is not None for pid in self.pids):
            self.memory_kind = "RSS"
            if not all(_read_process_rss(pid) is not None for pid in self.pids):
                raise RuntimeError("Unable to read process memory from /proc.")

        self._read_memory = (
            _read_process_pss if self.memory_kind == "PSS"
            else _read_process_rss)
        self._stop = Event()
        self._thread = Thread(target=self._sample_periodically, daemon=True)
        self._memory_sum = 0
        self._sample_count = 0
        self._peak_memory = 0

    def _sample(self):
        values = [self._read_memory(pid) for pid in self.pids]
        if any(value is None for value in values):
            return

        total_memory = sum(values)
        self._memory_sum += total_memory
        self._sample_count += 1
        self._peak_memory = max(self._peak_memory, total_memory)

    def _sample_periodically(self):
        while not self._stop.wait(_MEMORY_SAMPLE_INTERVAL):
            self._sample()

    def start(self):
        self._sample()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._sample()
        if self._sample_count == 0:
            return None

        return (
            self._memory_sum / self._sample_count,
            self._peak_memory,
            self.memory_kind,
        )


def _report_pool_memory(sampler, n_workers):
    if sampler is None:
        return

    memory = sampler.stop()
    if memory is None:
        return

    average_memory, peak_memory, memory_kind = memory
    print(
        f"[openmc.deplete] Bateman process memory (rank {comm.rank}, "
        f"parent + {n_workers} workers; sampled aggregate {memory_kind}): "
        f"average={average_memory / _BYTES_PER_MIB:.1f} MiB, "
        f"peak={peak_memory / _BYTES_PER_MIB:.1f} MiB",
        flush=True,
    )


def _distribute(items):
    """Distribute items across MPI communicator

    Parameters
    ----------
    items : list
        List of items of distribute

    Returns
    -------
    list
        Items assigned to process that called

    """
    min_size, extra = divmod(len(items), comm.size)
    j = 0
    for i in range(comm.size):
        chunk_size = min_size + int(i < extra)
        if comm.rank == i:
            return items[j:j + chunk_size]
        j += chunk_size


def _solve_matrices(func, inputs, batch_size=None, parallel=True):
    """Apply a depletion solver to material matrices.

    Parameters
    ----------
    func : callable
        Depletion solver accepting a matrix, composition, time interval, and
        substep count.
    inputs : iterable
        Tuples of arguments for ``func``.
    batch_size : int, optional
        Maximum number of prepared material matrices held while submitting
        work to worker processes. If omitted, all inputs are submitted at once.
    parallel : bool, optional
        Whether to use multiprocessing when it is enabled globally.

    Returns
    -------
    list of numpy.ndarray
        Depleted compositions in input order.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("Solver batch size must be positive.")

    if USE_MULTIPROCESSING and parallel:
        results = []
        with Pool(NUM_PROCESSES) as pool:
            sampler = None
            if sys.platform == "linux":
                try:
                    workers = multiprocessing.active_children()
                    sampler = _ProcessMemorySampler(
                        [os.getpid(), *(worker.pid for worker in workers)])
                    sampler.start()
                except (OSError, RuntimeError):
                    sampler = None

            try:
                if batch_size is None:
                    return list(pool.starmap(func, inputs))

                inputs = iter(inputs)
                while True:
                    batch = list(islice(inputs, batch_size))
                    if not batch:
                        break
                    results.extend(pool.starmap(func, batch))
            finally:
                _report_pool_memory(sampler, len(workers) if sampler else 0)
        return results

    return list(starmap(func, inputs))


def deplete(func, chain, n, rates, dt, current_timestep=None, matrix_func=None,
            transfer_rates=None, external_source_rates=None, substeps=1,
            *matrix_args):
    """Deplete materials using given reaction rates for a specified time

    Parameters
    ----------
    func : callable
        Function to use to get new compositions. Expected to have the signature
        ``func(A, n0, t, substeps=1) -> n1``.
    chain : openmc.deplete.Chain
        Depletion chain
    n : list of numpy.ndarray
        List of atom number arrays for each material. Each array in the list
        contains the number of [atom] of each nuclide.
    rates : openmc.deplete.ReactionRates
        Reaction rates (from transport operator)
    dt : float
        Time in [s] to deplete for
    current_timestep : int
        Current timestep index
    maxtrix_func : callable, optional
        Function to form the depletion matrix after calling ``matrix_func(chain,
        rates, fission_yields)``, where ``fission_yields = {parent: {product:
        yield_frac}}`` Expected to return the depletion matrix required by
        ``func``
    transfer_rates : openmc.deplete.TransferRates, Optional
        Transfer rates for continuous removal/feed.

        .. versionadded:: 0.14.0
    external_source_rates : openmc.deplete.ExternalSourceRates, Optional
        External source rates for continuous removal/feed.

        .. versionadded:: 0.15.3
    substeps : int, optional
        Number of substeps to pass to solvers that support substepping.
    matrix_args: Any, optional
        Additional arguments passed to matrix_func

    Returns
    -------
    n_result : list of numpy.ndarray
        Updated list of atom number arrays for each material. Each array in the
        list contains the number of [atom] of each nuclide.

    """

    fission_yields = chain.fission_yields
    if len(fission_yields) == 1:
        fission_yields = repeat(fission_yields[0])
    elif len(fission_yields) != len(n):
        raise ValueError(
            "Number of material fission yield distributions {} is not "
            "equal to the number of compositions {}".format(
                len(fission_yields), len(n)))

    if matrix_func is None:
        matrices = map(chain.form_matrix, rates, fission_yields)
    else:
        matrices = map(matrix_func, repeat(chain), rates, fission_yields,
                       *matrix_args)

    if (transfer_rates is not None and
        current_timestep in transfer_rates.external_timesteps):
        # Calculate transfer rate terms as diagonal matrices
        transfers = map(chain.form_rr_term, repeat(transfer_rates),
                        repeat(current_timestep), transfer_rates.local_mats)

        # Subtract transfer rate terms from Bateman matrices
        matrices = [matrix - transfer for (matrix, transfer) in zip(matrices,
                                                                    transfers)]

        if transfer_rates.redox:
            for mat_idx, mat_id in enumerate(transfer_rates.local_mats):
                if mat_id in transfer_rates.redox:
                    matrices[mat_idx] = chain.add_redox_term(matrices[mat_idx],
                                                transfer_rates.redox[mat_id][0],
                                                transfer_rates.redox[mat_id][1])

        if current_timestep in transfer_rates.index_transfer:
            # Gather all on comm.rank 0
            matrices = comm.gather(matrices)
            n = comm.gather(n)

            if comm.rank == 0:
                # Expand lists
                matrices = [elm for matrix in matrices for elm in matrix]
                n = [n_elm for n_mat in n for  n_elm in n_mat]

                # Calculate transfer rate terms as diagonal matrices
                transfer_pair = {}
                for mat_pair in transfer_rates.index_transfer[current_timestep]:
                    transfer_matrix = chain.form_rr_term(transfer_rates,
                                                         current_timestep,
                                                         mat_pair)

                    # check if destination material has a redox control
                    if mat_pair[0] in transfer_rates.redox:
                        transfer_matrix = chain.add_redox_term(transfer_matrix,
                                          transfer_rates.redox[mat_pair[0]][0],
                                          transfer_rates.redox[mat_pair[0]][1])
                    transfer_pair[mat_pair] = transfer_matrix

                # Combine all matrices together in a single matrix of matrices
                # to be solved in one go
                n_rows = n_cols = len(transfer_rates.burnable_mats)
                rows = []
                for row in range(n_rows):
                    cols = []
                    for col in range(n_cols):
                        mat_pair = (transfer_rates.burnable_mats[row],
                                    transfer_rates.burnable_mats[col])
                        if row == col:
                            # Fill the diagonals with the Bateman matrices
                            cols.append(matrices[row])
                        elif mat_pair in transfer_rates.index_transfer[current_timestep]:
                            # Fill the off-diagonals with the transfer pair matrices
                            cols.append(transfer_pair[mat_pair])
                        else:
                            cols.append(None)

                    rows.append(cols)
                matrix = block_array(rows)

                # Concatenate vectors of nuclides in one
                n_multi = np.concatenate(n)
                n_result = func(matrix, n_multi, dt, substeps)

                # Split back the nuclide vector result into the original form
                n_result = np.split(n_result, np.cumsum([len(i) for i in n])[:-1])

            else:
                n_result = None

            # Braodcast result to other ranks
            n_result = comm.bcast(n_result)
            # Distribute results across MPI
            n_result = _distribute(n_result)

            return n_result

    if (external_source_rates is not None and
        current_timestep in external_source_rates.external_timesteps):
        # Calculate external source term vectors
        sources = map(chain.form_ext_source_term, repeat(external_source_rates),
                      repeat(current_timestep), external_source_rates.local_mats)

        # stack vector column at the end of the matrix
        matrices = [
            hstack([matrix, source])
            for matrix, source in zip(matrices, sources)
        ]

        # Add a last row of zeroes to the matrices and append 1 to the last row
        # of the nuclide vectors
        for i, matrix in enumerate(matrices):
            if not np.equal(*matrix.shape):
                matrix.resize(matrix.shape[1], matrix.shape[1])
                n[i] = np.append(n[i], 1.0)

    inputs = zip(matrices, n, repeat(dt), repeat(substeps))
    n_result = _solve_matrices(func, inputs)

    # Remove extra value at the end of the nuclide vectors
    if (external_source_rates is not None and
        current_timestep in external_source_rates.external_timesteps):
        external_source_rates.reformat_nuclide_vectors(n)
        external_source_rates.reformat_nuclide_vectors(n_result)

    return n_result
