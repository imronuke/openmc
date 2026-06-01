from collections.abc import Mapping
from ctypes import c_int, c_int32, c_size_t, c_double, c_char_p, c_bool, POINTER
from weakref import WeakValueDictionary

import numpy as np
from numpy.ctypeslib import as_array
import scipy.stats

from openmc.exceptions import AllocationError, InvalidIDError
from openmc.data.reaction import REACTION_NAME
from openmc.tt import TT
from . import _dll, Nuclide
from .core import _FortranObjectWithID
from .error import _error_handler
from .filter import _get_filter


__all__ = ['Tally', 'tallies', 'global_tallies', 'num_realizations']

# Tally functions
_dll.openmc_extend_tallies.argtypes = [c_int32, POINTER(c_int32), POINTER(c_int32)]
_dll.openmc_extend_tallies.restype = c_int
_dll.openmc_extend_tallies.errcheck = _error_handler
_dll.openmc_get_tally_index.argtypes = [c_int32, POINTER(c_int32)]
_dll.openmc_get_tally_index.restype = c_int
_dll.openmc_get_tally_index.errcheck = _error_handler
_dll.openmc_global_tallies.argtypes = [POINTER(POINTER(c_double))]
_dll.openmc_global_tallies.restype = c_int
_dll.openmc_global_tallies.errcheck = _error_handler
_dll.openmc_tally_get_active.argtypes = [c_int32, POINTER(c_bool)]
_dll.openmc_tally_get_active.restype = c_int
_dll.openmc_tally_get_active.errcheck = _error_handler
_dll.openmc_tally_get_estimator.argtypes = [c_int32, POINTER(c_int)]
_dll.openmc_tally_get_estimator.restype = c_int
_dll.openmc_tally_get_estimator.errcheck = _error_handler
_dll.openmc_tally_get_id.argtypes = [c_int32, POINTER(c_int32)]
_dll.openmc_tally_get_id.restype = c_int
_dll.openmc_tally_get_id.errcheck = _error_handler
_dll.openmc_tally_get_filters.argtypes = [
    c_int32, POINTER(POINTER(c_int32)), POINTER(c_size_t)]
_dll.openmc_tally_get_filters.restype = c_int
_dll.openmc_tally_get_filters.errcheck = _error_handler
_dll.openmc_tally_get_multiply_density.argtypes = [c_int32, POINTER(c_bool)]
_dll.openmc_tally_get_multiply_density.restype = c_int
_dll.openmc_tally_get_multiply_density.errcheck = _error_handler
_dll.openmc_tally_get_n_realizations.argtypes = [c_int32, POINTER(c_int32)]
_dll.openmc_tally_get_n_realizations.restype = c_int
_dll.openmc_tally_get_n_realizations.errcheck = _error_handler
_dll.openmc_tally_get_nuclides.argtypes = [
    c_int32, POINTER(POINTER(c_int)), POINTER(c_int)]
_dll.openmc_tally_get_nuclides.restype = c_int
_dll.openmc_tally_get_nuclides.errcheck = _error_handler
_dll.openmc_tally_get_scores.argtypes = [
    c_int32, POINTER(POINTER(c_int)), POINTER(c_int)]
_dll.openmc_tally_get_scores.restype = c_int
_dll.openmc_tally_get_scores.errcheck = _error_handler
_dll.openmc_tally_get_use_tt.argtypes = [c_int32, POINTER(c_bool)]
_dll.openmc_tally_get_use_tt.restype = c_int
_dll.openmc_tally_get_use_tt.errcheck = _error_handler
_dll.openmc_tally_get_tt_n_cores.argtypes = [c_int32, c_int, POINTER(c_int)]
_dll.openmc_tally_get_tt_n_cores.restype = c_int
_dll.openmc_tally_get_tt_n_cores.errcheck = _error_handler
_dll.openmc_tally_get_tt_core.argtypes = [
    c_int32, c_int, c_int, POINTER(POINTER(c_double)), POINTER(c_int*3)]
_dll.openmc_tally_get_tt_core.restype = c_int
_dll.openmc_tally_get_tt_core.errcheck = _error_handler
_dll.openmc_tally_get_type.argtypes = [c_int32, POINTER(c_int32)]
_dll.openmc_tally_get_type.restype = c_int
_dll.openmc_tally_get_type.errcheck = _error_handler
_dll.openmc_tally_get_writable.argtypes = [c_int32, POINTER(c_bool)]
_dll.openmc_tally_get_writable.restype = c_int
_dll.openmc_tally_get_writable.errcheck = _error_handler
_dll.openmc_tally_reset.argtypes = [c_int32]
_dll.openmc_tally_reset.restype = c_int
_dll.openmc_tally_reset.errcheck = _error_handler
_dll.openmc_tally_results.argtypes = [
    c_int32, POINTER(POINTER(c_double)), POINTER(c_size_t*3)]
_dll.openmc_tally_results.restype = c_int
_dll.openmc_tally_results.errcheck = _error_handler
_dll.openmc_tally_set_active.argtypes = [c_int32, c_bool]
_dll.openmc_tally_set_active.restype = c_int
_dll.openmc_tally_set_active.errcheck = _error_handler
_dll.openmc_tally_set_filters.argtypes = [c_int32, c_size_t, POINTER(c_int32)]
_dll.openmc_tally_set_filters.restype = c_int
_dll.openmc_tally_set_filters.errcheck = _error_handler
_dll.openmc_tally_set_estimator.argtypes = [c_int32, c_char_p]
_dll.openmc_tally_set_estimator.restype = c_int
_dll.openmc_tally_set_estimator.errcheck = _error_handler
_dll.openmc_tally_set_id.argtypes = [c_int32, c_int32]
_dll.openmc_tally_set_id.restype = c_int
_dll.openmc_tally_set_id.errcheck = _error_handler
_dll.openmc_tally_set_multiply_density.argtypes = [c_int32, c_bool]
_dll.openmc_tally_set_multiply_density.restype = c_int
_dll.openmc_tally_set_multiply_density.errcheck = _error_handler
_dll.openmc_tally_set_nuclides.argtypes = [c_int32, c_int, POINTER(c_char_p)]
_dll.openmc_tally_set_nuclides.restype = c_int
_dll.openmc_tally_set_nuclides.errcheck = _error_handler
_dll.openmc_tally_set_scores.argtypes = [c_int32, c_int, POINTER(c_char_p)]
_dll.openmc_tally_set_scores.restype = c_int
_dll.openmc_tally_set_scores.errcheck = _error_handler
_dll.openmc_tally_set_tt_eps.argtypes = [c_int32, c_double]
_dll.openmc_tally_set_tt_eps.restype = c_int
_dll.openmc_tally_set_tt_eps.errcheck = _error_handler
_dll.openmc_tally_set_type.argtypes = [c_int32, c_char_p]
_dll.openmc_tally_set_type.restype = c_int
_dll.openmc_tally_set_type.errcheck = _error_handler
_dll.openmc_tally_set_writable.argtypes = [c_int32, c_bool]
_dll.openmc_tally_set_writable.restype = c_int
_dll.openmc_tally_set_writable.errcheck = _error_handler
_dll.openmc_remove_tally.argtypes = [c_int32]
_dll.openmc_remove_tally.restype = c_int
_dll.openmc_remove_tally.errcheck = _error_handler
_dll.tallies_size.restype = c_size_t


_SCORES = {
    -1: 'flux', -2: 'total', -3: 'scatter', -4: 'nu-scatter',
    -5: 'absorption', -6: 'fission', -7: 'nu-fission', -8: 'kappa-fission',
    -9: 'current', -10: 'events', -11: 'delayed-nu-fission',
    -12: 'prompt-nu-fission', -13: 'inverse-velocity', -14: 'fission-q-prompt',
    -15: 'fission-q-recoverable', -16: 'decay-rate', -17: 'pulse-height',
    -18: 'ifp-time-numerator', -19: 'ifp-beta-numerator',
    -20: 'ifp-denominator',
}
_ESTIMATORS = {
    0: 'analog', 1: 'tracklength', 2: 'collision'
}
_TALLY_TYPES = {
    0: 'volume', 1: 'mesh-surface', 2: 'surface', 3: 'pulse-height'
}


def global_tallies():
    """Mean and standard deviation of the mean for each global tally.

    Returns
    -------
    list of tuple
        For each global tally, a tuple of (mean, standard deviation)

    """
    ptr = POINTER(c_double)()
    _dll.openmc_global_tallies(ptr)
    array = as_array(ptr, (4, 3))

    # Get sum, sum-of-squares, and number of realizations
    sum_ = array[:, 1]
    sum_sq = array[:, 2]
    n = num_realizations()

    # Determine mean
    if n > 0:
        mean = sum_ / n
    else:
        mean = sum_.copy()

    # Determine standard deviation
    nonzero = np.abs(mean) > 0
    stdev = np.empty_like(mean)
    stdev.fill(np.inf)
    if n > 1:
        stdev[nonzero] = np.sqrt((sum_sq[nonzero]/n - mean[nonzero]**2)/(n - 1))

    return list(zip(mean, stdev))


def num_realizations():
    """Number of realizations of global tallies."""
    return c_int32.in_dll(_dll, 'n_realizations').value


class Tally(_FortranObjectWithID):
    """Tally stored internally.

    This class exposes a tally that is stored internally in the OpenMC
    library. To obtain a view of a tally with a given ID, use the
    :data:`openmc.lib.tallies` mapping.

    Parameters
    ----------
    uid : int or None
        Unique ID of the tally
    new : bool
        When `index` is None, this argument controls whether a new object is
        created or a view of an existing object is returned.
    index : int or None
        Index in the `tallies` array.

    Attributes
    ----------
    id : int
        ID of the tally
    estimator: str
        Estimator type of tally (analog, tracklength, collision)
    filters : list
        List of tally filters
    mean : numpy.ndarray
        An array containing the sample mean for each bin
    multiply_density : bool
        Whether reaction rates should be multiplied by atom density

        .. versionadded:: 0.14.0
    nuclides : list of str
        List of nuclides to score results for
    num_realizations : int
        Number of realizations
    results : numpy.ndarray
        Array of tally results
    std_dev : numpy.ndarray
        An array containing the sample standard deviation for each bin
    type : str
        Type of tally (volume, mesh_surface, surface)

    """
    __instances = WeakValueDictionary()

    def __new__(cls, uid=None, new=True, index=None):
        mapping = tallies
        if index is None:
            if new:
                # Determine ID to assign
                if uid is None:
                    uid = max(mapping, default=0) + 1
                else:
                    if uid in mapping:
                        raise AllocationError('A tally with ID={} has already '
                                              'been allocated.'.format(uid))

                index = c_int32()
                _dll.openmc_extend_tallies(1, index, None)
                index = index.value
            else:
                index = mapping[uid]._index

        if index not in cls.__instances:
            instance = super().__new__(cls)
            instance._index = index
            if uid is not None:
                instance.id = uid
            cls.__instances[index] = instance

        return cls.__instances[index]

    @property
    def active(self):
        active = c_bool()
        _dll.openmc_tally_get_active(self._index, active)
        return active.value

    @active.setter
    def active(self, active):
        _dll.openmc_tally_set_active(self._index, active)

    @property
    def type(self):
        type = c_int32()
        _dll.openmc_tally_get_type(self._index, type)
        return _TALLY_TYPES[type.value]

    @type.setter
    def type(self, type):
        _dll.openmc_tally_set_type(self._index, type.encode())

    @property
    def estimator(self):
        estimator = c_int()
        _dll.openmc_tally_get_estimator(self._index, estimator)
        return _ESTIMATORS[estimator.value]

    @estimator.setter
    def estimator(self, estimator):
        _dll.openmc_tally_set_estimator(self._index, estimator.encode())

    @property
    def id(self):
        tally_id = c_int32()
        _dll.openmc_tally_get_id(self._index, tally_id)
        return tally_id.value

    @id.setter
    def id(self, tally_id):
        _dll.openmc_tally_set_id(self._index, tally_id)

    @property
    def filters(self):
        filt_idx = POINTER(c_int32)()
        n = c_size_t()
        _dll.openmc_tally_get_filters(self._index, filt_idx, n)
        return [_get_filter(filt_idx[i]) for i in range(n.value)]

    @filters.setter
    def filters(self, filters):
        # Get filter indices as int32_t[]
        n = len(filters)
        indices = (c_int32*n)(*(f._index for f in filters))

        _dll.openmc_tally_set_filters(self._index, n, indices)

    def find_filter(self, filter_type):
        """
        Returns the first instance of a filter matching the specified type

        Parameters
        ----------
        filter_type : subclass of openmc.lib.Filter
            The filter type to match when retrieving a filter instance

        Returns
        -------
        filter : openmc.lib.Filter
            The filter instance matching the input filter type

        Raises
        ------
        ValueError if a filter instance matching the input filter type cannot be found.
        """
        for filter in self.filters:
            if isinstance(filter, filter_type):
                return filter

        raise ValueError(f'No filter of type {filter_type} on tally {self.id}')

    @property
    def mean(self):
        n = self.num_realizations
        sum_ = self.results[:, :, 1]
        if n > 0:
            return sum_ / n
        else:
            return sum_.copy()

    @property
    def nuclides(self):
        nucs = POINTER(c_int)()
        n = c_int()
        _dll.openmc_tally_get_nuclides(self._index, nucs, n)
        return [Nuclide(nucs[i]).name if nucs[i] >= 0 else 'total'
                for i in range(n.value)]

    @nuclides.setter
    def nuclides(self, nuclides):
        nucs = (c_char_p * len(nuclides))()
        nucs[:] = [x.encode() for x in nuclides]
        _dll.openmc_tally_set_nuclides(self._index, len(nuclides), nucs)

    @property
    def num_realizations(self):
        n = c_int32()
        _dll.openmc_tally_get_n_realizations(self._index, n)
        return n.value

    @property
    def results(self):
        data = POINTER(c_double)()
        shape = (c_size_t*3)()
        _dll.openmc_tally_results(self._index, data, shape)
        return as_array(data, tuple(shape))

    @property
    def uses_tt(self):
        use_tt = c_bool()
        _dll.openmc_tally_get_use_tt(self._index, use_tt)
        return use_tt.value

    @property
    def tt_shape(self):
        return self.tt_sum.shape

    def _tt(self, which):
        n_cores = c_int()
        _dll.openmc_tally_get_tt_n_cores(self._index, which, n_cores)
        cores = []
        for i in range(n_cores.value):
            data = POINTER(c_double)()
            shape = (c_int*3)()
            _dll.openmc_tally_get_tt_core(self._index, which, i, data, shape)
            core_shape = tuple(shape)
            cores.append(as_array(data, core_shape))
        return TT(cores)

    @property
    def tt_sum(self):
        return self._tt(0)

    @property
    def tt_sum_sq(self):
        return self._tt(1)

    @property
    def tt_ranks(self):
        return {
            'sum': self.tt_sum.ranks,
            'sum_sq': self.tt_sum_sq.ranks,
        }

    def tt_storage_report(self):
        if not self.uses_tt:
            raise RuntimeError(f'Tally ID="{self.id}" is not stored in tensor-train format.')

        tt_sum = self.tt_sum
        tt_sum_sq = self.tt_sum_sq
        dense_bytes = 2 * int(np.prod(tt_sum.shape)) * np.dtype(np.float64).itemsize
        tt_bytes = (
            sum(core.size for core in tt_sum.cores) +
            sum(core.size for core in tt_sum_sq.cores)
        ) * np.dtype(np.float64).itemsize

        return {
            'shape': tt_sum.shape,
            'dense_accumulated_bytes': dense_bytes,
            'tt_accumulated_bytes': tt_bytes,
            'compression_ratio': dense_bytes / tt_bytes if tt_bytes > 0 else np.inf,
            'tt_sum_ranks': tt_sum.ranks,
            'tt_sum_sq_ranks': tt_sum_sq.ranks,
        }

    @staticmethod
    def _flat_to_tt_index(flat_index, tt_shape):
        index = []
        for n in reversed(tt_shape):
            index.append(flat_index % n)
            flat_index //= n
        return tuple(reversed(index))

    def get_tt_value(self, filter_index, nuclide_index, score_index):
        if not self.uses_tt:
            raise RuntimeError(f'Tally ID="{self.id}" is not stored in tensor-train format.')

        n_scores = len(self.scores)
        n_nuclides = len(self.nuclides)
        dense_score_index = score_index + nuclide_index * n_scores
        flat_index = filter_index * (n_scores * n_nuclides) + dense_score_index

        n = self.num_realizations
        # self.tt_sum loads TT core views, so keep one local reference for both
        # shape lookup and value reconstruction.
        tt_sum = self.tt_sum
        tt_index = self._flat_to_tt_index(flat_index, tt_sum.shape)
        sum_ = tt_sum.at(tt_index)
        return sum_ / n if n > 0 else sum_

    def get_tt_values(self, filter_indices, nuclide_indices, score_indices):
        if not self.uses_tt:
            raise RuntimeError(f'Tally ID="{self.id}" is not stored in tensor-train format.')

        filter_indices = list(filter_indices)
        nuclide_indices = list(nuclide_indices)
        score_indices = list(score_indices)
        n_scores = len(self.scores)
        n_nuclides = len(self.nuclides)
        tt_sum = self.tt_sum
        tt_shape = tt_sum.shape
        n = self.num_realizations

        data = np.empty(
            (len(filter_indices), len(nuclide_indices), len(score_indices)))
        for i, filter_index in enumerate(filter_indices):
            for j, nuclide_index in enumerate(nuclide_indices):
                for k, score_index in enumerate(score_indices):
                    dense_score_index = score_index + nuclide_index * n_scores
                    flat_index = (
                        filter_index * (n_scores * n_nuclides) +
                        dense_score_index
                    )
                    tt_index = self._flat_to_tt_index(flat_index, tt_shape)
                    sum_ = tt_sum.at(tt_index)
                    data[i, j, k] = sum_ / n if n > 0 else sum_
        return data

    def set_tt_eps(self, eps):
        _dll.openmc_tally_set_tt_eps(self._index, eps)

    @property
    def scores(self):
        scores_as_int = POINTER(c_int)()
        n = c_int()
        try:
            _dll.openmc_tally_get_scores(self._index, scores_as_int, n)
        except AllocationError:
            return []
        else:
            scores = []
            for i in range(n.value):
                if scores_as_int[i] in _SCORES:
                    scores.append(_SCORES[scores_as_int[i]])
                elif scores_as_int[i] in REACTION_NAME:
                    scores.append(REACTION_NAME[scores_as_int[i]])
                else:
                    scores.append(str(scores_as_int[i]))
            return scores

    @scores.setter
    def scores(self, scores):
        scores_ = (c_char_p * len(scores))()
        scores_[:] = [x.encode() for x in scores]
        _dll.openmc_tally_set_scores(self._index, len(scores), scores_)

    @property
    def std_dev(self):
        results = self.results
        std_dev = np.empty(results.shape[:2])
        std_dev.fill(np.inf)

        n = self.num_realizations
        if n > 1:
            # Get sum and sum-of-squares from results
            sum_ = results[:, :, 1]
            sum_sq = results[:, :, 2]

            # Determine non-zero entries
            mean = sum_ / n
            nonzero = np.abs(mean) > 0

            # Calculate sample standard deviation of the mean
            std_dev[nonzero] = np.sqrt(
                (sum_sq[nonzero]/n - mean[nonzero]**2)/(n - 1))

        return std_dev

    @property
    def writable(self):
        writable = c_bool()
        _dll.openmc_tally_get_writable(self._index, writable)
        return writable.value

    @writable.setter
    def writable(self, writable):
        _dll.openmc_tally_set_writable(self._index, writable)

    @property
    def multiply_density(self):
        multiply_density = c_bool()
        _dll.openmc_tally_get_multiply_density(self._index, multiply_density)
        return multiply_density.value

    @multiply_density.setter
    def multiply_density(self, multiply_density):
        _dll.openmc_tally_set_multiply_density(self._index, multiply_density)

    def reset(self):
        """Reset results and num_realizations of tally"""
        _dll.openmc_tally_reset(self._index)

    def ci_width(self, alpha=0.05):
        """Confidence interval half-width based on a Student t distribution

        Parameters
        ----------
        alpha : float
            Significance level (one minus the confidence level!)

        Returns
        -------
        float
            Half-width of a two-sided (1 - :math:`alpha`) confidence interval

        """
        half_width = self.std_dev.copy()
        n = self.num_realizations
        if n > 1:
            half_width *= scipy.stats.t.ppf(1 - alpha/2, n - 1)
        return half_width


class _TallyMapping(Mapping):
    def __getitem__(self, key):
        index = c_int32()
        try:
            _dll.openmc_get_tally_index(key, index)
        except (AllocationError, InvalidIDError) as e:
            # __contains__ expects a KeyError to work correctly
            raise KeyError(str(e))
        return Tally(index=index.value)

    def __iter__(self):
        for i in range(len(self)):
            yield Tally(index=i).id

    def __len__(self):
        return _dll.tallies_size()

    def __repr__(self):
        return repr(dict(self))

    def __delitem__(self, key):
        """Delete a tally from tally vector and remove the ID,index pair from tally"""
        _dll.openmc_remove_tally(self[key]._index)

tallies = _TallyMapping()
