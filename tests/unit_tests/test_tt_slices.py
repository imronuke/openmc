"""Numerical tests for material slices read from TT tally channels."""

from collections import Counter
from itertools import product

import numpy as np
import pytest

from openmc.lib import Tally
from openmc.tt import TT


class _FakeTally:
    def __init__(self, channels, n_nuclides, n_scores, realizations,
                 tt_shape=None):
        self.calls = Counter()
        if tt_shape is None:
            tt_shape = channels[0].shape
        self.metadata = {
            'uses_tt': True,
            'id': 17,
            'scores': [f'score-{i}' for i in range(n_scores)],
            'nuclides': [f'nuclide-{i}' for i in range(n_nuclides)],
            'num_realizations': realizations,
            'tt_shape': tt_shape,
            'tt_sum': tuple(channels),
        }

    def __getattr__(self, name):
        if name not in self.metadata:
            raise AttributeError(name)
        self.calls[name] += 1
        return self.metadata[name]

    def _get_tt_slice_reader(self):
        return Tally._get_tt_slice_reader(self)


def _make_cores(shape, seed=192):
    rng = np.random.default_rng(seed)
    ranks = [1] + [2 + i % 2 for i in range(len(shape) - 1)] + [1]
    return [rng.uniform(-1.0, 2.0, size=(ranks[i], n, ranks[i + 1]))
            for i, n in enumerate(shape)]


def _make_channels(shape, n_channels):
    return tuple(TT(_make_cores(shape, seed=192 + i))
                 for i in range(n_channels))


def _dense_tensor(cores):
    """Evaluate every entry by explicitly summing internal rank indices."""
    shape = tuple(core.shape[1] for core in cores)
    rank_ranges = [range(core.shape[2]) for core in cores[:-1]]
    dense = np.zeros(shape)
    for indices in np.ndindex(shape):
        for ranks in product(*rank_ranges):
            bonds = (0,) + ranks + (0,)
            contribution = 1.0
            for k, core in enumerate(cores):
                contribution *= core[bonds[k], indices[k], bonds[k + 1]]
            dense[indices] += contribution
    return dense


@pytest.mark.parametrize('shape,n_nuclides,n_scores', [
    ((2, 3, 3, 4), 3, 4),
    ((2, 3, 2, 6), 3, 4),
    ((2, 2, 2, 3), 2, 6),
    ((3, 4), 3, 4),
    ((12,), 3, 4),
    ((2, 3, 1), 3, 1),
    ((2, 3), 3, 1),
    ((2, 1, 4), 1, 4),
    ((1, 4), 1, 4),
    ((2, 3, 1, 1), 1, 1),
    ((5,), 1, 1),
    ((1,), 1, 1),
])
@pytest.mark.parametrize('realizations', [0, 7])
def test_tt_slice_reader_matches_dense(shape, n_nuclides, n_scores,
                                       realizations):
    n_filters = int(np.prod(shape))
    n_channels = n_nuclides * n_scores
    channels = _make_channels(shape, n_channels)
    originals = [[core.copy() for core in tt.cores] for tt in channels]
    for tt in channels:
        for core in tt.cores:
            core.flags.writeable = False

    tally = _FakeTally(channels, n_nuclides, n_scores, realizations)
    dense_channels = np.stack(
        [_dense_tensor(tt.cores).ravel() for tt in channels], axis=1)
    expected = dense_channels.reshape(n_filters, n_nuclides, n_scores)
    if realizations:
        expected /= realizations

    reader = Tally._get_tt_slice_reader(tally)
    for filter_index in reversed(range(n_filters)):
        for score_index in reversed(range(n_scores)):
            selected = reader.get_slice(filter_index, score_index)
            assert selected.shape == (n_nuclides,)
            np.testing.assert_allclose(
                selected, expected[filter_index, :, score_index],
                rtol=2e-12, atol=2e-12)
        actual = reader.get_slice(filter_index)
        assert actual.shape == (n_nuclides, n_scores)
        np.testing.assert_allclose(actual, expected[filter_index],
                                   rtol=2e-12, atol=2e-12)

    reader.get_slice(0)
    reader.get_slice(0, 0)
    for name in ('uses_tt', 'tt_shape', 'scores', 'nuclides',
                 'num_realizations', 'tt_sum'):
        assert tally.calls[name] == 1
    for tt, channel_cores in zip(channels, originals):
        for core, original in zip(tt.cores, channel_cores):
            np.testing.assert_array_equal(core, original)


def test_tt_slice_reader_supports_zero_channels():
    shape = (2, 3)
    channels = list(_make_channels(shape, 2))
    channels[0] = TT([], shape=shape)
    tally = _FakeTally(channels, 1, 2, 3)

    actual = Tally._get_tt_slice_reader(tally).get_slice(4)
    np.testing.assert_array_equal(actual[0, 0], 0.0)
    assert actual[0, 1] != 0.0


@pytest.mark.parametrize('filter_index', [-1, 48])
@pytest.mark.parametrize('score_index', [None, 0])
def test_tt_slice_reader_rejects_filter_indices(filter_index, score_index):
    tally = _FakeTally(_make_channels((2, 3, 2, 4), 8), 2, 4, 3)
    reader = Tally._get_tt_slice_reader(tally)
    with pytest.raises(IndexError, match='filter'):
        reader.get_slice(filter_index, score_index)


@pytest.mark.parametrize('score_index', [-1, 4])
def test_tt_slice_reader_rejects_score_indices(score_index):
    tally = _FakeTally(_make_channels((2, 3, 2, 4), 8), 2, 4, 3)
    reader = Tally._get_tt_slice_reader(tally)
    with pytest.raises(IndexError, match='score'):
        reader.get_slice(0, score_index)


def test_tt_slice_reader_requires_tt_tally():
    tally = _FakeTally(_make_channels((2, 3), 2), 1, 2, 3)
    tally.metadata['uses_tt'] = False
    with pytest.raises(RuntimeError, match='tensor-train'):
        Tally._get_tt_slice_reader(tally)


def test_tt_slice_reader_rejects_incompatible_shape():
    tally = _FakeTally(_make_channels((2, 5), 6), 2, 3, 3,
                        tt_shape=(2, 4))
    with pytest.raises(RuntimeError, match='shape'):
        Tally._get_tt_slice_reader(tally)


def test_tt_slice_reader_refreshes_tally_state_between_cycles():
    shape = (2, 3, 4)
    channels = _make_channels(shape, 12)
    tally = _FakeTally(channels, 3, 4, 2)
    first = Tally._get_tt_slice_reader(tally)
    expected = np.stack(
        [_dense_tensor(tt.cores).ravel() for tt in channels], axis=1)
    expected = expected.reshape(2, 3, 4, 3, 4)
    np.testing.assert_allclose(first.get_slice(0), expected[0, 0, 0] / 2)

    updated = []
    for tt in channels:
        cores = [core.copy() for core in tt.cores]
        cores[0] *= 3
        updated.append(TT(cores))
    tally.metadata['tt_sum'] = tuple(updated)
    tally.metadata['num_realizations'] = 5
    second = Tally._get_tt_slice_reader(tally)
    np.testing.assert_allclose(
        second.get_slice(0), expected[0, 0, 0] * 3 / 5)
    np.testing.assert_allclose(
        second.get_slice(1, 2), expected[0, 0, 1, :, 2] * 3 / 5)


def test_tt_slice_public_api_matches_reader():
    channels = _make_channels((2, 3), 12)
    tally = _FakeTally(channels, 3, 4, 2)
    expected = np.stack(
        [_dense_tensor(tt.cores).ravel() for tt in channels], axis=1)
    expected = expected.reshape(2, 3, 3, 4) / 2
    np.testing.assert_allclose(Tally.get_tt_slice(tally, 1), expected[0, 1])
    reader = Tally._get_tt_slice_reader(tally)
    np.testing.assert_allclose(reader.get_slice(1, 2), expected[0, 1, :, 2])
