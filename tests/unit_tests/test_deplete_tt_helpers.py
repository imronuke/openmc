"""Test per-channel TT rate extraction and transport-cycle caches."""

from collections import Counter
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

import openmc.lib
from openmc.deplete import helpers
from openmc.deplete.helpers import DirectReactionRateHelper
from openmc.lib import Tally
from openmc.tt import TT


def _channel_tts(values):
    """Create one rank-one spatial TT for every nuclide-score channel."""
    n_materials, n_nuclides, n_scores = values.shape
    return tuple(
        TT([values[:, i_nuclide, i_score].reshape(1, n_materials, 1)])
        for i_nuclide in range(n_nuclides)
        for i_score in range(n_scores))


class _FakeTally:
    def __init__(self, values, realizations=2, uses_tt=True):
        n_materials, n_nuclides, n_scores = values.shape
        self.calls = Counter()
        self.before_nuclides_set = None
        self.reader_ref = None
        self.metadata = {
            'uses_tt': uses_tt,
            'id': 17,
            'scores': [f'score-{i}' for i in range(n_scores)],
            'nuclides': [f'nuclide-{i}' for i in range(n_nuclides)],
            'num_realizations': realizations,
            'tt_shape': (n_materials,),
            'tt_sum': _channel_tts(values),
            'mean': values.reshape(
                n_materials, n_nuclides * n_scores) / realizations,
        }

    def __getattr__(self, name):
        metadata = self.__dict__.get('metadata', {})
        if name not in metadata:
            raise AttributeError(name)
        self.calls[name] += 1
        return metadata[name]

    def _tt(self, which, channel):
        self.calls['tt_channel'] += 1
        return self.metadata['tt_sum'][channel]

    def __setattr__(self, name, value):
        metadata = self.__dict__.get('metadata', {})
        if name in metadata:
            if name == 'nuclides' and self.before_nuclides_set is not None:
                self.before_nuclides_set()
            metadata[name] = value
        else:
            object.__setattr__(self, name, value)

    def _get_tt_slice_reader(self):
        self.calls['reader'] += 1
        reader = Tally._get_tt_slice_reader(self)
        self.reader_ref = weakref.ref(reader)
        return reader


@pytest.fixture
def rate_helper(monkeypatch):
    monkeypatch.setattr(openmc.lib, 'settings', SimpleNamespace())
    return DirectReactionRateHelper(5, 4)


@pytest.fixture
def tally_values():
    return np.array([[[10., 20.], [30., 40.], [50., 60.]],
                     [[11., 21.], [31., 41.], [51., 61.]]])


def _expected_rates(values, material, realizations, nuc_indices, rx_indices):
    expected = np.zeros((5, 4))
    material_values = values[material] / realizations
    for tally_nuc, nuc_index in enumerate(nuc_indices):
        for tally_rx, rx_index in enumerate(rx_indices):
            expected[nuc_index, rx_index] = material_values[tally_nuc, tally_rx]
    return expected


def test_tt_helper_reorders_and_scatter_fills_rates(rate_helper, tally_values):
    tally = _FakeTally(tally_values)
    rate_helper._rate_tally = tally
    nuclides = [3, 0, 4]
    reactions = [2, 0]
    expected = _expected_rates(tally_values, 0, 2, nuclides, reactions)

    rates = rate_helper.get_material_rates(0, nuclides, reactions)
    np.testing.assert_array_equal(rates, expected)
    reader = tally.reader_ref()
    np.testing.assert_array_equal(
        reader.get_slice(0, 1), tally_values[0, :, 1] / 2)

    for material in (1, 0, 1):
        rates = rate_helper.get_material_rates(material, nuclides, reactions)
        np.testing.assert_array_equal(
            rates, _expected_rates(tally_values, material, 2,
                                   nuclides, reactions))
    assert tally.calls['reader'] == 1
    assert tally.calls['tt_sum'] == 0
    assert tally.calls['tt_channel'] == 6
    assert tally.calls['mean'] == 0


def test_tt_helper_changed_mappings_clear_missing_entries(
        rate_helper, tally_values):
    tally = _FakeTally(tally_values)
    rate_helper._rate_tally = tally
    rate_helper.get_material_rates(0, [3, 0, 4], [2, 0])

    nuc_indices = [1, 2, 0]
    rx_indices = [1, 3]
    actual = rate_helper.get_material_rates(1, nuc_indices, rx_indices)
    expected = _expected_rates(
        tally_values, 1, 2, nuc_indices, rx_indices)
    np.testing.assert_array_equal(actual, expected)


def test_tt_helper_reset_fetches_new_transport_results(
        rate_helper, tally_values):
    tally = _FakeTally(tally_values)
    rate_helper._rate_tally = tally
    old = rate_helper.get_material_rates(0, [3, 0, 4], [2, 0]).copy()
    first_reader = tally.reader_ref
    assert first_reader() is not None
    rate_helper.reset_tally_means()
    assert first_reader() is None

    updated_values = 3 * tally_values
    tally.metadata['tt_sum'] = _channel_tts(updated_values)
    tally.metadata['num_realizations'] = 5
    rates = rate_helper.get_material_rates(0, [3, 0, 4], [2, 0])
    np.testing.assert_allclose(rates, old * 6 / 5)
    assert tally.calls['reader'] == 2
    assert tally.calls['tt_sum'] == 0
    assert tally.calls['tt_channel'] == 12
    assert tally.calls['num_realizations'] == 2


def test_tt_helper_releases_views_before_changing_nuclides(
        rate_helper, tally_values):
    tally = _FakeTally(tally_values)
    rate_helper._rate_tally = tally
    rate_helper.get_material_rates(0, [3, 0, 4], [2, 0])
    old_reader = tally.reader_ref
    updated_values = tally_values[:, :2, :]

    def before_assignment():
        assert old_reader() is None
        tally.metadata['tt_sum'] = _channel_tts(updated_values)

    tally.before_nuclides_set = before_assignment
    rate_helper.nuclides = ['U235', 'Xe135']
    assert rate_helper.nuclides == ['U235', 'Xe135']
    selected = rate_helper.get_material_rates(0, [4, 1], [2, 0])
    expected = _expected_rates(updated_values, 0, 2, [4, 1], [2, 0])
    np.testing.assert_array_equal(selected, expected)
    assert tally.calls['reader'] == 2


def test_tt_helper_releases_views_before_generating_tallies(
        rate_helper, tally_values, monkeypatch):
    previous = _FakeTally(tally_values)
    rate_helper._rate_tally = previous
    rate_helper.get_material_rates(0, [3, 0, 4], [2, 0])
    old_reader = previous.reader_ref
    replacement = _FakeTally(4 * tally_values)

    def make_tally():
        assert old_reader() is None
        return replacement

    monkeypatch.setattr(helpers, 'Tally', make_tally)
    monkeypatch.setattr(helpers, 'MaterialFilter', lambda materials: materials)
    rate_helper.generate_tallies([object(), object()], ['fission', '(n,gamma)'])
    actual = rate_helper.get_material_rates(0, [3, 0, 4], [2, 0])
    expected = _expected_rates(4 * tally_values, 0, 2, [3, 0, 4], [2, 0])
    np.testing.assert_array_equal(actual, expected)
    assert replacement.calls['reader'] == 1


def test_tt_helper_extracts_only_selected_channels(rate_helper, tally_values):
    tally = _FakeTally(tally_values)
    rate_helper._rate_tally = tally

    actual = rate_helper.get_material_rates(
        1, [3, 0, 4], [2, 0], tally_channel_indices=[0, 3])
    expected = np.zeros((5, 4))
    expected[3, 2] = tally_values[1, 0, 0] / 2
    expected[0, 0] = tally_values[1, 1, 1] / 2
    np.testing.assert_array_equal(actual, expected)
    assert tally.calls['tt_channel'] == 2


def test_dense_helper_preserves_values_and_mean_cache(rate_helper, tally_values):
    tally = _FakeTally(tally_values, uses_tt=False)
    rate_helper._rate_tally = tally
    expected = _expected_rates(tally_values, 0, 2, [3, 0, 4], [2, 0])
    rates = rate_helper.get_material_rates(0, [3, 0, 4], [2, 0])
    np.testing.assert_array_equal(rates, expected)
    rate_helper.get_material_rates(1, [3, 0, 4], [2, 0])
    assert tally.calls['mean'] == 1
    assert tally.calls['uses_tt'] == 2
    assert tally.calls['reader'] == 0

    rate_helper.reset_tally_means()
    tally.metadata['mean'] *= 3
    rates = rate_helper.get_material_rates(0, [3, 0, 4], [2, 0])
    np.testing.assert_array_equal(rates, 3 * expected)
    assert tally.calls['mean'] == 2
    assert tally.calls['uses_tt'] == 3
    assert tally.calls['reader'] == 0
