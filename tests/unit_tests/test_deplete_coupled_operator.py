"""Basic unit tests for openmc.deplete.CoupledOperator instantiation

"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openmc.deplete import AtomNumber, CoupledOperator, Chain, ReactionRates
from openmc.deplete.tt_depletion import (
    _get_tt_reaction_rates, _prepare_tt_reaction_rates)
import openmc
import numpy as np

CHAIN_PATH = Path(__file__).parents[1] / "chain_simple.xml"


@pytest.fixture(scope="module")
def model():
    fuel = openmc.Material(name="uo2")
    fuel.add_element("U", 1, percent_type="ao", enrichment=4.25)
    fuel.add_element("O", 2)
    fuel.set_density("g/cc", 10.4)

    clad = openmc.Material(name="clad")
    clad.add_element("Zr", 1)
    clad.set_density("g/cc", 6)

    water = openmc.Material(name="water")
    water.add_element("O", 1)
    water.add_element("H", 2)
    water.set_density("g/cc", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")

    radii = [0.42, 0.45]
    fuel.volume = np.pi * radii[0] ** 2
    clad.volume = np.pi * (radii[1]**2 - radii[0]**2)
    water.volume = 1.24**2 - (np.pi * radii[1]**2)

    materials = openmc.Materials([fuel, clad, water])

    pin_surfaces = [openmc.ZCylinder(r=r) for r in radii]
    pin_univ = openmc.model.pin(pin_surfaces, materials)
    bound_box = openmc.model.RectangularPrism(
        1.24, 1.24, boundary_type="reflective")
    root_cell = openmc.Cell(fill=pin_univ, region=-bound_box)
    geometry = openmc.Geometry([root_cell])

    settings = openmc.Settings()
    settings.particles = 1000
    settings.inactive = 10
    settings.batches = 50

    return openmc.Model(geometry, materials, settings)


@pytest.fixture()
def model_with_volumes():
    mat1 = openmc.Material()
    mat1.add_element("Ag", 1, percent_type="ao")
    mat1.set_density("g/cm3", 10.49)
    mat1.depletable = True
    mat1.volume = 102

    mat2 = openmc.Material()
    mat2.add_element("Ag", 1, percent_type="ao")
    mat2.set_density("g/cm3", 10.49)

    sph1 = openmc.Sphere(r=1.0)
    sph2 = openmc.Sphere(r=2.0, x0=3)
    sph3 = openmc.Sphere(r=5.0, boundary_type="vacuum")

    cell1 = openmc.Cell(region=-sph1, fill=mat1)
    cell1.volume = 4.19
    cell2 = openmc.Cell(region=-sph2, fill=mat1)
    cell2.volume = 33.51
    cell3 = openmc.Cell(region=-sph3 & +sph1 & +sph2, fill=mat2)
    cell3.volume = 485.9

    geometry = openmc.Geometry([cell1, cell2, cell3])

    return openmc.Model(geometry)


def test_operator_init(model):
    """The test uses a temporary dummy chain. This file will be removed
    at the end of the test, and only contains a depletion_chain node."""

    CoupledOperator(model, CHAIN_PATH)


def test_tt_material_rate_helpers():
    """TT helper path normalizes one material slice at a time."""

    class FakeRateHelper:
        nuclides = ['U235', 'U238']

        def __init__(self, rates):
            self.rates = rates
            self.reset = False

        def reset_tally_means(self):
            self.reset = True

        def get_material_rates(self, mat_index, nuc_ind, rx_ind):
            assert nuc_ind == [0, 1]
            assert rx_ind == [0, 1]
            return self.rates[mat_index]

    class FakeNormalizationHelper:
        def __init__(self):
            self.updates = []

        def reset(self):
            self.energy = 0.0
            self.updates.clear()

        def update(self, fission_rates):
            self.updates.append(fission_rates.copy())
            self.energy += fission_rates.sum()

        def factor(self, source_rate):
            return source_rate / self.energy

    class FakeYieldHelper:
        def unpack(self):
            self.unpacked = True

        def weighted_yields(self, local_mat_index):
            return {'mat': local_mat_index}

    class FakeReaction:
        def __init__(self, reaction_type):
            self.type = reaction_type

    class FakeNuclide:
        def __init__(self, name, reactions):
            self.name = name
            self.reactions = [FakeReaction(r) for r in reactions]

    class FakeChain:
        reactions = ['fission', 'capture']
        nuclides = [
            FakeNuclide('U235', ['fission', 'capture']),
            FakeNuclide('U238', ['fission']),
            FakeNuclide('Xe135', []),
        ]

    tally_rates = {
        0: np.array([[4.0, 5.0], [1.0, 7.0], [0.0, 0.0]]),
        1: np.array([[2.0, 11.0], [3.0, 13.0], [0.0, 0.0]])
    }
    op = object.__new__(CoupledOperator)
    op.local_mats = ['1', '2']
    op._mat_index_map = {'1': 0, '2': 1}
    op.chain = FakeChain()
    op.reaction_rates = ReactionRates(
        op.local_mats, ['U235', 'U238', 'Xe135'], FakeChain.reactions)
    op.number = AtomNumber(
        op.local_mats, ['U235', 'U238', 'Xe135'],
        {'1': 2.0, '2': 4.0}, 3)
    op.number['1', 'U235'] = 2.0e24
    op.number['1', 'U238'] = 6.0e24
    op.number['2', 'U235'] = 4.0e24
    op.number['2', 'U238'] = 4.0e24
    op._rate_helper = FakeRateHelper(tally_rates)
    op._normalization_helper = FakeNormalizationHelper()
    op._yield_helper = FakeYieldHelper()

    normalization_factor = _prepare_tt_reaction_rates(op, 24.0)

    assert op._rate_helper.reset
    assert op._yield_helper.unpacked
    assert normalization_factor == pytest.approx(2.0)
    assert op.chain.fission_yields == [{'mat': 0}, {'mat': 1}]
    np.testing.assert_array_equal(
        op._tt_reaction_rate_mask,
        [[True, True], [True, False], [False, False]])
    np.testing.assert_allclose(
        op._normalization_helper.updates[0], [4.0, 3.0, 0.0])
    np.testing.assert_allclose(
        op._normalization_helper.updates[1], [2.0, 3.0, 0.0])

    material_rates = _get_tt_reaction_rates(op, '1', normalization_factor)

    assert material_rates.shape == (3, 2)
    assert material_rates.index_nuc == op.reaction_rates.index_nuc
    assert material_rates.index_rx == op.reaction_rates.index_rx
    expected_rates = tally_rates[0].copy()
    expected_rates[1, 1] = 0.0
    np.testing.assert_allclose(
        material_rates, expected_rates * normalization_factor / 2.0e24)


def test_tt_operator_keeps_dense_atom_number(model):
    """CoupledOperator keeps dense AtomNumber when TT rates are enabled."""

    op = CoupledOperator(model, CHAIN_PATH, tt_eps=1.0e-3)

    assert isinstance(op.number, AtomNumber)
    assert not hasattr(op, 'tt_number')


@pytest.mark.parametrize("fission_yield_mode", ["constant", "cutoff", "average"])
def test_tt_operator_allows_fission_yield_modes(model, fission_yield_mode):
    """TT reaction-rate mode works with all fission-yield helper modes."""

    op = CoupledOperator(
        model, CHAIN_PATH, tt_eps=1.0e-3,
        fission_yield_mode=fission_yield_mode)

    assert isinstance(op.number, AtomNumber)


def test_tt_operator_returns_rates_context(monkeypatch):
    """TT operator result keeps normalization data instead of dense rates."""

    op = object.__new__(CoupledOperator)
    op._n_calls = 0
    op._tt_depletion_used = True
    op._update_materials_and_nuclides = MagicMock()
    op._get_reaction_nuclides = lambda: ['U235']
    op._rate_helper = MagicMock()
    op._normalization_helper = MagicMock()
    op._yield_helper = MagicMock()
    op._print_tt_storage_reports = lambda: None

    def fail_dense_path(source_rate):
        raise AssertionError("dense reaction-rate path should not be used")

    op._calculate_reaction_rates = fail_dense_path
    monkeypatch.setattr(
        'openmc.deplete.coupled_operator._prepare_tt_reaction_rates',
        lambda operator, source_rate: 3.0)
    monkeypatch.setattr(openmc.lib, "reset", lambda: None)
    monkeypatch.setattr(openmc.lib, "run", lambda: None)
    monkeypatch.setattr(openmc.lib, "keff", lambda: (1.0, 0.0))

    vec = [np.array([1.0])]
    result = CoupledOperator.__call__(op, vec, 10.0)

    op._update_materials_and_nuclides.assert_called_once_with(vec)
    assert getattr(result.rates, '_tt_reaction_rates', False) is True
    assert result.rates.normalization_factor == 3.0


def test_tt_operator_call_updates_from_dense_atom_numbers(monkeypatch):
    """CoupledOperator.__call__ updates transport from dense atom numbers."""

    op = object.__new__(CoupledOperator)
    op._n_calls = 0
    op._tt_depletion_used = True
    op._update_materials_and_nuclides = MagicMock()
    op._get_reaction_nuclides = lambda: ['U235']
    op._rate_helper = MagicMock()
    op._normalization_helper = MagicMock()
    op._yield_helper = MagicMock()
    op._print_tt_storage_reports = lambda: None

    monkeypatch.setattr(
        'openmc.deplete.coupled_operator._prepare_tt_reaction_rates',
        lambda operator, source_rate: 3.0)
    monkeypatch.setattr(openmc.lib, "reset", lambda: None)
    monkeypatch.setattr(openmc.lib, "run", lambda: None)
    monkeypatch.setattr(openmc.lib, "keff", lambda: (1.0, 0.0))

    vec = [np.array([1.0])]
    CoupledOperator.__call__(op, vec, 10.0)

    op._update_materials_and_nuclides.assert_called_once_with(vec)


def test_tt_operator_zero_source_returns_rates_context(monkeypatch):
    """TT zero-source result avoids dense zero reaction rates."""

    op = object.__new__(CoupledOperator)
    op._tt_depletion_used = True
    op._n_calls = 0
    op._update_materials_and_nuclides = MagicMock()
    op._get_reaction_nuclides = lambda: ['U235']
    op._rate_helper = MagicMock()
    op._normalization_helper = MagicMock()
    op._yield_helper = MagicMock()
    monkeypatch.setattr(openmc.lib, "reset", lambda: None)

    vec = [np.array([1.0])]
    result = CoupledOperator.__call__(op, vec, 0.0)

    op._update_materials_and_nuclides.assert_called_once_with(vec)
    assert getattr(result.rates, '_tt_reaction_rates', False) is True
    assert result.rates.zero_source
    assert result.rates.normalization_factor == 0.0


def test_diff_volume_method_match_cell(model_with_volumes):
    """Tests the volumes assigned to the materials match the cell volumes"""

    operator = openmc.deplete.CoupledOperator(
        model=model_with_volumes,
        diff_burnable_mats=True,
        diff_volume_method='match cell',
        chain_file=CHAIN_PATH
    )

    all_cells = list(operator.model.geometry.get_all_cells().values())
    assert all_cells[0].fill.volume == 4.19
    assert all_cells[1].fill.volume == 33.51
    # mat2 is not depletable
    assert all_cells[2].fill.volume is None


def test_diff_volume_method_divide_equally(model_with_volumes):
    """Tests the volumes assigned to the materials are divided equally"""

    chain = Chain.from_xml(CHAIN_PATH)

    operator = openmc.deplete.CoupledOperator(
        model=model_with_volumes,
        diff_burnable_mats=True,
        diff_volume_method='divide equally',
        chain_file=chain
    )

    all_cells = list(operator.model.geometry.get_all_cells().values())
    assert all_cells[0].fill.volume == 51
    assert all_cells[1].fill.volume == 51
    # mat2 is not depletable
    assert all_cells[2].fill.volume is None
