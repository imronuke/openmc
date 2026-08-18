"""Tests for saving results

It is worth noting that openmc.deplete.integrate is extremely complex, to the
point I am unsure if it can be reasonably unit-tested.  For the time being, it
will be left unimplemented and testing will be done via regression.

"""

import copy
from random import uniform
from unittest.mock import MagicMock

import h5py
import numpy as np
from uncertainties import ufloat
import pytest

from openmc.mpi import comm
from openmc.tt import TT
from openmc.deplete import (
    ReactionRates, StepResult, Results, OperatorResult, PredictorIntegrator,
    CECMIntegrator, CF4Integrator, CELIIntegrator, EPCRK4Integrator,
    LEQIIntegrator, SICELIIntegrator, SILEQIIntegrator, TTDepletionRates,
    cram, pool)
from openmc.deplete.tt_depletion import _TTReactionRates

from tests import dummy_operator


INTEGRATORS = [
    PredictorIntegrator,
    CECMIntegrator,
    CF4Integrator,
    CELIIntegrator,
    EPCRK4Integrator,
    LEQIIntegrator,
    SICELIIntegrator,
    SILEQIIntegrator
]


def _assert_dense_atom_numbers(handle, step, block, expected):
    assert 'number' in handle
    expected = np.asarray(expected)
    np.testing.assert_allclose(
        handle['number'][step, block:block + len(expected)], expected)


def test_results_save(run_in_tmpdir):
    """Test data save module"""

    rng = np.random.RandomState(comm.rank)

    # Mock geometry
    op = MagicMock()

    # Avoid DummyOperator thinking it's doing a restart calculation
    op.prev_res = None

    vol_dict = {}
    full_burn_list = []

    for i in range(comm.size):
        vol_dict[str(2*i)] = 1.2
        vol_dict[str(2*i + 1)] = 1.2
        full_burn_list.append(str(2*i))
        full_burn_list.append(str(2*i + 1))

    burn_list = full_burn_list[2*comm.rank: 2*comm.rank + 2]
    nuc_list = ["na", "nb"]

    name_list = {mat: "" for mat in full_burn_list}
    op.get_results_info.return_value = (
        vol_dict, nuc_list, burn_list, full_burn_list, name_list)

    # Construct end-of-step concentrations
    x1 = [rng.random(2), rng.random(2)]
    x2 = [rng.random(2), rng.random(2)]

    # Construct reaction rates
    r1 = ReactionRates(burn_list, ["na", "nb"], ["ra", "rb"])
    r1[:] = rng.random((2, 2, 2))
    rate1 = copy.deepcopy(r1)

    r2 = ReactionRates(burn_list, ["na", "nb"], ["ra", "rb"])
    r2[:] = rng.random((2, 2, 2))
    rate2 = copy.deepcopy(r2)

    # Create global terms (eigenvalue and uncertainty)
    eigvl1 = rng.random(2)
    eigvl2 = rng.random(2)

    eigvl1 = comm.bcast(eigvl1, root=0)
    eigvl2 = comm.bcast(eigvl2, root=0)

    t1 = [0.0, 1.0]
    t2 = [1.0, 2.0]

    op_result1 = OperatorResult(ufloat(*eigvl1), rate1)
    op_result2 = OperatorResult(ufloat(*eigvl2), rate2)

    # saves within a subdirectory
    StepResult.save(
        op,
        x1,
        op_result1,
        t1,
        0,
        0,
        write_rates=True,
        path='out/put/depletion.h5'
    )
    res = Results('out/put/depletion.h5')

    # saves with default filename
    StepResult.save(op, x1, op_result1, t1, 0, 0, write_rates=True)
    StepResult.save(op, x2, op_result2, t2, 0, 1, write_rates=True)

    # Load the files
    res = Results("depletion_results.h5")

    for mat_i, mat in enumerate(burn_list):
        for nuc_i, nuc in enumerate(nuc_list):
            assert res[0][mat, nuc] == x1[mat_i][nuc_i]
            assert res[1][mat, nuc] == x2[mat_i][nuc_i]
    np.testing.assert_array_equal(res[0].rates, rate1)
    np.testing.assert_array_equal(res[1].rates, rate2)

    np.testing.assert_array_equal(res[0].k, eigvl1)
    np.testing.assert_array_equal(res[0].time, t1)

    np.testing.assert_array_equal(res[1].k, eigvl2)
    np.testing.assert_array_equal(res[1].time, t2)


def test_results_save_without_rates(run_in_tmpdir):
    """StepResult.save skips reaction-rate datasets by default"""

    op = MagicMock()
    op.prev_res = None
    vol_dict = {"0": 1.0}
    nuc_list = ["na"]
    burn_list = ["0"]
    name_list = {mat: "" for mat in burn_list}
    op.get_results_info.return_value = (vol_dict, nuc_list, burn_list, burn_list, name_list)

    x = [np.array([1.0])]
    rates = ReactionRates(burn_list, nuc_list, ["ra"])
    rates[:] = np.array([[[2.0]]])
    op_result = OperatorResult(ufloat(1.0, 0.1), rates)

    StepResult.save(op, x, op_result, [0.0, 1.0], 0.0, 0)

    with h5py.File('depletion_results.h5', 'r') as handle:
        assert 'reaction rates' not in handle
        assert 'reactions' not in handle

    res = Results('depletion_results.h5')
    assert res[0].rates.size == 0


def test_results_save_tt_rates(run_in_tmpdir):
    """StepResult.save writes TT reaction-rate data without dense rates."""

    op = MagicMock()
    op.prev_res = None
    op._tt_depletion_used = True
    vol_dict = {"1": 2.0}
    nuc_list = ["U235", "Xe135"]
    burn_list = ["1"]
    name_list = {mat: "" for mat in burn_list}
    op.get_results_info.return_value = (
        vol_dict, nuc_list, burn_list, burn_list, name_list)
    op.reaction_rates = ReactionRates(
        burn_list, ["U235"], ["fission", "capture"])

    cores = [
        np.arange(1.0, 3.0).reshape(1, 2, 1),
        np.arange(3.0, 9.0).reshape(1, 3, 2),
        np.arange(9.0, 17.0).reshape(2, 4, 1),
    ]
    tally = MagicMock()
    tally.num_realizations = 4
    tally.tt_sum = TT(cores)
    op._rate_helper = MagicMock()
    op._rate_helper._tt_eps = 1.0e-20
    op._rate_helper._tt_shape = (2, 3, 4)
    op._rate_helper._rate_tally = tally

    x = [np.array([1.0, 2.0])]
    reaction_rate_mask = np.array([[True, False]])
    op_result = OperatorResult(
        ufloat(1.0, 0.1),
        _TTReactionRates(3.0, reaction_rate_mask=reaction_rate_mask))

    StepResult.save(
        op, x, op_result, [0.0, 1.0], 1.0, 0, write_rates=True)

    with h5py.File('depletion_results.h5', 'r') as handle:
        assert 'reaction rates' not in handle
        _assert_dense_atom_numbers(handle, 0, 0, x)
        assert 'reactions' in handle
        assert handle['nuclides/U235'].attrs['reaction rate index'] == 0
        assert handle['reactions/fission'].attrs['index'] == 0
        assert handle['reactions/capture'].attrs['index'] == 1

        group = handle['tt_depletion_reaction_rates']
        assert group.attrs['format_version'] == 1
        assert group.attrs['stored_values'] == b'tally_mean'
        assert group.attrs['normalization_mode'] == b'fission-q'
        assert group.attrs['tt_eps'] == 1.0e-20
        np.testing.assert_array_equal(
            group['reaction_rate_mask'][()], reaction_rate_mask)

        step = group['steps/0']
        assert not step.attrs['zero_source']
        assert step['normalization_factor'][()] == 3.0
        assert step['n_realizations'][()] == 4
        np.testing.assert_array_equal(step['tt_shape'][()], [2, 3, 4])

        tt_mean = step['tt_mean']
        assert tt_mean['n_cores'][()] == 3
        np.testing.assert_array_equal(tt_mean['core_0_shape'][()], (1, 2, 1))
        np.testing.assert_array_equal(tt_mean['core_1_shape'][()], (1, 3, 2))
        np.testing.assert_array_equal(tt_mean['core_2_shape'][()], (2, 4, 1))
        np.testing.assert_allclose(
            tt_mean['core_0'][()], (cores[0] / 4).ravel())
        np.testing.assert_allclose(tt_mean['core_1'][()], cores[1].ravel())
        np.testing.assert_allclose(tt_mean['core_2'][()], cores[2].ravel())

    results = Results('depletion_results.h5')
    np.testing.assert_allclose(results[0].data, [[1.0, 2.0]])
    times, atoms = results.get_atoms("1", "Xe135")
    np.testing.assert_allclose(times, [0.0])
    np.testing.assert_allclose(atoms, [2.0])


def test_tt_depletion_rates_reader(run_in_tmpdir):
    """TTDepletionRates reconstructs requested reaction rates on demand."""

    op = MagicMock()
    op.prev_res = None
    op._tt_depletion_used = True
    vol_dict = {"1": 2.0}
    nuc_list = ["U235"]
    burn_list = ["1"]
    name_list = {mat: "" for mat in burn_list}
    op.get_results_info.return_value = (
        vol_dict, nuc_list, burn_list, burn_list, name_list)
    op.reaction_rates = ReactionRates(
        burn_list, ["U235"], ["fission", "capture"])

    cores = [
        np.array([[[70.0], [110.0]]]),
    ]
    tally = MagicMock()
    tally.num_realizations = 5
    tally.tt_sum = TT(cores)
    op._rate_helper = MagicMock()
    op._rate_helper._tt_eps = 1.0e-12
    op._rate_helper._tt_shape = (2,)
    op._rate_helper._rate_tally = tally

    x = [np.array([1.0])]
    reaction_rate_mask = np.array([[True, False]])
    op_result = OperatorResult(
        ufloat(1.0, 0.1),
        _TTReactionRates(3.0, reaction_rate_mask=reaction_rate_mask))

    StepResult.save(
        op, x, op_result, [0.0, 1.0], 1.0, 0, write_rates=True)

    with h5py.File('depletion_results.h5', 'r') as handle:
        _assert_dense_atom_numbers(handle, 0, 0, x)

    reader = TTDepletionRates('depletion_results.h5')
    assert reader.tt_eps == 1.0e-12
    np.testing.assert_array_equal(reader.reaction_rate_mask, reaction_rate_mask)
    expected = np.array([[14.0, 22.0]]) * 3.0 / 2.0e24

    material_rates = reader.get_material_rates(0, "1")
    np.testing.assert_allclose(material_rates, expected)
    assert material_rates.index_nuc == {"U235": 0}
    assert material_rates.index_rx == {"fission": 0, "capture": 1}
    assert reader.get_rate(0, "1", "U235", "capture") == pytest.approx(
        expected[0, 1])

    dense_rates = reader.to_reaction_rates(0)
    assert dense_rates.shape == (1, 1, 2)
    np.testing.assert_allclose(dense_rates[0], expected)

    with pytest.raises(IndexError, match="out of bounds"):
        reader.get_material_rates(1, "1")


def test_results_save_tt_zero_source_rates(run_in_tmpdir):
    """Zero-source TT steps write a marker without stale TT cores."""

    op = MagicMock()
    op.prev_res = None
    op._tt_depletion_used = True
    vol_dict = {"1": 2.0}
    nuc_list = ["U235"]
    burn_list = ["1"]
    name_list = {mat: "" for mat in burn_list}
    op.get_results_info.return_value = (
        vol_dict, nuc_list, burn_list, burn_list, name_list)
    op.reaction_rates = ReactionRates(burn_list, nuc_list, ["fission"])
    op._rate_helper = MagicMock()
    op._rate_helper._tt_eps = 1.0e-30
    op._rate_helper._tt_shape = (1,)

    x = [np.array([1.0])]
    reaction_rate_mask = np.array([[True]])
    op_result = OperatorResult(
        ufloat(0.0, 0.0), _TTReactionRates(
            0.0, zero_source=True, reaction_rate_mask=reaction_rate_mask))

    StepResult.save(
        op, x, op_result, [0.0, 0.0], 0.0, 0, write_rates=True)

    with h5py.File('depletion_results.h5', 'r') as handle:
        assert 'reaction rates' not in handle
        _assert_dense_atom_numbers(handle, 0, 0, x)
        group = handle['tt_depletion_reaction_rates']
        assert group.attrs['tt_eps'] == 1.0e-30
        np.testing.assert_array_equal(
            group['reaction_rate_mask'][()], reaction_rate_mask)
        step = handle['tt_depletion_reaction_rates/steps/0']
        assert step.attrs['zero_source']
        assert step['normalization_factor'][()] == 0.0
        assert step['n_realizations'][()] == 0
        np.testing.assert_array_equal(step['tt_shape'][()], [1])
        assert 'tt_mean' not in step

    reader = TTDepletionRates('depletion_results.h5')
    assert reader.tt_eps == 1.0e-30
    material_rates = reader.get_material_rates(0, "1")
    np.testing.assert_allclose(material_rates, [[0.0]])
    assert reader.get_rate(0, "1", "U235", "fission") == 0.0
    np.testing.assert_allclose(reader.to_reaction_rates(0), [[[0.0]]])


def test_results_save_tt_mode_dense_atom_number(run_in_tmpdir):
    """StepResult.save writes dense atom-number data in TT-rate mode."""

    op = MagicMock()
    op.prev_res = None
    op._tt_depletion_used = True
    vol_dict = {"1": 2.0, "2": 4.0}
    nuc_list = ["U235", "Xe135"]
    burn_list = ["1", "2"]
    name_list = {mat: "" for mat in burn_list}
    op.get_results_info.return_value = (
        vol_dict, nuc_list, burn_list, burn_list, name_list)
    dense = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    op_result = OperatorResult(
        ufloat(1.0, 0.1), _TTReactionRates(3.0))

    StepResult.save(
        op, list(dense), op_result, [0.0, 1.0], 1.0, 0, write_rates=False)

    with h5py.File('depletion_results.h5', 'r') as handle:
        _assert_dense_atom_numbers(handle, 0, 0, dense)

    results = Results('depletion_results.h5')
    assert len(results) == 1
    time, keff = results.get_keff()
    np.testing.assert_allclose(time, [0.0])
    np.testing.assert_allclose(keff, [[1.0, 0.1]])
    np.testing.assert_allclose(results.get_times(time_units="s"), [0.0])
    times, atoms = results.get_atoms("2", "Xe135")
    np.testing.assert_allclose(times, [0.0])
    np.testing.assert_allclose(atoms, [4.0])
    material = results[0].get_material("2")
    atom_densities = material.get_nuclide_atom_densities()
    np.testing.assert_allclose(atom_densities["U235"], 0.75e-24)
    np.testing.assert_allclose(atom_densities["Xe135"], 1.0e-24)


def test_bad_integrator_inputs():
    """Test failure modes for Integrator inputs"""

    op = MagicMock()
    op.prev_res = None
    op.chain = None
    op.heavy_metal = 1.0
    timesteps = [1]

    # No power nor power density given
    with pytest.raises(ValueError, match="Either power"):
        PredictorIntegrator(op, timesteps)

    # Length of power != length time
    with pytest.raises(ValueError, match="number of powers"):
        PredictorIntegrator(op, timesteps, power=[1, 2])

    # Length of power density != length time
    with pytest.raises(ValueError, match="number of powers"):
        PredictorIntegrator(op, timesteps, power_density=[1, 2])

    # SI integrator with bad steps
    with pytest.raises(TypeError, match="n_steps"):
        SICELIIntegrator(op, timesteps, [1], n_steps=2.5)

    with pytest.raises(ValueError, match="n_steps"):
        SICELIIntegrator(op, timesteps, [1], n_steps=0)

    with pytest.raises(ValueError, match="Solver failure"):
        PredictorIntegrator(op, timesteps, power=1, solver="failure")

    with pytest.raises(TypeError, match=".*callable.*NoneType"):
        PredictorIntegrator(op, timesteps, power=1, solver=None)

    with pytest.raises(ValueError, match="four arguments"):
        PredictorIntegrator(op, timesteps, power=1, solver=mock_bad_solver_nargs)

    with pytest.raises(ValueError, match="default to 1"):
        PredictorIntegrator(op, timesteps, power=1,
                            solver=mock_bad_solver_fourth_required)

    with pytest.raises(ValueError, match="substeps"):
        PredictorIntegrator(op, timesteps, power=1, substeps=0)

    with pytest.raises(ValueError, match="substeps"):
        PredictorIntegrator(op, timesteps, power=1, substeps=-1)


def test_tt_depletion_allows_predictor():
    op = MagicMock()
    op.prev_res = None
    op.chain = None
    op.heavy_metal = 1.0
    op._tt_depletion_used = True

    PredictorIntegrator(op, [1], power=1)


@pytest.mark.parametrize("integrator", INTEGRATORS[1:])
def test_tt_depletion_requires_predictor(integrator):
    op = MagicMock()
    op.prev_res = None
    op.chain = None
    op.heavy_metal = 1.0
    op._tt_depletion_used = True

    with pytest.raises(ValueError, match="only supported with PredictorIntegrator"):
        integrator(op, [1], power=1)


def test_predictor_tt_depletes_material_slices(monkeypatch):
    class RateSlice(np.ndarray):
        def __new__(cls, values):
            obj = np.asarray(values).view(cls)
            obj.index_nuc = {'1': 0, '2': 1}
            obj.index_rx = {'rx': 0}
            return obj

    class Chain:
        def __init__(self):
            self.fission_yields = [{'mat': '1'}, {'mat': '2'}]
            self.seen_yields = []

        def form_matrix(self, rates, fission_yields=None):
            assert rates.index_nuc == {'1': 0, '2': 1}
            assert rates.index_rx == {'rx': 0}
            self.seen_yields.append(fission_yields)
            return rates

    class Context:
        _tt_reaction_rates = True
        normalization_factor = 5.0

    def solver(matrix, n0, dt, substeps=1):
        return n0 + dt * matrix[:, 0]

    chain = Chain()
    op = MagicMock()
    op.prev_res = None
    op.chain = chain
    op.heavy_metal = 1.0
    op.local_mats = ['1', '2']
    op._tt_depletion_used = True
    n = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    rate_calls = []

    def get_tt_reaction_rates(mat, normalization_factor):
        rate_calls.append((mat, normalization_factor))
        if mat == '1':
            return RateSlice([[1.0], [-2.0]])
        return RateSlice([[3.0], [4.0]])

    monkeypatch.setattr(
        'openmc.deplete.tt_depletion._get_tt_reaction_rates',
        lambda operator, mat, normalization_factor:
            get_tt_reaction_rates(mat, normalization_factor))
    integrator = PredictorIntegrator(op, [1], power=1, solver=solver)

    _, result = integrator(
        n, Context(), 1.0, 1.0, 0)

    assert rate_calls == [('1', 5.0), ('2', 5.0)]
    assert chain.seen_yields == [{'mat': '1'}, {'mat': '2'}]
    np.testing.assert_allclose(result[0], [1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(result[1], [3.0, 4.0], atol=1.0e-12)


def test_predictor_tt_uses_dense_atom_numbers(monkeypatch):
    class RateSlice(np.ndarray):
        def __new__(cls, values):
            obj = np.asarray(values).view(cls)
            obj.index_nuc = {'1': 0, '2': 1}
            obj.index_rx = {'rx': 0}
            return obj

    class Chain:
        fission_yields = [None]

        @staticmethod
        def form_matrix(rates, fission_yields=None):
            return rates

    class Context:
        _tt_reaction_rates = True
        normalization_factor = 5.0

    def solver(matrix, n0, dt, substeps=1):
        return n0 + dt * matrix[:, 0]

    op = MagicMock()
    op.prev_res = None
    op.chain = Chain()
    op.heavy_metal = 1.0
    op.local_mats = ['1', '2']
    op._tt_depletion_used = True
    n = [np.array([10.0, 20.0]), np.array([30.0, 40.0])]

    def get_tt_reaction_rates(mat, normalization_factor):
        if mat == '1':
            return RateSlice([[1.0], [2.0]])
        return RateSlice([[3.0], [4.0]])

    monkeypatch.setattr(
        'openmc.deplete.tt_depletion._get_tt_reaction_rates',
        lambda operator, mat, normalization_factor:
            get_tt_reaction_rates(mat, normalization_factor))
    integrator = PredictorIntegrator(op, [1], power=1, solver=solver)

    _, result = integrator(
        n, Context(), 1.0, 1.0, 0)

    np.testing.assert_allclose(result[0], [11.0, 22.0], atol=1.0e-12)
    np.testing.assert_allclose(result[1], [33.0, 44.0], atol=1.0e-12)


def test_predictor_tt_zero_source_uses_zero_rates(monkeypatch):
    class Chain:
        fission_yields = [None]

        def __init__(self):
            self.matrices = []

        def form_matrix(self, rates, fission_yields=None):
            np.testing.assert_allclose(rates, 0.0)
            self.matrices.append(rates.copy())
            return rates

    def solver(matrix, n0, dt, substeps=1):
        return n0 + matrix[:, 0] * dt

    chain = Chain()
    op = MagicMock()
    op.prev_res = None
    op.chain = chain
    op.heavy_metal = 1.0
    op.local_mats = ['1', '2']
    op.reaction_rates = ReactionRates(op.local_mats, ['1', '2'], ['rx'])
    op._tt_depletion_used = True
    n = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    monkeypatch.setattr(
        'openmc.deplete.tt_depletion._get_tt_reaction_rates',
        lambda *args: pytest.fail("TT rates should not be reconstructed"))

    integrator = PredictorIntegrator(op, [1], power=1, solver=solver)
    rates = _TTReactionRates(0.0, zero_source=True)

    _, result = integrator(
        n, rates, 1.0, 1.0, 0)

    assert len(chain.matrices) == 2
    np.testing.assert_allclose(result[0], [1.0, 2.0], atol=1.0e-12)
    np.testing.assert_allclose(result[1], [3.0, 4.0], atol=1.0e-12)


def test_predictor_tt_requires_composition_per_local_mat():
    class Context:
        _tt_reaction_rates = True
        normalization_factor = 1.0

    op = MagicMock()
    op.prev_res = None
    op.chain = MagicMock()
    op.heavy_metal = 1.0
    op.local_mats = ['1', '2']
    op._tt_depletion_used = True

    integrator = PredictorIntegrator(op, [1], power=1)

    with pytest.raises(ValueError, match="Number of compositions"):
        integrator([np.array([1.0])], Context(), 1.0, 1.0, 0)


def mock_good_solver(A, n, t, substeps=1):
    return n.copy()


def mock_good_solver_substeps(A, n, t, substeps=1):
    return n + substeps


def mock_unsupported_substeps_solver(A, n, t, substeps=1):
    if substeps > 1:
        raise NotImplementedError("substeps > 1 not supported")
    return n.copy()


def mock_bad_solver_nargs(A, n):
    pass


def mock_bad_solver_fourth_required(A, n, t, substeps):
    pass


@pytest.mark.parametrize("scheme", dummy_operator.SCHEMES)
def test_integrator(run_in_tmpdir, scheme):
    """Test the integrators against their expected values"""

    bundle = dummy_operator.SCHEMES[scheme]
    operator = dummy_operator.DummyOperator()
    bundle.solver(operator, [0.75, 0.75], 1.0).integrate()

    # get expected results

    res = Results(operator.output_dir / "depletion_results.h5")

    t1, y1 = res.get_atoms("1", "1")
    t2, y2 = res.get_atoms("1", "2")

    assert (t1 == [0.0, 0.75, 1.5]).all()
    assert y1 == pytest.approx(bundle.atoms_1)
    assert (t2 == [0.0, 0.75, 1.5]).all()
    assert y2 == pytest.approx(bundle.atoms_2)

    # test structure of depletion time dataset
    dep_time = res.get_depletion_time()
    assert dep_time.shape == (2, )
    assert all(dep_time > 0)

    integrator = bundle.solver(operator, [0.75], 1, solver=cram.CRAM48)
    assert integrator.solver is cram.CRAM48

    integrator = bundle.solver(operator, [0.75], 1, solver="cram16")
    assert integrator.solver is cram.CRAM16

    integrator = bundle.solver(operator, [0.75], 1, solver=cram.Cram48Solver,
                               substeps=2)
    assert integrator.solver is cram.Cram48Solver
    assert integrator.substeps == 2

    integrator.solver = mock_good_solver
    assert integrator.solver is mock_good_solver

    lfunc = lambda A, n, t, substeps=1: mock_good_solver(A, n, t, substeps)
    integrator.solver = lfunc
    assert integrator.solver is lfunc

    integrator.solver = mock_good_solver_substeps
    assert integrator.solver is mock_good_solver_substeps


def test_custom_solver_with_default_substeps(monkeypatch):
    operator = dummy_operator.DummyOperator()
    n = operator.initial_condition()
    rates = operator(n, 1.0).rates
    integrator = PredictorIntegrator(
        operator, [0.75], power=1.0, solver=mock_good_solver)
    monkeypatch.setattr(pool, "USE_MULTIPROCESSING", False)

    _, result = integrator._timed_deplete(n, rates, 0.75)

    np.testing.assert_array_equal(result[0], n[0])


def test_substep_aware_custom_solver_receives_substeps(monkeypatch):
    operator = dummy_operator.DummyOperator()
    n = operator.initial_condition()
    rates = operator(n, 1.0).rates
    integrator = PredictorIntegrator(
        operator, [0.75], power=1.0, solver=mock_good_solver_substeps,
        substeps=3)
    monkeypatch.setattr(pool, "USE_MULTIPROCESSING", False)

    _, result = integrator._timed_deplete(n, rates, 0.75)

    np.testing.assert_array_equal(result[0], n[0] + 3)


def test_custom_solver_propagates_substeps_error(monkeypatch):
    operator = dummy_operator.DummyOperator()
    n = operator.initial_condition()
    rates = operator(n, 1.0).rates
    integrator = PredictorIntegrator(
        operator, [0.75], power=1.0,
        solver=mock_unsupported_substeps_solver, substeps=2)
    monkeypatch.setattr(pool, "USE_MULTIPROCESSING", False)

    with pytest.raises(NotImplementedError, match="not supported"):
        integrator._timed_deplete(n, rates, 0.75)


def test_custom_solver_requires_four_args():
    op = MagicMock()
    op.prev_res = None
    op.chain = None
    op.heavy_metal = 1.0

    with pytest.raises(ValueError, match="four arguments"):
        PredictorIntegrator(op, [1], power=1, solver=mock_bad_solver_nargs)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_timesteps(integrator):
    # Crate fake operator
    op = MagicMock()
    op.prev_res = None
    op.chain = None

    # Set heavy metal mass and power randomly
    op.heavy_metal = uniform(0, 10000)
    power = uniform(0, 1e6)

    # Reference timesteps in seconds
    day = 86400.0
    ref_timesteps = [1*day, 2*day, 5*day, 10*day]

    # Case 1, timesteps in seconds
    timesteps = ref_timesteps
    x = integrator(op, timesteps, power, timestep_units='s')
    assert np.allclose(x.timesteps, ref_timesteps)

    # Case 2, timesteps in minutes
    minute = 60
    timesteps = [t / minute for t in ref_timesteps]
    x = integrator(op, timesteps, power, timestep_units='min')
    assert np.allclose(x.timesteps, ref_timesteps)

    # Case 3, timesteps in hours
    hour = 60*60
    timesteps = [t / hour for t in ref_timesteps]
    x = integrator(op, timesteps, power, timestep_units='h')
    assert np.allclose(x.timesteps, ref_timesteps)

    # Case 4, timesteps in days
    timesteps = [t / day for t in ref_timesteps]
    x = integrator(op, timesteps, power, timestep_units='d')
    assert np.allclose(x.timesteps, ref_timesteps)

    # Case 5, timesteps in MWd/kg
    kilograms = op.heavy_metal / 1000.0
    days = [t/day for t in ref_timesteps]
    megawatts = power / 1000000.0
    burnup = [t * megawatts / kilograms for t in days]
    x = integrator(op, burnup, power, timestep_units='MWd/kg')
    assert np.allclose(x.timesteps, ref_timesteps)

    # Case 6, mixed units
    burnup_per_day = (1e-6*power) / kilograms
    timesteps = [(burnup_per_day, 'MWd/kg'), (2*day, 's'), (5, 'd'),
                 (10*burnup_per_day, 'MWd/kg')]
    x = integrator(op, timesteps, power)
    assert np.allclose(x.timesteps, ref_timesteps)

    # Bad units should raise an exception
    with pytest.raises(ValueError, match="unit"):
        integrator(op, ref_timesteps, power, timestep_units='🐨')
    with pytest.raises(ValueError, match="unit"):
        integrator(op, [(800.0, 'gorillas')], power)
