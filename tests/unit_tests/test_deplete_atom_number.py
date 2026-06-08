""" Tests for the AtomNumber class """

import numpy as np
from openmc.deplete import atom_number
from openmc.deplete.tt_depletion import TTAtomDensities
from openmc.tt import _auto_tt_shape, tt_svd


def test_auto_tt_shape():
    assert _auto_tt_shape(1) == (1,)
    assert _auto_tt_shape(41) == (41,)
    assert _auto_tt_shape(600) == (20, 30)
    assert np.prod(_auto_tt_shape(10_800)) == 10_800


def test_indexing():
    """Tests the __getitem__ and __setitem__ routines simultaneously."""

    local_mats = ["10000", "10001"]
    nuclides = ["U238", "U235", "U234"]
    volume = {"10000" : 0.38, "10001" : 0.21}

    number = atom_number.AtomNumber(local_mats, nuclides, volume, 2)

    number["10000", "U238"] = 1.0
    number["10001", "U238"] = 2.0
    number["10000", "U235"] = 3.0
    number["10001", "U235"] = 4.0

    # String indexing
    assert number["10000", "U238"] == 1.0
    assert number["10001", "U238"] == 2.0
    assert number["10000", "U235"] == 3.0
    assert number["10001", "U235"] == 4.0

    # Int indexing
    assert number[0, 0] == 1.0
    assert number[1, 0] == 2.0
    assert number[0, 1] == 3.0
    assert number[1, 1] == 4.0

    number[0, 0] = 5.0

    assert number[0, 0] == 5.0
    assert number["10000", "U238"] == 5.0


def test_properties():
    """Test properties. """
    local_mats = ["10000", "10001"]
    nuclides = ["U238", "U235", "Gd157"]
    volume = {"10000" : 0.38, "10001" : 0.21}

    number = atom_number.AtomNumber(local_mats, nuclides, volume, 2)

    assert list(number.materials) == ["10000", "10001"]
    assert number.n_nuc == 3
    assert list(number.nuclides) == ["U238", "U235", "Gd157"]
    assert number.burnable_nuclides == ["U238", "U235"]


def test_density_indexing():
    """Tests the get and set_atom_density routines simultaneously."""

    local_mats = ["10000", "10001", "10002"]
    nuclides = ["U238", "U235", "U234"]
    volume = {"10000" : 0.38, "10001" : 0.21}

    number = atom_number.AtomNumber(local_mats, nuclides, volume, 2)

    number.set_atom_density("10000", "U238", 1.0)
    number.set_atom_density("10001", "U238", 2.0)
    number.set_atom_density("10002", "U238", 3.0)
    number.set_atom_density("10000", "U235", 4.0)
    number.set_atom_density("10001", "U235", 5.0)
    number.set_atom_density("10002", "U235", 6.0)
    number.set_atom_density("10000", "U234", 7.0)
    number.set_atom_density("10001", "U234", 8.0)
    number.set_atom_density("10002", "U234", 9.0)

    # String indexing
    assert number.get_atom_density("10000", "U238") == 1.0
    assert number.get_atom_density("10001", "U238") == 2.0
    assert number.get_atom_density("10002", "U238") == 3.0
    assert number.get_atom_density("10000", "U235") == 4.0
    assert number.get_atom_density("10001", "U235") == 5.0
    assert number.get_atom_density("10002", "U235") == 6.0
    assert number.get_atom_density("10000", "U234") == 7.0
    assert number.get_atom_density("10001", "U234") == 8.0
    assert number.get_atom_density("10002", "U234") == 9.0

    # Int indexing
    assert number.get_atom_density(0, 0) == 1.0
    assert number.get_atom_density(1, 0) == 2.0
    assert number.get_atom_density(2, 0) == 3.0
    assert number.get_atom_density(0, 1) == 4.0
    assert number.get_atom_density(1, 1) == 5.0
    assert number.get_atom_density(2, 1) == 6.0
    assert number.get_atom_density(0, 2) == 7.0
    assert number.get_atom_density(1, 2) == 8.0
    assert number.get_atom_density(2, 2) == 9.0


    number.set_atom_density(0, 0, 5.0)
    assert number.get_atom_density(0, 0) == 5.0

    # Verify volume is used correctly
    assert number[0, 0] == 5.0 * 0.38
    assert number[1, 0] == 2.0 * 0.21
    assert number[2, 0] == 3.0 * 1.0
    assert number[0, 1] == 4.0 * 0.38
    assert number[1, 1] == 5.0 * 0.21
    assert number[2, 1] == 6.0 * 1.0
    assert number[0, 2] == 7.0 * 0.38
    assert number[1, 2] == 8.0 * 0.21
    assert number[2, 2] == 9.0 * 1.0


def test_get_mat_slice():
    """Tests getting slices."""

    local_mats = ["10000", "10001", "10002"]
    nuclides = ["U238", "U235", "U234"]
    volume = {"10000" : 0.38, "10001" : 0.21}

    number = atom_number.AtomNumber(local_mats, nuclides, volume, 2)

    number.number = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    sl = number.get_mat_slice(0)

    np.testing.assert_array_equal(sl, np.array([1.0, 2.0]))

    sl = number.get_mat_slice("10000")

    np.testing.assert_array_equal(sl, np.array([1.0, 2.0]))


def test_set_mat_slice():
    """Tests getting slices."""

    local_mats = ["10000", "10001", "10002"]
    nuclides = ["U238", "U235", "U234"]
    volume = {"10000" : 0.38, "10001" : 0.21}

    number = atom_number.AtomNumber(local_mats, nuclides, volume, 2)

    number.set_mat_slice(0, [1.0, 2.0])

    assert number[0, 0] == 1.0
    assert number[0, 1] == 2.0

    number.set_mat_slice("10000", [3.0, 4.0])

    assert number[0, 0] == 3.0
    assert number[0, 1] == 4.0


def test_tt_atom_densities():
    """Tests TT-backed density indexing and atom-number conversion."""

    local_mats = ["10000", "10001", "10002"]
    nuclides = ["U238", "U235", "U234"]
    volume = {"10000": 0.38, "10001": 0.21}
    density = 1.0e-24 * np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    density_tt = tt_svd(density, eps=0.0)
    number = TTAtomDensities(
        local_mats, nuclides, volume, 2, density_tt, (3,), (3,), 0.0)

    assert list(number.materials) == local_mats
    assert list(number.nuclides) == nuclides
    assert number.n_nuc == 3
    assert number.burnable_nuclides == ["U238", "U235"]

    np.testing.assert_allclose(number["10000", "U238"], density[0, 0])
    np.testing.assert_allclose(number["10001", "U235"], density[1, 1])
    np.testing.assert_allclose(number[2, 2], density[2, 2])

    np.testing.assert_allclose(
        number.get_mat_full_density_slice("10000"), density[0])
    np.testing.assert_allclose(
        number.get_mat_density_slice("10000"), density[0, :2])
    np.testing.assert_allclose(
        number.get_mat_density_slice(np.s_[:]), density[:, :2])

    np.testing.assert_allclose(
        number.get_mat_atom_slice("10001"),
        density[1, :2] * volume["10001"] * 1.0e24)

    assert number.get_mat_volume("10000") == 0.38
    assert number.get_mat_volume("10002") == 1.0
    np.testing.assert_allclose(
        number.get_atom_density("10000", "U238"), 1.0)
    np.testing.assert_allclose(number.get_atom_density("10002", "U234"), 9.0)

    atom_per_cc = number.get_atom_densities("10001", units='atom/cm3')
    np.testing.assert_allclose(atom_per_cc["U238"], 4.0)
    atom_per_bcm = number.get_atom_densities("10001")
    np.testing.assert_allclose(atom_per_bcm["U238"], density[1, 0])


def test_tt_atom_densities_clips_significant_negative_values():
    """TTAtomDensities clips negatives below the density warning threshold."""

    local_mats = ["1", "2"]
    nuclides = ["n1", "n2"]
    volume = {"1": 2.0, "2": 4.0}
    density = np.array([
        [-2.0e-21, -1.0e-22],
        [-5.0e-22, 1.0e-24],
    ])
    expected = np.array([
        [0.0, -1.0e-22],
        [-5.0e-22, 1.0e-24],
    ])

    number = TTAtomDensities(
        local_mats, nuclides, volume, 2, tt_svd(density, eps=0.0),
        (2,), (2,), 0.0)

    np.testing.assert_allclose(
        number.get_mat_full_density_slice(np.s_[:]), expected)
    np.testing.assert_allclose(
        number.get_mat_full_density_slice("1"), expected[0])
    np.testing.assert_allclose(number["1", "n1"], 0.0)


def test_tt_atom_densities_factored_shape():
    """Tests TTAtomDensities with split material and nuclide dimensions."""

    local_mats = ["1", "2", "3", "4"]
    nuclides = ["n1", "n2", "n3", "n4", "n5", "n6"]
    volume = {}
    density = 1.0e-24 * np.arange(24.0).reshape((4, 6))

    mat_shape = (2, 2)
    nuc_shape = (2, 3)
    density_tt = tt_svd(
        density.reshape(mat_shape + nuc_shape, order='C'), eps=0.0)
    number = TTAtomDensities(
        local_mats, nuclides, volume, 4, density_tt,
        mat_shape, nuc_shape, 0.0)

    assert number.density_tt.shape == (2, 2, 2, 3)
    np.testing.assert_allclose(
        number.get_mat_full_density_slice("4"), density[3])
    np.testing.assert_allclose(number["3", "n5"], density[2, 4])


def test_tt_atom_densities_compress_from_density():
    """Tests replacing TTAtomDensities data while preserving metadata."""

    local_mats = ["1", "2"]
    nuclides = ["n1", "n2", "n3"]
    volume = {"1": 2.0, "2": 4.0}
    density = 1.0e-24 * np.arange(6.0).reshape((2, 3))
    updated = density + 1.0e-24

    density_tt = tt_svd(density, eps=0.0)
    number = TTAtomDensities(
        local_mats, nuclides, volume, 2, density_tt, (2,), (3,), 0.0)
    number.compress_from_density(updated)

    assert number.mat_shape == (2,)
    assert number.nuc_shape == (3,)
    assert number.tt_eps == 0.0
    np.testing.assert_allclose(
        number.get_mat_full_density_slice("1"), updated[0])
    np.testing.assert_allclose(
        number.get_mat_density_slice("2"), updated[1, :2])
    np.testing.assert_allclose(
        number.get_atom_density("2", "n3"), updated[1, 2] * 1.0e24)


def test_tt_atom_densities_dense_to_tt_density():
    """Tests creating updated TT densities while preserving metadata."""

    local_mats = ["1", "2"]
    nuclides = ["n1", "n2", "n3"]
    volume = {"1": 2.0, "2": 4.0}
    density = 1.0e-24 * np.array([
        [1.0, 2.0, 100.0],
        [3.0, 4.0, 200.0],
    ])
    density_tt = tt_svd(density, eps=0.0)
    number = TTAtomDensities(
        local_mats, nuclides, volume, 2, density_tt, (2,), (3,), 0.0)

    expected = density.copy()
    expected[0, :2] = np.array([10.0, 20.0]) / volume["1"] * 1.0e-24
    expected[1, :2] = np.array([30.0, 40.0]) / volume["2"] * 1.0e-24
    updated_number = number.dense_to_tt_density(expected)

    assert updated_number is not number
    assert updated_number.density_tt is not number.density_tt
    atoms = TTAtomDensities._density_to_atoms(
        expected, np.array([volume["1"], volume["2"]])[:, None])
    np.testing.assert_allclose(
        TTAtomDensities._atoms_to_density(
            atoms, np.array([volume["1"], volume["2"]])[:, None]),
        expected)
    np.testing.assert_allclose(
        number.get_mat_full_density_slice(np.s_[:]), density)
    np.testing.assert_allclose(
        updated_number.get_mat_full_density_slice(np.s_[:]), expected)
    np.testing.assert_allclose(
        updated_number.get_mat_atom_slice(np.s_[:]),
        [[10.0, 20.0], [30.0, 40.0]])


def test_tt_atom_densities_invalid_shape():
    local_mats = ["1", "2"]
    nuclides = ["n1", "n2", "n3"]
    density = np.ones((2, 3))

    with np.testing.assert_raises(ValueError):
        density_tt = tt_svd(density, eps=0.0)
        TTAtomDensities(
            local_mats, nuclides, {}, 2, density_tt, (3,), (3,), 0.0)
