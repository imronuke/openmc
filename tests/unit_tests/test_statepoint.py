import numpy as np
import openmc


def test_get_tally_filter_type(run_in_tmpdir):
    """Test various ways of retrieving tallies from a StatePoint object."""

    mat = openmc.Material()
    mat.add_nuclide("H1", 1.0)
    mat.set_density("g/cm3", 10.0)

    sphere = openmc.Sphere(r=10.0, boundary_type="vacuum")
    cell = openmc.Cell(fill=mat, region=-sphere)
    geometry = openmc.Geometry([cell])

    settings = openmc.Settings()
    settings.particles = 10
    settings.batches = 2
    settings.run_mode = "fixed source"

    reg_mesh = openmc.RegularMesh().from_domain(cell)
    tally1 = openmc.Tally(tally_id=1)
    mesh_filter = openmc.MeshFilter(reg_mesh)
    tally1.filters = [mesh_filter]
    tally1.scores = ["flux"]

    tally2 = openmc.Tally(tally_id=2, name="heating tally")
    cell_filter = openmc.CellFilter(cell)
    tally2.filters = [cell_filter]
    tally2.scores = ["heating"]

    tallies = openmc.Tallies([tally1, tally2])
    model = openmc.Model(
        geometry=geometry, materials=[mat], settings=settings, tallies=tallies
    )

    sp_filename = model.run()

    sp = openmc.StatePoint(sp_filename)

    tally_found = sp.get_tally(filter_type=openmc.MeshFilter)
    assert tally_found.id == 1

    tally_found = sp.get_tally(filter_type=openmc.CellFilter)
    assert tally_found.id == 2

    tally_found = sp.get_tally(filters=[mesh_filter])
    assert tally_found.id == 1

    tally_found = sp.get_tally(filters=[cell_filter])
    assert tally_found.id == 2

    tally_found = sp.get_tally(scores=["heating"])
    assert tally_found.id == 2

    tally_found = sp.get_tally(name="heating tally")
    assert tally_found.id == 2

    tally_found = sp.get_tally(name=None)
    assert tally_found.id == 1

    tally_found = sp.get_tally(id=1)
    assert tally_found.id == 1

    tally_found = sp.get_tally(id=2)
    assert tally_found.id == 2


def test_tt_tally_statepoint_round_trip(run_in_tmpdir):
    """Test per-channel TT statepoint write, read, and restart."""

    mat = openmc.Material()
    mat.add_nuclide("H1", 1.0)
    mat.set_density("g/cm3", 1.0)

    sphere = openmc.Sphere(r=10.0, boundary_type="vacuum")
    cell = openmc.Cell(fill=mat, region=-sphere)
    geometry = openmc.Geometry([cell])

    settings = openmc.Settings()
    settings.particles = 10
    settings.batches = 2
    settings.run_mode = "fixed source"
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0)))

    tally = openmc.Tally(tally_id=1)
    tally.filters = [openmc.MaterialFilter([mat])]
    tally.nuclides = ["H1", "O16"]
    tally.scores = ["total", "absorption"]
    tally.tt_shape = (1,)
    tally.tt_eps = 1.0e-6

    model = openmc.Model(
        geometry=geometry,
        materials=[mat],
        settings=settings,
        tallies=[tally])

    statepoint_path = model.run()
    with openmc.StatePoint(statepoint_path) as statepoint:
        tally_result = statepoint.get_tally(id=1)
        assert tally_result.tt_shape == (1,)
        assert len(tally_result.tt_sum) == 4
        assert not tally_result.tt_sum[2].cores
        assert not tally_result.tt_sum_sq[2].cores
        assert tally_result.get_tt_value(0, 1, 0) == 0.0
        first_mean = tally_result.get_tt_value(0, 0, 0)
        assert np.isfinite(first_mean)
        assert np.isfinite(
            tally_result.get_tt_value(0, 0, 0, value='std_dev'))

    model.settings.batches = 3
    restarted_statepoint_path = model.run(restart_file=statepoint_path)
    with openmc.StatePoint(restarted_statepoint_path) as statepoint:
        tally_result = statepoint.get_tally(id=1)
        assert tally_result.num_realizations == 3
        assert len(tally_result.tt_sum) == 4
        assert not tally_result.tt_sum[2].cores
        assert tally_result.get_tt_value(0, 1, 0) == 0.0
        assert np.isfinite(tally_result.get_tt_value(0, 0, 0))
        assert np.isfinite(
            tally_result.get_tt_value(0, 0, 0, value='std_dev'))
