"""Pin-level geometry builders for the 3D assembly depletion example."""

import openmc

from constants import (
    ACTIVE_AXIAL_HEIGHTS, MAIN_FUEL_AXIAL_START, MAIN_FUEL_AXIAL_STOP,
    N_RADIAL,
)
from surfaces import get_fuel_pin_surfaces, get_tube_quadrant_surfaces


def fuel_pin_universe(
        materials, fuel_material, cutback_material=None, n_radial=N_RADIAL):
    """Return one fuel pin universe with independent fuel materials."""

    (
        fuel_ors, gap_or, clad_or, fuel_axial_surfaces, fuel_areas
    ) = get_fuel_pin_surfaces(n_radial)

    bottom_end_region = -fuel_axial_surfaces[0]
    top_end_region = +fuel_axial_surfaces[-1]

    cells = []
    for r in range(len(fuel_ors)):
        radial_region = (
            -fuel_ors[0] if r == 0 else +fuel_ors[r - 1] & -fuel_ors[r])

        for z in range(len(ACTIVE_AXIAL_HEIGHTS)):
            is_cutback_zone = (
                z < MAIN_FUEL_AXIAL_START or z >= MAIN_FUEL_AXIAL_STOP)
            if cutback_material is not None and is_cutback_zone:
                zone_template = cutback_material
            else:
                zone_template = fuel_material

            zone_height = (
                fuel_axial_surfaces[z + 1].z0 - fuel_axial_surfaces[z].z0)
            fuel_mat = zone_template.clone()
            fuel_mat.volume = fuel_areas[r] * zone_height

            zone_region = (
                +fuel_axial_surfaces[z] & -fuel_axial_surfaces[z + 1])
            cells.append(openmc.Cell(
                fill=fuel_mat,
                region=radial_region & zone_region))

    cells.extend([
        openmc.Cell(
            fill=materials.ss304,
            region=-fuel_ors[-1] & bottom_end_region),
        openmc.Cell(
            fill=materials.spring,
            region=-fuel_ors[-1] & top_end_region),
        openmc.Cell(
            fill=materials.air,
            region=+fuel_ors[-1] & -gap_or),
        openmc.Cell(
            fill=materials.zirlo,
            region=+gap_or & -clad_or),
        openmc.Cell(
            fill=materials.gridmix,
            region=+clad_or),
    ])

    universe = openmc.Universe()
    universe.add_cells(cells)
    return universe


def tube_quadrant_universe(x0, y0, mod_or, clad_or, materials):
    """Return one lattice slot containing a tube quadrant."""

    surfaces = get_tube_quadrant_surfaces(x0, y0, mod_or, clad_or)

    cells = [
        openmc.Cell(
            fill=materials.gridmix,
            region=-surfaces.mod_or),
        openmc.Cell(
            fill=materials.inc625,
            region=+surfaces.mod_or & -surfaces.clad_or),
        openmc.Cell(
            fill=materials.gridmix,
            region=+surfaces.clad_or),
    ]

    universe = openmc.Universe()
    universe.add_cells(cells)
    return universe


def tube_quadrants(origin, mod_or, clad_or, pin_pitch, materials):
    """Return the four lattice entries occupied by a 2 x 2 tube."""

    row, col = origin
    half_pitch = 0.5 * pin_pitch
    quadrant_specs = (
        (row - 1, col - 1, half_pitch, -half_pitch),
        (row - 1, col, -half_pitch, -half_pitch),
        (row, col - 1, half_pitch, half_pitch),
        (row, col, -half_pitch, half_pitch),
    )

    return {
        (r, c): tube_quadrant_universe(x0, y0, mod_or, clad_or, materials)
        for r, c, x0, y0 in quadrant_specs
    }
