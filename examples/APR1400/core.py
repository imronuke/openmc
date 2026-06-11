"""Core-level geometry builder for the 3D depletion example."""

import openmc

from assemblies import fuel_assembly, reflector_assembly
from constants import (
    ASSEMBLY_WIDTH, BOTTOM_REFLECTOR_HEIGHT, CORE_MAP, STACK_BOTTOM_Z,
    STACK_TOP_Z, TOP_REFLECTOR_HEIGHT
)
from materials import create_materials


def core_lattice_entry(assembly_type, materials):
    """Return the assembly universe for one core-map entry."""

    if assembly_type.upper() == 'RF':
        return reflector_assembly(materials=materials)

    return fuel_assembly(assembly_type, materials=materials)


def core_geometry(core_map=CORE_MAP, materials=None):
    """Return the top-level core geometry and model bounds."""

    if materials is None:
        materials = create_materials()

    assembly_rows = []

    for row in core_map:
        assembly_row = []
        for assembly_type in row:
            assembly_row.append(core_lattice_entry(assembly_type, materials))
        assembly_rows.append(assembly_row)

    n_rows = len(core_map)
    n_cols = len(core_map[0])
    core_width = n_cols * ASSEMBLY_WIDTH
    core_depth = n_rows * ASSEMBLY_WIDTH
    model_bottom_z = STACK_BOTTOM_Z - BOTTOM_REFLECTOR_HEIGHT
    model_top_z = STACK_TOP_Z + TOP_REFLECTOR_HEIGHT

    core_lattice = openmc.RectLattice(name='Core')
    core_lattice.pitch = (ASSEMBLY_WIDTH, ASSEMBLY_WIDTH)
    core_lattice.lower_left = (-0.5 * core_width, -0.5 * core_depth)
    core_lattice.outer = openmc.Universe(cells=[
        openmc.Cell(fill=materials.mod)
    ])
    core_lattice.universes = assembly_rows

    outer_boundary = openmc.model.RectangularPrism(
        core_width, core_depth, boundary_type='reflective')
    model_bottom = openmc.ZPlane(z0=model_bottom_z, boundary_type='vacuum')
    stack_bottom = openmc.ZPlane(z0=STACK_BOTTOM_Z)
    stack_top = openmc.ZPlane(z0=STACK_TOP_Z)
    model_top = openmc.ZPlane(z0=model_top_z, boundary_type='vacuum')

    bottom_reflector = openmc.Cell(
        name='Bottom reflector',
        fill=materials.mod,
        region=-outer_boundary & +model_bottom & -stack_bottom)
    core_cell = openmc.Cell(
        name='Core',
        fill=core_lattice,
        region=-outer_boundary & +stack_bottom & -stack_top)
    top_reflector = openmc.Cell(
        name='Top reflector',
        fill=materials.mod,
        region=-outer_boundary & +stack_top & -model_top)

    bounds = [
        -0.5 * core_width, -0.5 * core_depth, model_bottom_z,
        0.5 * core_width, 0.5 * core_depth, model_top_z
    ]

    geometry = openmc.Geometry([bottom_reflector, core_cell, top_reflector])
    return geometry, bounds
