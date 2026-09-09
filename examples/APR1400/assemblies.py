"""Assembly-level geometry builders for the 3D depletion example."""

import openmc

from constants import (
    ASSEMBLY_PIN_MAPS, LATTICE_WIDTH, N_RADIAL, PIN_MAP_FUEL_MATERIALS,
    PIN_PITCH, PINS_PER_SIDE, TUBE_CLAD_OR, TUBE_MOD_OR
)
from materials import create_materials
from pins import fuel_pin_universe, tube_quadrants


def tube_locations(pin_map, materials):
    """Return tube quadrant universes from 2 x 2 tube blocks in a pin map."""

    locations = {}
    for row in range(PINS_PER_SIDE - 1):
        for col in range(PINS_PER_SIDE - 1):
            if (
                    pin_map[row][col] == 5
                    and pin_map[row][col + 1] == 5
                    and pin_map[row + 1][col] == 5
                    and pin_map[row + 1][col + 1] == 5):
                locations.update(tube_quadrants(
                    (row + 1, col + 1), TUBE_MOD_OR, TUBE_CLAD_OR, PIN_PITCH,
                    materials))
    return locations


def reflector_assembly(materials=None):
    """Return homogeneous MOD reflector assembly universe."""

    if materials is None:
        materials = create_materials()

    return openmc.Universe(cells=[
        openmc.Cell(fill=materials.mod)
    ])


def fuel_assembly(assembly_type='A0', materials=None):
    """Return a 16 x 16 3D fuel assembly universe."""

    if materials is None:
        materials = create_materials()

    assembly_type = assembly_type.upper()
    try:
        pin_map = ASSEMBLY_PIN_MAPS[assembly_type]
    except KeyError as exc:
        raise ValueError(
            f'Unsupported assembly type: {assembly_type}') from exc

    tube_universes = tube_locations(pin_map, materials)

    assembly_lattice = openmc.RectLattice(name='Fuel Assembly')
    assembly_lattice.pitch = (PIN_PITCH, PIN_PITCH)
    assembly_lattice.lower_left = (-0.5 * LATTICE_WIDTH, -0.5 * LATTICE_WIDTH)
    assembly_lattice.outer = openmc.Universe(cells=[
        openmc.Cell(fill=materials.mod)
    ])

    def lattice_universe(row, col):
        if (row, col) in tube_universes:
            return tube_universes[(row, col)]

        pin_code = pin_map[row][col]
        fuel_material_name = PIN_MAP_FUEL_MATERIALS[pin_code]
        # Only 1.71 wt% fuel has no axial fuel cutbacks.
        cutback_material = None if pin_code == 0 else materials.uo2_200
        return fuel_pin_universe(
            materials,
            fuel_material=getattr(materials, fuel_material_name),
            cutback_material=cutback_material,
            n_radial=N_RADIAL)

    assembly_lattice.universes = [
        [
            lattice_universe(row, col)
            for col in range(PINS_PER_SIDE)
        ]
        for row in range(PINS_PER_SIDE)
    ]
    assembly = openmc.Universe(cells=[
        openmc.Cell(fill=assembly_lattice)
    ])

    return assembly
