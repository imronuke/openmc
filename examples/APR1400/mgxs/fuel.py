"""Fuel-adjacent MGXS case builders."""

import openmc

from assemblies import tube_locations
from constants import (
    ASSEMBLY_PIN_MAPS, ASSEMBLY_WIDTH, CLAD_OR, FUEL_RAD, GAP_OR,
    LATTICE_WIDTH, PIN_MAP_FUEL_MATERIALS, PIN_PITCH, PINS_PER_SIDE
)
from materials import create_materials

from .mgxs_constants import MODEL_HEIGHT
from .utils import OutputSpec, create_mgxs_library, create_settings, make_case


# Fuel MGXS libraries use higher soluble boron to make downstream fixed-source
# calculations more subcritical. Non-fuel cases keep create_materials() default.
FUEL_MGXS_BORON_PPM = 2000


def fuel_pin_universe_2d(materials, fuel_material):
    """Return a 2D fuel pin universe with fuel, gap, clad, and gridmix."""

    fuel_or = openmc.ZCylinder(r=FUEL_RAD)
    gap_or = openmc.ZCylinder(r=GAP_OR)
    clad_or = openmc.ZCylinder(r=CLAD_OR)

    cells = [
        openmc.Cell(fill=fuel_material, region=-fuel_or),
        openmc.Cell(fill=materials.air, region=+fuel_or & -gap_or),
        openmc.Cell(fill=materials.zirlo, region=+gap_or & -clad_or),
        openmc.Cell(fill=materials.gridmix, region=+clad_or),
    ]

    return openmc.Universe(cells=cells)


def fuel_assembly_2d(assembly_type, materials, fuel_selector=None):
    """Return a 2D fuel assembly universe."""

    try:
        pin_map = ASSEMBLY_PIN_MAPS[assembly_type]
    except KeyError as exc:
        raise ValueError(f'Unsupported assembly type: {assembly_type}') from exc

    if fuel_selector is None:
        fuel_selector = lambda pin_code: getattr(  # noqa: E731
            materials, PIN_MAP_FUEL_MATERIALS[pin_code])

    tube_universes = tube_locations(pin_map, materials)
    fuel_universes = {}

    assembly_lattice = openmc.RectLattice(name=f'{assembly_type} Fuel Assembly')
    assembly_lattice.pitch = (PIN_PITCH, PIN_PITCH)
    assembly_lattice.lower_left = (-0.5 * LATTICE_WIDTH, -0.5 * LATTICE_WIDTH)
    assembly_lattice.outer = openmc.Universe(cells=[
        openmc.Cell(fill=materials.mod)
    ])

    def lattice_universe(row, col):
        if (row, col) in tube_universes:
            return tube_universes[(row, col)]

        pin_code = pin_map[row][col]
        fuel_material = fuel_selector(pin_code)
        cache_key = (pin_code, fuel_material.id)
        if cache_key not in fuel_universes:
            fuel_universes[cache_key] = fuel_pin_universe_2d(
                materials, fuel_material)
        return fuel_universes[cache_key]

    assembly_lattice.universes = [
        [
            lattice_universe(row, col)
            for col in range(PINS_PER_SIDE)
        ]
        for row in range(PINS_PER_SIDE)
    ]

    assembly = openmc.Universe(name=f'{assembly_type} 2D Fuel Assembly')
    assembly.add_cell(openmc.Cell(fill=assembly_lattice))
    assembly.volume = ASSEMBLY_WIDTH * ASSEMBLY_WIDTH * MODEL_HEIGHT
    return assembly


def reflected_assembly_geometry(fill, name):
    """Return a reflected full-assembly geometry and model bounds."""

    outer_boundary = openmc.model.RectangularPrism(
        ASSEMBLY_WIDTH, ASSEMBLY_WIDTH, boundary_type='reflective')
    bottom = openmc.ZPlane(
        z0=-0.5 * MODEL_HEIGHT, boundary_type='reflective')
    top = openmc.ZPlane(z0=0.5 * MODEL_HEIGHT, boundary_type='reflective')

    root_cell = openmc.Cell(
        name=name,
        fill=fill,
        region=-outer_boundary & +bottom & -top)

    bounds = [
        -0.5 * ASSEMBLY_WIDTH, -0.5 * ASSEMBLY_WIDTH, -0.5 * MODEL_HEIGHT,
        0.5 * ASSEMBLY_WIDTH, 0.5 * ASSEMBLY_WIDTH, 0.5 * MODEL_HEIGHT
    ]

    return openmc.Geometry([root_cell]), bounds


def build_fuel_case(assembly_type):
    """Return the standard reflected fuel-assembly MGXS case."""

    materials = create_materials(ppm=FUEL_MGXS_BORON_PPM)
    assembly = fuel_assembly_2d(assembly_type, materials)
    geometry, bounds = reflected_assembly_geometry(
        assembly, f'{assembly_type} 2D reflected fuel assembly')
    settings = create_settings(bounds, [PINS_PER_SIDE, PINS_PER_SIDE, 1])
    library = create_mgxs_library(
        geometry, [assembly], 'universe', f'{assembly_type} fuel assembly mgxs')
    return make_case(
        assembly_type,
        geometry,
        settings,
        library,
        [OutputSpec(assembly, assembly_type)]
    )


def build_cutback_case():
    """Return the cutback assembly MGXS case."""

    materials = create_materials(ppm=FUEL_MGXS_BORON_PPM)
    assembly = fuel_assembly_2d(
        'A0',
        materials,
        fuel_selector=lambda _pin_code: materials.uo2_200)
    geometry, bounds = reflected_assembly_geometry(
        assembly, 'CB 2D reflected cutback assembly')
    settings = create_settings(bounds, [PINS_PER_SIDE, PINS_PER_SIDE, 1])
    library = create_mgxs_library(
        geometry, [assembly], 'universe', 'CB fuel assembly mgxs')
    return make_case(
        'CB',
        geometry,
        settings,
        library,
        [OutputSpec(assembly, 'CB')]
    )
