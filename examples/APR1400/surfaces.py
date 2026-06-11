"""Shared surface builders for the APR1400 depletion example."""

from math import pi, sqrt
from typing import NamedTuple

import openmc

from constants import (
    ACTIVE_AXIAL_HEIGHTS, ACTIVE_BOTTOM_Z, ACTIVE_TOP_Z, CLAD_OR, FUEL_RAD,
    GAP_OR, N_RADIAL
)


class FuelPinSurfaces(NamedTuple):
    """Surfaces and areas used by one fuel pin type."""

    fuel_ors: tuple
    gap_or: openmc.ZCylinder
    clad_or: openmc.ZCylinder
    fuel_axial_surfaces: tuple
    fuel_areas: tuple


class TubeQuadrantSurfaces(NamedTuple):
    """Surfaces used by one tube quadrant type."""

    mod_or: openmc.ZCylinder
    clad_or: openmc.ZCylinder


_fuel_pin_surfaces = {}
_tube_quadrant_surfaces = {}


def reset_surfaces():
    """Clear cached surface objects."""

    _fuel_pin_surfaces.clear()
    _tube_quadrant_surfaces.clear()


def get_fuel_pin_surfaces(n_radial=N_RADIAL):
    """Return shared fuel pin surfaces for a radial mesh."""

    if n_radial not in _fuel_pin_surfaces:
        _fuel_pin_surfaces[n_radial] = create_fuel_pin_surfaces(n_radial)
    return _fuel_pin_surfaces[n_radial]


def get_tube_quadrant_surfaces(x0, y0, mod_or, clad_or):
    """Return shared tube surfaces for one quadrant offset."""

    key = (x0, y0, mod_or, clad_or)
    if key not in _tube_quadrant_surfaces:
        _tube_quadrant_surfaces[key] = create_tube_quadrant_surfaces(
            x0, y0, mod_or, clad_or)
    return _tube_quadrant_surfaces[key]


def create_fuel_surfaces(fuel_rad, n_radial):
    """Create equal-area radial fuel zones."""

    fuel_area = pi * fuel_rad**2
    fuel_zone_area = fuel_area / n_radial
    fuel_areas = (fuel_zone_area,) * n_radial
    fuel_ors = tuple(
        openmc.ZCylinder(r=sqrt(fuel_zone_area * (i + 1) / pi))
        for i in range(n_radial)
    )
    return fuel_ors, fuel_areas


def create_fuel_pin_radial_surfaces(n_radial=N_RADIAL):
    """Create regular fuel pin radial surfaces and zone areas."""

    fuel_ors, fuel_areas = create_fuel_surfaces(FUEL_RAD, n_radial)
    gap_or = openmc.ZCylinder(r=GAP_OR)
    clad_or = openmc.ZCylinder(r=CLAD_OR)
    return fuel_ors, gap_or, clad_or, fuel_areas


def create_active_axial_surfaces(active_bottom, active_top):
    """Return active fuel axial surfaces from the explicit height mesh."""

    surfaces = [active_bottom]
    z0 = active_bottom.z0
    for height in ACTIVE_AXIAL_HEIGHTS[:-1]:
        z0 += height
        surfaces.append(openmc.ZPlane(z0=z0))
    surfaces.append(active_top)
    return tuple(surfaces)


def create_fuel_pin_surfaces(n_radial=N_RADIAL):
    """Create regular pin radial and active axial surfaces."""

    fuel_bottom = openmc.ZPlane(z0=ACTIVE_BOTTOM_Z)
    fuel_top = openmc.ZPlane(z0=ACTIVE_TOP_Z)
    fuel_axial_surfaces = create_active_axial_surfaces(fuel_bottom, fuel_top)

    fuel_ors, gap_or, clad_or, fuel_areas = (
        create_fuel_pin_radial_surfaces(n_radial))
    return FuelPinSurfaces(
        fuel_ors, gap_or, clad_or, fuel_axial_surfaces, fuel_areas)


def create_tube_quadrant_surfaces(x0, y0, mod_or, clad_or):
    """Create surfaces for one tube quadrant offset."""

    return TubeQuadrantSurfaces(
        mod_or=openmc.ZCylinder(x0=x0, y0=y0, r=mod_or),
        clad_or=openmc.ZCylinder(x0=x0, y0=y0, r=clad_or))
