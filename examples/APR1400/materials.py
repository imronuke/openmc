"""Material definitions for the 3D assembly depletion example."""

from typing import NamedTuple

import openmc

from constants import (
    FUEL_CODE_PLOT_COLORS, FUEL_TEMPERATURE_K, GRIDMIX_BORON_PPM,
    GRIDMIX_TEMPERATURE_K, MATERIAL_TYPE_PLOT_COLORS, MOD_BORON_PPM,
    MOD_TEMPERATURE_K
)


class MaterialSet(NamedTuple):
    """Materials available to the assembly model."""

    uo2_171: openmc.Material
    uo2_200: openmc.Material
    uo2_264: openmc.Material
    uo2_314: openmc.Material
    uo2_364: openmc.Material
    gd: openmc.Material
    b4c: openmc.Material
    air: openmc.Material
    zirlo: openmc.Material
    inc625: openmc.Material
    gridmix: openmc.Material
    mod: openmc.Material
    ss304: openmc.Material
    spring: openmc.Material


def material_type_colors(model):
    """Return colors that group materials and distinguish fuel codes."""

    colors = {}
    for material in model.geometry.get_all_materials().values():
        fuel_code = getattr(material, '_plot_fuel_code', None)
        if fuel_code in FUEL_CODE_PLOT_COLORS:
            colors[material] = FUEL_CODE_PLOT_COLORS[fuel_code]
        else:
            color = MATERIAL_TYPE_PLOT_COLORS.get(material.name)
            if color is not None:
                colors[material] = color
    return colors


UO2_171 = (
    ('U235', 3.98302e-4),
    ('U238', 2.26050e-2),
    ('O16', 4.60066e-2),
)

UO2_200 = (
    ('U235', 4.65848e-4),
    ('U238', 2.25382e-2),
    ('O16', 4.60081e-2),
)

UO2_264 = (
    ('U235', 6.14913e-4),
    ('U238', 2.23908e-2),
    ('O16', 4.60115e-2),
)

UO2_314 = (
    ('U235', 7.31369e-4),
    ('U238', 2.22757e-2),
    ('O16', 4.60141e-2),
)

UO2_364 = (
    ('U235', 8.47822e-4),
    ('U238', 2.21605e-2),
    ('O16', 4.60167e-2),
)

GD = (
    ('U235', 4.12776e-4),
    ('U238', 1.99705e-2),
    ('O16', 4.47270e-2),
    ('Gd152', 5.35705e-6),
    ('Gd154', 5.75810e-5),
    ('Gd155', 3.90750e-4),
    ('Gd156', 5.40362e-4),
    ('Gd157', 4.13245e-4),
    ('Gd158', 6.55696e-4),
    ('Gd160', 5.77249e-4),
)

B4C = (
    ('B10', 1.58942e-2),
    ('B11', 6.42987e-2),
    ('C12', 1.98261e-2),
    ('C13', 2.22094e-4),
)

AIR = (
    ('N14', 3.29892e-5),
    ('O16', 8.76928e-6),
)

ZIRLO = (
    ('Fe54', 8.16874e-6),
    ('Fe56', 1.28232e-4),
    ('Fe57', 2.96157e-6),
    ('Fe58', 3.93972e-7),
    ('Zr90', 2.15258e-2),
    ('Zr91', 4.69534e-3),
    ('Zr92', 7.17333e-3),
    ('Zr94', 7.27123e-3),
    ('Zr96', 1.17101e-3),
    ('Nb93', 4.20030e-4),
    ('Sn112', 3.19853e-6),
    ('Sn114', 2.16632e-6),
    ('Sn115', 1.11439e-6),
    ('Sn116', 4.77840e-5),
    ('Sn117', 2.52332e-5),
    ('Sn118', 7.96280e-5),
    ('Sn119', 2.82214e-5),
    ('Sn120', 1.07143e-4),
    ('Sn122', 1.52169e-5),
    ('Sn124', 1.90301e-5),
)

INC625 = (
    ('Si28', 4.17375e-4),
    ('Si29', 2.11931e-5),
    ('Si30', 1.39706e-5),
    ('Cr50', 9.13426e-4),
    ('Cr52', 1.76138e-2),
    ('Cr53', 1.99717e-3),
    ('Cr54', 4.97095e-4),
    ('Mn55', 2.31347e-4),
    ('Fe54', 1.33026e-4),
    ('Fe56', 2.08822e-3),
    ('Fe57', 4.82283e-5),
    ('Fe58', 6.41573e-6),
    ('Co59', 4.31327e-4),
    ('Ni58', 3.69132e-2),
    ('Ni60', 1.42189e-2),
    ('Ni61', 6.18085e-4),
    ('Ni62', 1.97073e-3),
    ('Ni64', 5.01886e-4),
    ('Nb93', 1.99730e-3),
    ('Mo92', 6.98566e-4),
    ('Mo94', 4.38100e-4),
    ('Mo95', 7.56934e-4),
    ('Mo96', 7.95084e-4),
    ('Mo97', 4.56936e-4),
    ('Mo98', 1.15841e-3),
    ('Mo100', 4.64661e-4),
)

SS304 = (
    ('Cr50', 7.71070e-4),
    ('Cr52', 1.48687e-2),
    ('Cr53', 1.68591e-3),
    ('Cr54', 4.19624e-4),
    ('Mn55', 1.76615e-3),
    ('Fe54', 3.54740e-3),
    ('Fe56', 5.56866e-2),
    ('Fe57', 1.28611e-3),
    ('Fe58', 1.71088e-4),
    ('Ni58', 4.74025e-3),
    ('Ni60', 1.82594e-3),
    ('Ni61', 7.93722e-5),
    ('Ni62', 2.53073e-4),
    ('Ni64', 6.44503e-5),
)

SPRING = (
    ('Cr50', 2.89151e-4),
    ('Cr52', 5.57577e-3),
    ('Cr53', 6.32217e-4),
    ('Cr54', 1.57359e-4),
    ('Mn55', 6.62306e-4),
    ('Fe54', 1.33028e-3),
    ('Fe56', 2.08825e-2),
    ('Fe57', 4.82290e-4),
    ('Fe58', 6.41582e-5),
    ('Ni58', 1.77759e-3),
    ('Ni60', 6.84726e-4),
    ('Ni61', 2.97646e-5),
    ('Ni62', 9.49024e-5),
    ('Ni64', 2.41689e-5),
)

GRIDMIX_H1 = {
    300: 5.74660e-2,
    600: 3.78675e-2,
}

GRIDMIX_O16 = {
    300: 2.87330e-2,
    600: 1.89338e-2,
}

GRIDMIX_BORON = {
    300: {
        0: (
            ('B10', 0.0),
            ('B11', 0.0),
        ),
        1000: (
            ('B10', 9.48664e-6),
            ('B11', 3.83774e-5),
        ),
        2000: (
            ('B10', 1.89733e-5),
            ('B11', 7.67547e-5),
        ),
    },
    600: {
        0: (
            ('B10', 0.0),
            ('B11', 0.0),
        ),
        1000: (
            ('B10', 6.25128e-6),
            ('B11', 2.52890e-5),
        ),
        2000: (
            ('B10', 1.25026e-5),
            ('B11', 5.05780e-5),
        ),
    },
}

GRIDMIX_STRUCTURAL = (
    ('Fe54', 1.17265e-6),
    ('Fe56', 1.84081e-5),
    ('Fe57', 4.25143e-7),
    ('Fe58', 5.65560e-8),
    ('Zr90', 3.09011e-3),
    ('Zr91', 6.74031e-4),
    ('Zr92', 1.02976e-3),
    ('Zr94', 1.04381e-3),
    ('Zr96', 1.68102e-4),
    ('Nb93', 6.02967e-5),
    ('Sn112', 4.59160e-7),
    ('Sn114', 3.10983e-7),
    ('Sn115', 1.59975e-7),
    ('Sn116', 6.85956e-6),
    ('Sn117', 3.62231e-6),
    ('Sn118', 1.14309e-5),
    ('Sn119', 4.05127e-6),
    ('Sn120', 1.53807e-5),
    ('Sn122', 2.18443e-6),
    ('Sn124', 2.73184e-6),
)

MOD = {
    300: {
        0: (
            ('H1', 6.70981e-2),
            ('B10', 0.0),
            ('B11', 0.0),
            ('O16', 3.35491e-2),
        ),
        1000: (
            ('H1', 6.70981e-2),
            ('B10', 1.10767e-5),
            ('B11', 4.48100e-5),
            ('O16', 3.35491e-2),
        ),
        2000: (
            ('H1', 6.70981e-2),
            ('B10', 2.21535e-5),
            ('B11', 8.96199e-5),
            ('O16', 3.35491e-2),
        ),
    },
    600: {
        0: (
            ('H1', 4.42147e-2),
            ('B10', 0.0),
            ('B11', 0.0),
            ('O16', 2.21073e-2),
        ),
        1000: (
            ('H1', 4.42147e-2),
            ('B10', 7.29909e-6),
            ('B11', 2.95278e-5),
            ('O16', 2.21073e-2),
        ),
        2000: (
            ('H1', 4.42147e-2),
            ('B10', 1.45982e-5),
            ('B11', 5.90556e-5),
            ('O16', 2.21073e-2),
        ),
    },
}


def _atom_density_material(name, atom_densities):
    """Return a material using tabulated atom densities in atom/b-cm."""

    material = openmc.Material(name=name)
    material.set_density('sum')
    for nuclide, atom_density in atom_densities:
        material.add_nuclide(nuclide, atom_density)
    return material


def _fuel_material(name, atom_densities, fuel_code):
    """Return a fuel material with plot-only fuel code metadata."""

    material = _atom_density_material(name, atom_densities)
    material.temperature = FUEL_TEMPERATURE_K
    material._plot_fuel_code = fuel_code
    return material


def _gridmix():
    """Return the selected GRIDMIX material."""

    try:
        atom_densities = (
            ('H1', GRIDMIX_H1[GRIDMIX_TEMPERATURE_K]),
            *GRIDMIX_BORON[GRIDMIX_TEMPERATURE_K][GRIDMIX_BORON_PPM],
            ('O16', GRIDMIX_O16[GRIDMIX_TEMPERATURE_K]),
            *GRIDMIX_STRUCTURAL,
        )
    except KeyError as exc:
        raise ValueError(
            'Unsupported GRIDMIX temperature or boron ppm') from exc

    material = _atom_density_material('GRIDMIX', atom_densities)
    material.temperature = GRIDMIX_TEMPERATURE_K
    material.add_s_alpha_beta('c_H_in_H2O')
    return material


def _mod():
    """Return the selected MOD material."""

    try:
        atom_densities = MOD[MOD_TEMPERATURE_K][MOD_BORON_PPM]
    except KeyError as exc:
        raise ValueError(
            'Unsupported MOD temperature or boron ppm') from exc

    material = _atom_density_material('MOD', atom_densities)
    material.temperature = MOD_TEMPERATURE_K
    material.add_s_alpha_beta('c_H_in_H2O')
    return material


def create_materials():
    """Create the materials used by the assembly model."""

    return MaterialSet(
        uo2_171=_fuel_material('F', UO2_171, 0),
        uo2_200=_fuel_material('F', UO2_200, 'cutback'),
        uo2_264=_fuel_material('F', UO2_264, 1),
        uo2_314=_fuel_material('F', UO2_314, 2),
        uo2_364=_fuel_material('F', UO2_364, 3),
        gd=_fuel_material('GD', GD, 4),
        b4c=_atom_density_material('B4C', B4C),
        air=_atom_density_material('AIR', AIR),
        zirlo=_atom_density_material('ZIRLO', ZIRLO),
        inc625=_atom_density_material('INC625', INC625),
        gridmix=_gridmix(),
        mod=_mod(),
        ss304=_atom_density_material('SS304', SS304),
        spring=_atom_density_material('SPRING', SPRING),
    )
