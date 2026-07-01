"""Non-fuel MGXS case builders."""

import openmc

from assemblies import fuel_assembly
from constants import (
    ACTIVE_BOTTOM_Z, ACTIVE_TOP_Z, ASSEMBLY_WIDTH, BOTTOM_REFLECTOR_HEIGHT,
    N_AXIAL, PINS_PER_SIDE, STACK_BOTTOM_Z, STACK_TOP_Z, TOP_REFLECTOR_HEIGHT
)
from materials import create_materials

from .fuel import fuel_assembly_2d
from .mgxs_constants import (
    BAFFLE, BAFFLE_GAP, DET_INNER, DET_WALL, MODEL_HEIGHT, RR_AIR_THICKNESS,
    RR_RPV_THICKNESS, U_THICK
)
from .utils import OutputSpec, create_mgxs_library, create_settings, make_case


def _outer_boundary():
    return openmc.model.RectangularPrism(
        ASSEMBLY_WIDTH, ASSEMBLY_WIDTH, boundary_type='reflective')
    

def build_radial_reflector_case():
    """Return the radial reflector MGXS case."""

    materials = create_materials()
    assembly = fuel_assembly_2d('A0', materials)

    x_inner = openmc.XPlane(x0=0.0, boundary_type='reflective')
    x = 0.5 * ASSEMBLY_WIDTH
    x_fa_outer = openmc.XPlane(x0=x)
    x += BAFFLE_GAP
    x_baffle = openmc.XPlane(x0=x)
    x += BAFFLE
    x_baffle_out = openmc.XPlane(x0=x)
    x += ASSEMBLY_WIDTH - BAFFLE - BAFFLE_GAP
    x_reflector_outer = openmc.XPlane(x0=x, boundary_type='vacuum')
    y_min = openmc.YPlane(y0=0.0, boundary_type='reflective')
    y_max = openmc.YPlane(
        y0=0.5 * ASSEMBLY_WIDTH, boundary_type='reflective')
    bottom = openmc.ZPlane(z0=-0.5 * MODEL_HEIGHT, boundary_type='reflective')
    top = openmc.ZPlane(z0=0.5 * MODEL_HEIGHT, boundary_type='reflective')

    base_region = +y_min & -y_max & +bottom & -top

    fa_cell = openmc.Cell(
        name='Quarter A0 fuel assembly',
        fill=assembly,
        region=+x_inner & -x_fa_outer & base_region)

    baffle_gap = openmc.Cell(
        name='Baffle gap',
        fill=materials.mod,
        region=+x_fa_outer & -x_baffle & base_region)
    baffle = openmc.Cell(
        name='Baffle',
        fill=materials.ss304,
        region=+x_baffle & -x_baffle_out & base_region)
    reflector_cell = openmc.Cell(
        name='Radial reflector',
        fill=materials.mod,
        region=+x_baffle_out & -x_reflector_outer & base_region)
    radial_reflector_universe = openmc.Universe(
        name='Radial reflector homogenization domain',
        cells=[baffle_gap, baffle, reflector_cell])
    radial_reflector_cell = openmc.Cell(
        name='Radial reflector domain',
        fill=radial_reflector_universe,
        region=+x_fa_outer & -x_reflector_outer & base_region)

    bounds = [
        0.0,
        0.0,
        -0.5 * MODEL_HEIGHT,
        x_reflector_outer.x0,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * MODEL_HEIGHT
    ]
    source_bounds = [
        0.0,
        0.0,
        -0.5 * MODEL_HEIGHT,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * MODEL_HEIGHT
    ]

    geometry = openmc.Geometry([
        fa_cell, radial_reflector_cell
    ])
    settings = create_settings(
        bounds,
        [PINS_PER_SIDE // 2, PINS_PER_SIDE // 2, 1],
        source_bounds=source_bounds
    )
    library = create_mgxs_library(
        geometry,
        [
            radial_reflector_universe
        ],
        'universe',
        'RR non-fuel mgxs'
    )
    case = make_case(
        'RR',
        geometry,
        settings,
        library,
        [
            OutputSpec(
                radial_reflector_universe, 'radial_reflector',
                'radial_reflector'),
        ]
    )
    
    return case

def build_rpv_case():
    """Return the radial reflector MGXS case."""

    materials = create_materials()
    assembly = fuel_assembly_2d('A0', materials)

    x_inner = openmc.XPlane(x0=0.0, boundary_type='reflective')
    x = 0.5 * ASSEMBLY_WIDTH
    x_fa_outer = openmc.XPlane(x0=x)
    x += RR_RPV_THICKNESS
    x_rpv_outer = openmc.XPlane(x0=x, boundary_type='vacuum')
    y_min = openmc.YPlane(y0=0.0, boundary_type='reflective')
    y_max = openmc.YPlane(
        y0=0.5 * ASSEMBLY_WIDTH, boundary_type='reflective')
    bottom = openmc.ZPlane(z0=-0.5 * MODEL_HEIGHT, boundary_type='reflective')
    top = openmc.ZPlane(z0=0.5 * MODEL_HEIGHT, boundary_type='reflective')

    base_region = +y_min & -y_max & +bottom & -top

    fa_cell = openmc.Cell(
        name='Quarter A0 fuel assembly',
        fill=assembly,
        region=+x_inner & -x_fa_outer & base_region)

    rpv_universe = openmc.Universe(cells=[
        openmc.Cell(name='RPV material', fill=materials.rpv)
    ])
    rpv_cell = openmc.Cell(
        name='RPV approximation',
        fill=rpv_universe,
        region=+x_fa_outer & -x_rpv_outer & base_region)

    bounds = [
        0.0,
        0.0,
        -0.5 * MODEL_HEIGHT,
        x_rpv_outer.x0,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * MODEL_HEIGHT
    ]
    source_bounds = [
        0.0,
        0.0,
        -0.5 * MODEL_HEIGHT,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * MODEL_HEIGHT
    ]

    geometry = openmc.Geometry([
        fa_cell, rpv_cell
    ])
    settings = create_settings(
        bounds,
        [PINS_PER_SIDE // 2, PINS_PER_SIDE // 2, 1],
        source_bounds=source_bounds
    )
    library = create_mgxs_library(
        geometry,
        [rpv_universe],
        'universe',
        'RR non-fuel mgxs'
    )
    case = make_case(
        'RR',
        geometry,
        settings,
        library,
        [OutputSpec(rpv_universe, 'rpv', 'rpv')]
    )
    
    return case

def build_air_case():
    """Return the radial reflector MGXS case."""

    materials = create_materials()
    assembly = fuel_assembly_2d('A0', materials)

    x_inner = openmc.XPlane(x0=0.0, boundary_type='reflective')
    x = 0.5 * ASSEMBLY_WIDTH
    x_fa_outer = openmc.XPlane(x0=x)
    x += RR_AIR_THICKNESS
    x_air_outer = openmc.XPlane(x0=x, boundary_type='vacuum')
    y_min = openmc.YPlane(y0=0.0, boundary_type='reflective')
    y_max = openmc.YPlane(
        y0=0.5 * ASSEMBLY_WIDTH, boundary_type='reflective')
    bottom = openmc.ZPlane(z0=-0.5 * MODEL_HEIGHT, boundary_type='reflective')
    top = openmc.ZPlane(z0=0.5 * MODEL_HEIGHT, boundary_type='reflective')

    base_region = +y_min & -y_max & +bottom & -top

    fa_cell = openmc.Cell(
        name='Quarter A0 fuel assembly',
        fill=assembly,
        region=+x_inner & -x_fa_outer & base_region)

    air_universe = openmc.Universe(cells=[
        openmc.Cell(name='Air material', fill=materials.air)
    ])
    air_cell = openmc.Cell(
        name='Air gap',
        fill=air_universe,
        region=+x_fa_outer & -x_air_outer & base_region)


    bounds = [
        0.0,
        0.0,
        -0.5 * MODEL_HEIGHT,
        x_air_outer.x0,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * MODEL_HEIGHT
    ]
    source_bounds = [
        0.0,
        0.0,
        -0.5 * MODEL_HEIGHT,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * MODEL_HEIGHT
    ]

    geometry = openmc.Geometry([
        fa_cell, air_cell
    ])
    settings = create_settings(
        bounds,
        [PINS_PER_SIDE // 2, PINS_PER_SIDE // 2, 1],
        source_bounds=source_bounds
    )
    library = create_mgxs_library(
        geometry,
        [air_universe],
        'universe',
        'RR non-fuel mgxs'
    )
    case = make_case(
        'RR',
        geometry,
        settings,
        library,
        [OutputSpec(air_universe, 'air', 'air')]
    )

    return case

def build_detector_case():
    """Return the radial reflector MGXS case."""

    materials = create_materials()
    assembly = fuel_assembly_2d('A0', materials)

    x_inner = openmc.XPlane(x0=0.0, boundary_type='reflective')
    x = 0.5 * ASSEMBLY_WIDTH
    x_fa_outer = openmc.XPlane(x0=x)
    x += DET_WALL
    x_det_left = openmc.XPlane(x0=x)
    x += U_THICK
    x_det_coat_left = openmc.XPlane(x0=x)
    x += DET_INNER
    x_det_inner = openmc.XPlane(x0=x)
    x += U_THICK
    x_det_coat_right = openmc.XPlane(x0=x)
    x += DET_WALL
    x_detector_outer = openmc.XPlane(x0=x, boundary_type='vacuum')
    y_min = openmc.YPlane(y0=0.0, boundary_type='reflective')
    y_max = openmc.YPlane(
        y0=0.5 * ASSEMBLY_WIDTH, boundary_type='reflective')
    bottom = openmc.ZPlane(z0=-0.5 * MODEL_HEIGHT, boundary_type='reflective')
    top = openmc.ZPlane(z0=0.5 * MODEL_HEIGHT, boundary_type='reflective')

    base_region = +y_min & -y_max & +bottom & -top

    fa_cell = openmc.Cell(
        name='Quarter A0 fuel assembly',
        fill=assembly,
        region=+x_inner & -x_fa_outer & base_region)

    det_wall_left = openmc.Cell(
        name='Detector left wall',
        fill=materials.ss304,
        region=+x_fa_outer & -x_det_left & base_region)
    det_coat_left = openmc.Cell(
        name='Detector left coating',
        fill=materials.detector,
        region=+x_det_left & -x_det_coat_left & base_region)
    det_inner = openmc.Cell(
        name='Detector inner gas',
        fill=materials.air,
        region=+x_det_coat_left & -x_det_inner & base_region)
    det_coat_right = openmc.Cell(
        name='Detector right coating',
        fill=materials.detector,
        region=+x_det_inner & -x_det_coat_right & base_region)
    det_wall_right = openmc.Cell(
        name='Detector right wall',
        fill=materials.ss304,
        region=+x_det_coat_right & -x_detector_outer & base_region)
    detector_universe = openmc.Universe(
        name='Detector homogenization domain',
        cells=[
            det_wall_left, det_coat_left, det_inner, det_coat_right,
            det_wall_right
        ])
    detector_cell = openmc.Cell(
        name='Detector domain',
        fill=detector_universe,
        region=+x_fa_outer & -x_detector_outer & base_region)

    bounds = [
        0.0,
        0.0,
        -0.5 * MODEL_HEIGHT,
        x_detector_outer.x0,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * MODEL_HEIGHT
    ]
    source_bounds = [
        0.0,
        0.0,
        -0.5 * MODEL_HEIGHT,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * MODEL_HEIGHT
    ]

    geometry = openmc.Geometry([
        fa_cell, detector_cell
    ])
    settings = create_settings(
        bounds,
        [PINS_PER_SIDE // 2, PINS_PER_SIDE // 2, 1],
        source_bounds=source_bounds
    )
    library = create_mgxs_library(
        geometry,
        [detector_universe],
        'universe',
        'RR non-fuel mgxs'
    )
    case = make_case(
        'RR',
        geometry,
        settings,
        library,
        [
            OutputSpec(detector_universe, 'detector', 'detector'),
        ]
    )
    
    return case

def build_bottom_reflector_case():
    """Return the bottom reflector MGXS case."""

    materials = create_materials()
    assembly = fuel_assembly('A0', materials)

    outer_boundary = _outer_boundary()
    model_bottom = openmc.ZPlane(
        z0=STACK_BOTTOM_Z - BOTTOM_REFLECTOR_HEIGHT, boundary_type='vacuum')
    stack_bottom = openmc.ZPlane(z0=STACK_BOTTOM_Z)
    active_bottom = openmc.ZPlane(z0=ACTIVE_BOTTOM_Z)
    model_top = openmc.ZPlane(z0=STACK_TOP_Z, boundary_type='reflective')

    bottom_reflector_cell = openmc.Cell(
        name='Bottom reflector',
        fill=materials.mod,
        region=-outer_boundary & +model_bottom & -stack_bottom)
    bottom_end_cell = openmc.Cell(
        name='Bottom end region',
        fill=assembly,
        region=-outer_boundary & +stack_bottom & -active_bottom)
    bottom_universe = openmc.Universe(
        name='Bottom reflector homogenization domain',
        cells=[bottom_reflector_cell, bottom_end_cell])
    bottom_domain_cell = openmc.Cell(
        name='Bottom reflector domain',
        fill=bottom_universe,
        region=-outer_boundary & +model_bottom & -active_bottom)
    upper_assembly_cell = openmc.Cell(
        name='Upper A0 assembly',
        fill=assembly,
        region=-outer_boundary & +active_bottom & -model_top)

    bounds = [
        -0.5 * ASSEMBLY_WIDTH,
        -0.5 * ASSEMBLY_WIDTH,
        model_bottom.z0,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        model_top.z0
    ]
    source_bounds = [
        -0.5 * ASSEMBLY_WIDTH,
        -0.5 * ASSEMBLY_WIDTH,
        ACTIVE_BOTTOM_Z,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        ACTIVE_TOP_Z
    ]

    geometry = openmc.Geometry([bottom_domain_cell, upper_assembly_cell])
    settings = create_settings(
        bounds,
        [PINS_PER_SIDE, PINS_PER_SIDE, N_AXIAL],
        source_bounds=source_bounds
    )
    library = create_mgxs_library(
        geometry,
        [bottom_universe],
        'universe',
        'BR non-fuel mgxs'
    )
    return make_case(
        'BR',
        geometry,
        settings,
        library,
        [OutputSpec(bottom_universe, 'bottom_reflector')]
    )


def build_top_end_case():
    """Return the top reflector MGXS case."""

    materials = create_materials()
    assembly = fuel_assembly('A0', materials)

    outer_boundary = _outer_boundary()
    model_bottom = openmc.ZPlane(z0=STACK_BOTTOM_Z, boundary_type='reflective')
    active_top = openmc.ZPlane(z0=ACTIVE_TOP_Z)
    stack_top = openmc.ZPlane(z0=STACK_TOP_Z, boundary_type='vacuum')

    lower_assembly_cell = openmc.Cell(
        name='Lower A0 assembly',
        fill=assembly,
        region=-outer_boundary & +model_bottom & -active_top)
    top_end_cell = openmc.Cell(
        name='Top end region',
        fill=assembly,
        region=-outer_boundary & +active_top & -stack_top)

    bounds = [
        -0.5 * ASSEMBLY_WIDTH,
        -0.5 * ASSEMBLY_WIDTH,
        model_bottom.z0,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        stack_top.z0
    ]
    source_bounds = [
        -0.5 * ASSEMBLY_WIDTH,
        -0.5 * ASSEMBLY_WIDTH,
        ACTIVE_BOTTOM_Z,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        ACTIVE_TOP_Z
    ]

    geometry = openmc.Geometry([
        lower_assembly_cell, top_end_cell
    ])
    settings = create_settings(
        bounds,
        [PINS_PER_SIDE, PINS_PER_SIDE, N_AXIAL],
        source_bounds=source_bounds
    )
    library = create_mgxs_library(
        geometry,
        [top_end_cell],
        'cell',
        'TR non-fuel mgxs'
    )
    return make_case(
        'TR',
        geometry,
        settings,
        library,
        [OutputSpec(top_end_cell, 'TR_top_end', 'top_end')]
    )

def build_top_reflector_case():
    """Return the top reflector MGXS case."""

    materials = create_materials()
    assembly = fuel_assembly('A0', materials)

    outer_boundary = _outer_boundary()
    model_bottom = openmc.ZPlane(z0=STACK_BOTTOM_Z, boundary_type='reflective')
    active_top = openmc.ZPlane(z0=ACTIVE_TOP_Z)
    model_top = openmc.ZPlane(
        z0=ACTIVE_TOP_Z + TOP_REFLECTOR_HEIGHT, boundary_type='vacuum')

    lower_assembly_cell = openmc.Cell(
        name='Lower A0 assembly',
        fill=assembly,
        region=-outer_boundary & +model_bottom & -active_top)
    top_reflector_cell = openmc.Cell(
        name='Top reflector',
        fill=materials.mod,
        region=-outer_boundary & +active_top & -model_top)

    bounds = [
        -0.5 * ASSEMBLY_WIDTH,
        -0.5 * ASSEMBLY_WIDTH,
        model_bottom.z0,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        model_top.z0
    ]
    source_bounds = [
        -0.5 * ASSEMBLY_WIDTH,
        -0.5 * ASSEMBLY_WIDTH,
        ACTIVE_BOTTOM_Z,
        0.5 * ASSEMBLY_WIDTH,
        0.5 * ASSEMBLY_WIDTH,
        ACTIVE_TOP_Z
    ]

    geometry = openmc.Geometry([
        lower_assembly_cell, top_reflector_cell
    ])
    settings = create_settings(
        bounds,
        [PINS_PER_SIDE, PINS_PER_SIDE, N_AXIAL],
        source_bounds=source_bounds
    )
    library = create_mgxs_library(
        geometry,
        [top_reflector_cell],
        'cell',
        'TR non-fuel mgxs'
    )
    return make_case(
        'TR',
        geometry,
        settings,
        library,
        [
            OutputSpec(top_reflector_cell, 'TR_reflector', 'top_reflector'),
        ]
    )
