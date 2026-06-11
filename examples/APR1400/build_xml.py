"""
Build and optionally deplete a single 16 x 16 3D fuel assembly.

The fuel in each pin is split into independent radial and axial depletion
zones, following the 3D pin depletion example.
"""

import argparse
from pathlib import Path

import openmc

from constants import (
    BATCHES, BURNUP_STEPS, INACTIVE, N_AXIAL, PARTICLES, PIN_PITCH,
    PINS_PER_SIDE, POWER_DENSITY, POWER_VOLUME
)
from core import core_geometry
from materials import create_materials, material_type_colors


def create_plots(model, color_mode):
    """Return material plots sized from the model bounds."""

    lower_left, upper_right = model.geometry.bounding_box
    x_min, y_min, z_min = lower_left
    x_max, y_max, z_max = upper_right
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    model_width = x_max - x_min
    model_depth = y_max - y_min
    model_height = z_max - z_min

    xy_plot = openmc.SlicePlot()
    xy_plot.basis = 'xy'
    xy_plot.origin = (x_center, y_center, 0.0)
    xy_plot.width = (model_width, model_depth)
    xy_plot.pixels = (1200, 1200)
    xy_plot.color_by = 'material'

    xz_plot = openmc.SlicePlot()
    xz_plot.basis = 'xz'
    xz_plot.origin = (
        x_center, y_center + 0.5 * PIN_PITCH, z_center)
    xz_plot.width = (model_width, model_height)
    xz_plot.pixels = (1200, 1800)
    xz_plot.color_by = 'material'

    if color_mode == 'material-type':
        colors = material_type_colors(model)
        xy_plot.colors = colors
        xz_plot.colors = colors

    return openmc.Plots([xy_plot, xz_plot])


def create_settings(bounds):
    """Return top-level OpenMC settings for the assembly model."""

    settings = openmc.Settings()
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.particles = PARTICLES

    uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
    settings.source = openmc.IndependentSource(
        space=uniform_dist, constraints={'fissionable': True})

    entropy_mesh = openmc.RegularMesh()
    entropy_mesh.lower_left = bounds[:3]
    entropy_mesh.upper_right = bounds[3:]
    entropy_mesh.dimension = [PINS_PER_SIDE, PINS_PER_SIDE, N_AXIAL]
    settings.entropy_mesh = entropy_mesh

    return settings


def create_model():
    """Return the top-level OpenMC model for the core."""

    materials = create_materials()
    geometry, bounds = core_geometry(materials=materials)
    settings = create_settings(bounds)
    return openmc.Model(geometry=geometry, settings=settings)


def run_depletion(model):
    """Run a one-step predictor depletion calculation."""

    import openmc.deplete

    chain_file = Path(__file__).with_name('chain_simple.xml')
    op = openmc.deplete.CoupledOperator(model, str(chain_file))

    power = POWER_DENSITY * POWER_VOLUME
    integrator = openmc.deplete.PredictorIntegrator(
        op, BURNUP_STEPS, power, timestep_units='MWd/kg')
    integrator.integrate(write_rates=True)


def run_normal(model):
    """Run a single criticality calculation without depletion."""

    model.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate', action='store_true',
        help='export OpenMC XML input files without running')
    parser.add_argument(
        '--run', choices=('normal', 'depletion'),
        help='run a normal criticality or depletion calculation')
    parser.add_argument(
        '--plot-color-mode',
        choices=('depletion-zone', 'material-type'),
        help='include xy and xz plots using the selected material color mode')
    args = parser.parse_args()

    model = create_model()
    if args.plot_color_mode:
        model.plots = create_plots(model, args.plot_color_mode)
    if args.generate or args.plot_color_mode or args.run is None:
        model.export_to_xml()
    if args.run == 'normal':
        run_normal(model)
    elif args.run == 'depletion':
        run_depletion(model)
