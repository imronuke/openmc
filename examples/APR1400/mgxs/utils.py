"""Shared utilities for APR1400 MGXS generation."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
import warnings

import openmc
import openmc.mgxs
from openmc.mixin import IDWarning

from .mgxs_constants import (
    BATCHES, INACTIVE, LEGENDRE_ORDER, MGXS_METADATA_TEMPERATURE_K,
    MGXS_TYPES, OPENMC_EXECUTABLE, PARTICLES, WIMS_69_GROUP_EDGES
)


@dataclass(frozen=True)
class OutputSpec:
    """One OpenSn-loadable MGXS export from a model domain."""

    domain: object
    dataset_name: str
    subdir: str = ''


@dataclass(frozen=True)
class CaseSpec:
    """A complete MGXS case definition."""

    case_name: str
    model: openmc.Model
    library: openmc.mgxs.Library
    outputs: tuple[OutputSpec, ...]


def create_settings(bounds, entropy_shape, source_bounds=None):
    """Return OpenMC settings for one MGXS model."""

    settings = openmc.Settings()
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.particles = PARTICLES

    if source_bounds is None:
        source_bounds = bounds

    uniform_dist = openmc.stats.Box(source_bounds[:3], source_bounds[3:])
    settings.source = openmc.IndependentSource(
        space=uniform_dist, constraints={'fissionable': True})

    if entropy_shape is not None:
        entropy_mesh = openmc.RegularMesh()
        entropy_mesh.lower_left = bounds[:3]
        entropy_mesh.upper_right = bounds[3:]
        entropy_mesh.dimension = entropy_shape
        settings.entropy_mesh = entropy_mesh

    return settings


def create_mgxs_library(geometry, domains, domain_type, name):
    """Return a loaded-independent MGXS library definition."""

    energy_groups = openmc.mgxs.EnergyGroups(WIMS_69_GROUP_EDGES)
    mgxs_lib = openmc.mgxs.Library(geometry, name=name)
    mgxs_lib.by_nuclide = False
    mgxs_lib.mgxs_types = MGXS_TYPES
    mgxs_lib.energy_groups = energy_groups
    mgxs_lib.correction = None
    mgxs_lib.legendre_order = LEGENDRE_ORDER
    mgxs_lib.domain_type = domain_type
    mgxs_lib.domains = list(domains)
    mgxs_lib.build_library()
    return mgxs_lib


def make_case(case_name, geometry, settings, library, outputs):
    """Create a case definition and attach its tallies to a model."""

    model = openmc.Model(geometry=geometry, settings=settings)
    if model.tallies is None:
        model.tallies = openmc.Tallies()
    library.add_to_tallies(model.tallies, merge=False)
    return CaseSpec(case_name, model, library, tuple(outputs))


def run_openmc_model(model, output_dir):
    """Export XML, run OpenMC, and return the final statepoint path."""

    if not OPENMC_EXECUTABLE.exists():
        raise FileNotFoundError(
            f'Unable to find OpenMC executable at {OPENMC_EXECUTABLE}')

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Exporting OpenMC XML files in {output_dir}', flush=True)

    with NamedTemporaryFile() as fp:
        tstart = Path(fp.name).stat().st_mtime

    # OpenMC's tally XML export deduplicates identical filters by reassigning
    # IDs, which can emit harmless IDWarning messages for the filter objects.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'Another Filter instance already exists with id=.*',
            category=IDWarning
        )
        model.export_to_xml(output_dir)

    print(f'Launching OpenMC in {output_dir}', flush=True)
    openmc.run(
        cwd=output_dir,
        openmc_exec=str(OPENMC_EXECUTABLE),
        output=True
    )
    print(f'OpenMC completed in {output_dir}', flush=True)

    statepoint_path = None
    for candidate in output_dir.glob('statepoint.*.h5'):
        if candidate.stat().st_mtime >= tstart:
            if statepoint_path is None or (
                    candidate.stat().st_mtime
                    >= statepoint_path.stat().st_mtime):
                statepoint_path = candidate

    if statepoint_path is None:
        raise RuntimeError('OpenMC did not produce a statepoint file.')

    return statepoint_path


def export_domain_hdf5(library, output_spec, case_dir):
    """Write one OpenSn-loadable MGXS library for a domain."""

    xsdata = library.get_xsdata(
        output_spec.domain,
        output_spec.dataset_name,
        xs_type='macro',
        temperature=MGXS_METADATA_TEMPERATURE_K
    )
    xsdata.temperatures = [MGXS_METADATA_TEMPERATURE_K]

    mgxs_file = openmc.MGXSLibrary(
        library.energy_groups, num_delayed_groups=library.num_delayed_groups)
    mgxs_file.add_xsdata(xsdata)

    output_dir = case_dir / output_spec.subdir if output_spec.subdir else case_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'mgxs.h5'
    mgxs_file.export_to_hdf5(output_path)
    return output_path


def run_case(case_spec, output_root):
    """Run one case and export all requested MGXS libraries."""

    case_dir = output_root / case_spec.case_name
    statepoint_path = run_openmc_model(case_spec.model, case_dir)

    with openmc.StatePoint(statepoint_path) as statepoint:
        case_spec.library.load_from_statepoint(statepoint)

    exported_paths = [
        export_domain_hdf5(case_spec.library, output_spec, case_dir)
        for output_spec in case_spec.outputs
    ]

    return case_dir, statepoint_path, tuple(exported_paths)
