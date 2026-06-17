"""Generate APR1400 fuel and non-fuel MGXS inputs."""

import argparse

from constants import ASSEMBLY_PIN_MAPS
from mgxs.fuel import build_cutback_case, build_fuel_case
from mgxs.mgxs_constants import OUTPUT_ROOT
from mgxs.non_fuel_assembly import (
    build_bottom_reflector_case, build_radial_reflector_case,
    build_top_reflector_case
)
from mgxs.utils import run_case


def parse_args():
    """Parse command line arguments for MGXS generation."""

    parser = argparse.ArgumentParser(
        description='Generate APR1400 OpenSn-loadable MGXS libraries.')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        '--assembly',
        help='fuel assembly type to generate, for example A0')
    target.add_argument(
        '--case',
        choices=('CB', 'RR', 'BR', 'TR'),
        help='special non-fuel MGXS case to generate')
    args = parser.parse_args()

    if args.assembly is not None:
        assembly_type = args.assembly.upper()
        if assembly_type not in ASSEMBLY_PIN_MAPS:
            valid = ', '.join(sorted(ASSEMBLY_PIN_MAPS))
            parser.error(
                f'unsupported assembly {args.assembly!r}; choose {valid}')
        return ('assembly', assembly_type)

    return ('case', args.case)


def main():
    """Generate the requested MGXS files."""

    target_kind, target_value = parse_args()
    print(f'Building MGXS model for {target_kind} {target_value}', flush=True)

    if target_kind == 'assembly':
        case_spec = build_fuel_case(target_value)
    elif target_value == 'CB':
        case_spec = build_cutback_case()
    elif target_value == 'RR':
        case_spec = build_radial_reflector_case()
    elif target_value == 'BR':
        case_spec = build_bottom_reflector_case()
    else:
        case_spec = build_top_reflector_case()

    print(f'Running OpenMC for {case_spec.case_name}', flush=True)
    case_dir, _statepoint_path, exported_paths = run_case(case_spec, OUTPUT_ROOT)
    print(f'Generated {case_spec.case_name} MGXS files in {case_dir}', flush=True)
    for path in exported_paths:
        print(f'Generated MGXS library {path}', flush=True)


if __name__ == '__main__':
    main()
