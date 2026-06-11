# 3D Assembly Depletion Example

This example builds a 3D OpenMC depletion model for a single 16 x 16 fuel
assembly. The default core map contains one `C3` assembly with reflective
radial boundary conditions, moderator regions above and below the assembly
stack, and vacuum axial boundaries at the top and bottom of the model.

The input is generated from Python source files rather than hand-written XML.
Run commands from this directory.

## Generate Input Files

To export `geometry.xml`, `materials.xml`, and `settings.xml` without running
depletion:

```bash
python build_xml.py --generate
```

If no command-line options are given, `build_xml.py` also exports the XML input
files.

To include plot definitions:

```bash
python build_xml.py --plot-color-mode material-type
```

The plot color mode can be `material-type` or `depletion-zone`.

## Run Depletion

To run a single criticality calculation without depletion:

```bash
python build_xml.py --run=normal
```

To run the one-step predictor depletion calculation:

```bash
python build_xml.py --run=depletion
```

The depletion operator uses `chain_simple.xml` from this directory. The default
power is computed as:

```python
POWER_DENSITY * POWER_VOLUME
```

where `POWER_DENSITY` is in MW/m3 and `POWER_VOLUME` is the active volume of
all non-reflector assembly boxes in m3.

## Model Structure

The main source files are:

- `constants.py`: geometry dimensions, axial mesh, depletion settings, assembly
  maps, and plot colors.
- `materials.py`: material definitions using tabulated atom densities.
- `surfaces.py`: shared surface construction and caching.
- `pins.py`: fuel pin and tube quadrant universe builders.
- `assemblies.py`: 16 x 16 assembly lattice construction.
- `core.py`: top-level core lattice and axial reflector regions.
- `build_xml.py`: XML export, plot setup, settings, and depletion driver.

The active fuel height is defined by `ACTIVE_AXIAL_HEIGHTS`. Normal fuel pins
use `N_RADIAL` equal-area radial fuel zones, while gadolinium fuel pins use
`GD_N_RADIAL` zones. Each fuel radial and axial depletion zone receives an
independent cloned material with a volume assigned from its ring area and axial
height.

Fuel pin and tube universes are intentionally unbounded in the axial
direction. The parent core geometry trims the assembly lattice between
`STACK_BOTTOM_Z` and `STACK_TOP_Z`. This keeps the lattice entries reusable
while preserving the intended finite assembly stack in the top-level geometry.

## Requirements

Running this example requires an OpenMC Python environment and nuclear data
configured through `OPENMC_CROSS_SECTIONS`. Depletion also requires the OpenMC
executable and data needed by the coupled transport calculation.
