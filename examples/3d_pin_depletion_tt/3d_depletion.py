from math import pi, sqrt

import matplotlib.pyplot as plt

import openmc
import openmc.deplete

###############################################################################
#                              Define materials
###############################################################################

uo2 = openmc.Material(name='UO2 fuel at 2.4% wt enrichment')
uo2.set_density('g/cm3', 10.29769)
uo2.add_element('U', 1., enrichment=2.4)
uo2.add_element('O', 2.)

helium = openmc.Material(name='Helium for gap')
helium.set_density('g/cm3', 0.001598)
helium.add_element('He', 2.4044e-4)

zircaloy = openmc.Material(name='Zircaloy 4')
zircaloy.set_density('g/cm3', 6.55)
zircaloy.add_element('Sn', 0.014, 'wo')
zircaloy.add_element('Fe', 0.00165, 'wo')
zircaloy.add_element('Cr', 0.001, 'wo')
zircaloy.add_element('Zr', 0.98335, 'wo')

borated_water = openmc.Material(name='Borated water')
borated_water.set_density('g/cm3', 0.740582)
borated_water.add_element('B', 4.0e-5)
borated_water.add_element('H', 5.0e-2)
borated_water.add_element('O', 2.4e-2)
borated_water.add_s_alpha_beta('c_H_in_H2O')

###############################################################################
#                             Create geometry
###############################################################################

pitch = 1.25984
fuel_length = 360.0
n_axial = 120
n_radial = 5
fuel_rad = 0.39218

zone_height = fuel_length / n_axial

fuel_ors = []
clad_ir = openmc.ZCylinder(r=0.40005, name='Clad IR')
clad_or = openmc.ZCylinder(r=0.45720, name='Clad OR')
box = openmc.model.RectangularPrism(pitch, pitch, boundary_type='reflective')
bottom = openmc.ZPlane(z0=-0.5 * fuel_length, boundary_type='vacuum')
top = openmc.ZPlane(z0=0.5 * fuel_length, boundary_type='vacuum')
z_planes = [
    openmc.ZPlane(z0=-0.5 * fuel_length + i * zone_height)
    for i in range(1, n_axial)
]
axial_surfaces = [bottom, *z_planes, top]


fuel_area = pi * fuel_rad**2
fuel_zone_area = fuel_area / n_radial
fuel_areas = [fuel_zone_area] * n_radial
for i in range(n_radial):
    rad = sqrt(fuel_zone_area * (i + 1) / pi)
    fuel_ors.append(openmc.ZCylinder(r=rad, name=f'Fuel OR {i + 1}'))

fuel_cells = []
fuel_mats = []
for r in range(n_radial):
    for i in range(n_axial):
        fuel_mat = uo2.clone()
        fuel_mat.name = f'UO2 fuel radial zone {r + 1} axial zone {i + 1}'
        fuel_mat.volume = fuel_areas[r] * zone_height
        fuel_mats.append(fuel_mat)

        axial_region = +axial_surfaces[i] & -axial_surfaces[i + 1]
        radial_region = -fuel_ors[0] if r == 0 else +fuel_ors[r - 1] & -fuel_ors[r]
        fuel_cells.append(openmc.Cell(
            name=f'Fuel radial zone {r + 1} axial zone {i + 1}',
            fill=fuel_mat,
            region=radial_region & axial_region))

axial_region = +bottom & -top
gap = openmc.Cell(
    name='Helium gap',
    fill=helium,
    region=+fuel_ors[-1] & -clad_ir & axial_region)
clad = openmc.Cell(
    name='Zircaloy cladding',
    fill=zircaloy,
    region=+clad_ir & -clad_or & axial_region)
water = openmc.Cell(
    name='Borated water',
    fill=borated_water,
    region=+clad_or & -box & axial_region)

geometry = openmc.Geometry([*fuel_cells, gap, clad, water])

###############################################################################
#                     Transport calculation settings
###############################################################################

settings = openmc.Settings()
settings.batches = 150
settings.inactive = 20
settings.particles = 10000

bounds = [
    -fuel_rad, -fuel_rad, -0.5 * fuel_length,
    fuel_rad, fuel_rad, 0.5 * fuel_length
]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
settings.source = openmc.IndependentSource(
    space=uniform_dist, constraints={'fissionable': True})

entropy_mesh = openmc.RegularMesh()
entropy_mesh.lower_left = [-fuel_rad, -fuel_rad, -0.5 * fuel_length]
entropy_mesh.upper_right = [fuel_rad, fuel_rad, 0.5 * fuel_length]
entropy_mesh.dimension = [10, 10, n_axial]
settings.entropy_mesh = entropy_mesh

###############################################################################
#                   Initialize and run depletion calculation
###############################################################################

model = openmc.Model(geometry=geometry, settings=settings)

chain_file = 'chain_simple.xml'
op = openmc.deplete.CoupledOperator(model, chain_file, tt_eps=0.01)

burnup_steps = [0.04]  # MWd/kgHM
linear_power = 174.0  # W/cm
power = linear_power * fuel_length  # W
integrator = openmc.deplete.PredictorIntegrator(
    op, burnup_steps, power, timestep_units='MWd/kg')
integrator.integrate(write_rates=True)

###############################################################################
#                    Read depletion calculation results
###############################################################################

results = openmc.deplete.Results("depletion_results.h5")

time, keff = results.get_keff(time_units='d')
for t, (mean, std_dev) in zip(time, keff):
    print(f"t = {t:.6f} d, keff = {mean:.6f} +/- {std_dev:.6f}")

###############################################################################
#                            Generate plots
###############################################################################

fig, ax = plt.subplots()
ax.errorbar(time, keff[:, 0], keff[:, 1], label="K-effective")
ax.set_xlabel("Time [d]")
ax.set_ylabel("Keff")
plt.show()
