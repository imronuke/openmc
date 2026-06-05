from math import log10

import numpy as np

import openmc
import openmc.mgxs

###############################################################################
# Create multigroup data

# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(group_edges=[
    1e-5, 0.0635, 20.0e6])

# Instantiate the 7-group (C5G7) cross section data
fu_xsdata = openmc.XSdata('fuel', groups)
fu_xsdata.order = 0
fu_xsdata.set_total([0.2351, 0.8928])
scatter_matrix = np.array(
    [[[0.2093, 0.0174],
      [0.0000, 0.8261]]])
tmp = fu_xsdata.total - np.sum(np.sum(scatter_matrix, axis=0), axis=1)
fu_xsdata.set_absorption(tmp.reshape(2,))
scatter_matrix = np.rollaxis(scatter_matrix, 0, 3)
fu_xsdata.set_scatter_matrix(scatter_matrix)
fu_xsdata.set_fission([0.0054, 0.1043])
fu_xsdata.set_nu_fission([0.0054, 0.1043])
fu_xsdata.set_chi([1.0, 0.0])

co_xsdata = openmc.XSdata('coolant', groups)
co_xsdata.order = 0
co_xsdata.set_total([0.2290, 1.1497])
scatter_matrix = np.array(
    [[[0.1195, 0.0290],
      [0.0000, 1.1402]]])
tmp = co_xsdata.total - np.sum(np.sum(scatter_matrix, axis=0), axis=1)
co_xsdata.set_absorption(tmp.reshape(2,))
scatter_matrix = np.rollaxis(scatter_matrix, 0, 3)
co_xsdata.set_scatter_matrix(scatter_matrix)

# scatter_matrix = np.array(
#     [[[0.1195, 0.0290],
#       [0.0000, 1.1402]]])
# print(np.sum(np.sum(scatter_matrix, axis=0), axis=1))
# print(co_xsdata.absorption)

mg_cross_sections_file = openmc.MGXSLibrary(groups)
mg_cross_sections_file.add_xsdatas([fu_xsdata, co_xsdata])
mg_cross_sections_file.export_to_hdf5()

###############################################################################
# Create materials for the problem

# Instantiate some Macroscopic Data
fuel_data = openmc.Macroscopic('fuel')
coolant_data = openmc.Macroscopic('coolant')

# Instantiate some Materials and register the appropriate Macroscopic objects
fuel = openmc.Material(name='fuel_material')
fuel.set_density('macro', 1.0)
fuel.add_macroscopic(fuel_data)

cool = openmc.Material(name='coolant_material')
cool.set_density('macro', 1.0)
cool.add_macroscopic(coolant_data)

# Instantiate a Materials collection and export to XML
materials_file = openmc.Materials([fuel, cool])
materials_file.cross_sections = "mgxs.h5"
materials_file.export_to_xml()

###############################################################################
# Define problem geometry

# Create a surface for the fuel outer radius
xmin = openmc.XPlane(x0=0.0, boundary_type='vacuum')
xmax = openmc.XPlane(x0=60.0, boundary_type='vacuum')
ymin = openmc.YPlane(y0=0.0, boundary_type='vacuum')
ymax = openmc.YPlane(y0=60.0, boundary_type='vacuum')
zmin = openmc.ZPlane(z0=-1.0, boundary_type= 'reflective')
zmax = openmc.ZPlane(z0=1.0, boundary_type='reflective')
x1 = openmc.XPlane(x0=10.0)
x2 = openmc.XPlane(x0=50.0)
y1 = openmc.YPlane(y0=10.0)
y2 = openmc.YPlane(y0=50.0)

cell0 = openmc.Cell(cell_id=0, name='Cell 0')
cell1 = openmc.Cell(cell_id=1, name='Cell 1')
cell2 = openmc.Cell(cell_id=2, name='cell 2')

cell0.region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax
cell1.region = +x1 & -x2 & +y1 & -y2
cell2.region = -x1 | +x2 | -y1 | +y2

cell1.fill = fuel
cell2.fill = cool

univ = openmc.Universe(universe_id=1)
root = openmc.Universe(universe_id=0, name='root universe')

univ.add_cells([cell1, cell2])
root.add_cell(cell0)
cell0.fill = univ

# Create a geometry with the two cells and export to XML
geometry = openmc.Geometry(root)
geometry.export_to_xml()

###############################################################################
# Define problem settings

# Instantiate a Settings object, set all runtime parameters, and export to XML
settings = openmc.Settings()
settings.energy_mode = "multi-group"
settings.batches = 100
settings.inactive = 10
settings.particles = 100000

# Create an initial uniform spatial source distribution over fissionable zones
lower_left = (10.0, 10.0, -1.0)
upper_right = (50.0, 50.0, 1.0)
uniform_dist = openmc.stats.Box(lower_left, upper_right)
settings.source = openmc.IndependentSource(
    space=uniform_dist, constraints={'fissionable': True})
settings.export_to_xml()

###############################################################################
# Define tallies

# Create a mesh that will be used for tallying
mesh = openmc.RegularMesh()
mesh.dimension = (120, 120, 1)
mesh.lower_left = (0.0, 0.0, -1.0)
mesh.upper_right = (60.0, 60.0, 1.0)

# Create a mesh filter that can be used in a tally
mesh_filter = openmc.MeshFilter(mesh)

# Now use the mesh filter in a tally and indicate what scores are desired
mesh_tally = openmc.Tally(name="Mesh tally")
mesh_tally.filters = [mesh_filter]
mesh_tally.scores = ['flux']

# Instantiate a Tallies collection and export to XML
tallies = openmc.Tallies([mesh_tally])
tallies.export_to_xml()

###############################################################################
#                   Exporting to OpenMC plots.xml file
###############################################################################

plot1 = openmc.Plot(plot_id=1)
plot1.origin = [30, 30, 0]
plot1.width = [60, 60]
plot1.pixels = [600, 600]
plot1.color_by = 'material'

plot2 = openmc.Plot(plot_id=2)
plot2.origin = [30, 60, 0]
plot2.basis = 'xz'
plot2.width = [60, 2.0]
plot2.pixels = [600, 40]
plot2.color_by = 'material'


# Instantiate a Plots collection and export to XML
plot_file = openmc.Plots([plot1, plot2])
plot_file.export_to_xml()

# Run the simulation
# openmc.run()
