import openmc

# Load the summary file
summary = openmc.Summary('summary.h5')

# Access reconstructed geometry and materials
geometry = summary.geometry
materials = summary.materials

# Print some information
print(f"OpenMC Version: {summary.version}")
print(f"Simulation Start Time: {summary.date_and_time}")
print(f"Materials: {summary.materials}")
print(f"Materials: {summary.macroscopics}")