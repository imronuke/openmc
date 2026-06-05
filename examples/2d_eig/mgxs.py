import openmc

lib = openmc.MGXSLibrary.from_hdf5(filename='mgxs.h5')

for name in lib.names:
    print('\n----------------------------------------\n')
    xsdata = lib.get_by_name(name)
    print(f'MGXS Data for {name}:')
    print(f'Total cross section: {xsdata.total[0]}')
    print(f'Absorption cross section: {xsdata.absorption[0]}')
    if (xsdata.fission[0] is not None):
        print(f'Fission cross section: {xsdata.fission[0]}')
        print(f'Nu-fission cross section: {xsdata.nu_fission[0]}')
        print(f'Chi: {xsdata.chi}')
    print(f'Scattering matrix:\n{xsdata.scatter_matrix[0]}')