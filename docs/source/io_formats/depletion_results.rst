.. _io_depletion_results:

=============================
Depletion Results File Format
=============================

The current version of the depletion results file format is 1.3.

**/**

:Attributes: - **filetype** (*char[]*) -- String indicating the type of file.
             - **version** (*int[2]*) -- Major and minor version of the
               statepoint file format.

:Datasets: - **eigenvalues** (*double[][2]*) -- k-eigenvalues at each timestep.
             This array has shape (number of timesteps, 2). The second axis
             contains the eigenvalue and its associated uncertainty.
           - **number** (*double[][][]*) -- Total number of atoms at each
             timestep. This array has shape (number of timesteps, number of
             materials, number of nuclides).
           - **reaction rates** (*double[][][][]*) -- Reaction rates at each
             timestep. This array has shape (number of timesteps, number of
             materials, number of nuclides, number of reactions). Only stored if
             write_rates=True for non-tensor-train depletion.
           - **time** (*double[][2]*) -- Time in [s] at beginning/end of each
             step.
           - **source_rate** (*double[]*) -- Power in [W] or source rate in
             [neutron/sec] for each timestep.
           - **depletion time** (*double[]*) -- Average process time in [s]
             spent depleting a material across all burnable materials and,
             if applicable, MPI processes.
           - **keff_search_root** (*double[]*) -- Root of the keff search at the
             end of the timestep, if applicable.

**/materials/<id>/**

:Attributes: - **index** (*int*) -- Index used in results for this material
             - **volume** (*double*) -- Volume of this material in [cm^3]
             - **name** (*char[]*) -- Name of this material

**/nuclides/<name>/**

:Attributes: - **atom number index** (*int*) -- Index in array of total atoms
               for this nuclide
             - **reaction rate index** (*int*) -- Index in array of reaction
               rates for this nuclide

**/reactions/<name>/**

:Attributes: - **index** (*int*) -- Index user in results for this reaction

**/tt_depletion_reaction_rates/**

:Attributes: - **format_version** (*int*) -- Version of the tensor-train
               depletion reaction-rate storage format.
             - **stored_values** (*char[]*) -- Indicates the stored tensor-train
               values. Currently ``tally_mean``.
             - **normalization_mode** (*char[]*) -- Depletion normalization mode
               used with the stored tensor-train tally values.

**/tt_depletion_reaction_rates/steps/<step>/**

:Attributes: - **zero_source** (*bool*) -- Whether this step corresponds to a
               zero-source operator evaluation with no tensor-train tally cores.

:Datasets: - **normalization_factor** (*double*) -- Factor needed to normalize
             reconstructed tally means to depletion reaction rates.
           - **n_realizations** (*int*) -- Number of realizations used to
             convert the raw tensor-train tally sum to a mean.
           - **tt_shape** (*int[]*) -- Shape of the tensor-train tally.

**/tt_depletion_reaction_rates/steps/<step>/tt_mean/**

:Datasets: - **n_cores** (*int*) -- Number of tensor-train cores.
           - **core_<i>_shape** (*int[3]*) -- Shape of tensor-train core
             ``i``.
           - **core_<i>** (*double[]*) -- Flattened tensor-train core ``i``.
