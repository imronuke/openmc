.. _io_depletion_results:

=============================
Depletion Results File Format
=============================

The current version of the depletion results file format is 1.4.

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
             ``write_rates=True`` and tensor-train reaction-rate storage is not
             used.
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
               depletion reaction-rate storage format. Version 2 stores
               per-channel means and channel-to-rate indices.
             - **tt_layout_version** (*int*) -- Tensor-train layout version.
               Version 1 stores packed per-channel cores.
             - **stored_values** (*char[]*) -- Indicates the stored tensor-train
               values. Currently ``tally_mean``.
             - **normalization_mode** (*char[]*) -- Depletion normalization mode
               used with the stored tensor-train tally values.
             - **tt_eps** (*double*) -- Tensor-train truncation tolerance used
               for depletion reaction-rate tally accumulation.

:Datasets: - **reaction_rate_mask** (*bool[][]*) -- Mask indicating which
             nuclide/reaction pairs are present in the depletion chain. This
             array has shape (number of reaction-rate nuclides, number of
             reactions). The mask is solver metadata; stored tensor-train tally
             means are not masked.
           - **channel_indices** (*int64[n_channels][2]*) -- For each
             tensor-train channel, the corresponding nuclide and reaction
             indices in the reaction-rate matrix. The channel ordering
             matches this dataset.

**/tt_depletion_reaction_rates/steps/<step>/**

:Attributes: - **zero_source** (*bool*) -- Whether this step corresponds to a
               zero-source operator evaluation with no tensor-train tally cores.

:Datasets: - **normalization_factor** (*double*) -- Factor needed to normalize
             reconstructed tally means to depletion reaction rates.
           - **n_realizations** (*int*) -- Number of realizations used to
             convert the raw tensor-train tally sum to a mean.
           - **tt_shape** (*int[]*) -- Spatial shape of each channel's
             tensor-train. Its product equals the number of material filter
             bins.

**/tt_depletion_reaction_rates/steps/<step>/tt_mean/**

:Datasets: - **channel_core_offsets** (*int64[]*) -- Length is one greater
             than the number of channels. Zero-based start and end core indices
             for each channel, with end offsets excluded.
           - **core_shapes** (*int[]*) -- Flattened triples
             ``(r_left, mode_size, r_right)`` for each core, in channel order.
           - **core_data_offsets** (*int64[]*) -- Length is one greater than
             the number of cores. Zero-based start and end offsets into
             **core_data** for each core, with end offsets excluded.
           - **core_data** (*double[]*) -- Concatenated row-major values for
             every core. All-zero channels have no cores and equal adjacent
             entries in **channel_core_offsets**.

Each TT channel is one nuclide-reaction pair and uses the shared spatial
``tt_shape``. The channel's indices in **channel_indices** map it into the
reaction-rate matrix. The stored cores represent the tally mean; the shared
normalization factor and material volume convert a material slice to a
depletion reaction rate.
