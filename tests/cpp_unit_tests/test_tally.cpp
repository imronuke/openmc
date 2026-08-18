#include "openmc/tallies/tally.h"
#include "openmc/settings.h"
#include "openmc/tallies/filter_particle.h"

#include <catch2/catch_test_macros.hpp>

using namespace openmc;

TEST_CASE("Test add/set_filter")
{
  // create a new tally object
  Tally* tally = Tally::create();

  // create a new particle filter
  Filter* particle_filter = Filter::create("particle");

  // add the particle filter to the tally
  tally->add_filter(particle_filter);

  // the filter should be added to the tally
  REQUIRE(tally->filters().size() == 1);
  REQUIRE(model::filter_map[particle_filter->id()] == tally->filters(0));

  // add the particle filter to the tally again
  tally->add_filter(particle_filter);
  // the tally should have the same number of filters
  REQUIRE(tally->filters().size() == 1);

  // create a cell filter
  Filter* cell_filter = Filter::create("cell");
  tally->add_filter(cell_filter);

  // now the size of the filters should have increased
  REQUIRE(tally->filters().size() == 2);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(1));

  // if we set the filters explicitly there shouldn't be extra filters hanging
  // around
  tally->set_filters({&cell_filter, 1});

  REQUIRE(tally->filters().size() == 1);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(0));

  // set filters again using both filters
  std::vector<Filter*> filters = {cell_filter, particle_filter};
  tally->set_filters(filters);

  REQUIRE(tally->filters().size() == 2);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(0));
  REQUIRE(model::filter_map[particle_filter->id()] == tally->filters(1));

  // set filters with a duplicate filter, should only add the filter to the tally once
  filters = {cell_filter, cell_filter};
  tally->set_filters(filters);
  REQUIRE(tally->filters().size() == 1);
  REQUIRE(model::filter_map[cell_filter->id()] == tally->filters(0));

}

TEST_CASE("Tensor-train tally uses explicit shape")
{
  settings::reduce_tallies = true;

  Tally* tally = Tally::create();
  auto* particle_filter =
    dynamic_cast<ParticleFilter*>(Filter::create("particle"));
  vector<ParticleType> particles {
    ParticleType(PDG_NEUTRON), ParticleType(PDG_PHOTON)};
  particle_filter->set_particles(particles);

  std::vector<Filter*> filters {particle_filter};
  tally->set_filters(filters);
  tally->set_strides();
  tally->set_scores({"fission", "absorption"});
  tally->use_tt_ = true;
  tally->tt_eps_ = 1.0e-3;
  tally->tt_shape_ = {2, 2};

  tally->init_results();

  REQUIRE(tally->tt_shape_.size() == 2);
  REQUIRE(tally->tt_shape_[0] == 2);
  REQUIRE(tally->tt_shape_[1] == 2);
}
