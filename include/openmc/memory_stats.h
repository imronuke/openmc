//! \file memory_stats.h
//! \brief Runtime process memory statistics

#ifndef OPENMC_MEMORY_STATS_H
#define OPENMC_MEMORY_STATS_H

namespace openmc {
namespace memory_stats {

//! Summary of process memory statistics in bytes
struct Summary {
  double average_resident {0.0};
  double max_average_resident {0.0};
  double peak_resident {0.0};
  bool resident_available {false};
  bool peak_available {false};
};

//! Reset sampled resident memory statistics
void reset();

//! Sample the current resident set size
void sample();

//! Collect rank-local memory statistics into a reportable summary
void collect();

//! Get the last collected memory statistics summary
const Summary& summary();

} // namespace memory_stats
} // namespace openmc

#endif // OPENMC_MEMORY_STATS_H
