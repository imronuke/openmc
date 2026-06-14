#include "openmc/memory_stats.h"

#include "openmc/message_passing.h"

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

#if defined(__linux__)
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(__unix__)
#include <sys/resource.h>
#endif

namespace openmc {
namespace memory_stats {

namespace {

double resident_sum_ {0.0};
int resident_samples_ {0};
Summary summary_;

double current_resident_memory()
{
#if defined(__linux__)
  std::ifstream statm {"/proc/self/statm"};
  long total_pages;
  long resident_pages;
  statm >> total_pages >> resident_pages;
  if (!statm)
    return 0.0;
  (void)total_pages;

  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return 0.0;

  return static_cast<double>(resident_pages) * page_size;
#elif defined(__APPLE__) && defined(__MACH__)
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  kern_return_t result = task_info(mach_task_self(), TASK_VM_INFO,
    reinterpret_cast<task_info_t>(&info), &count);
  if (result != KERN_SUCCESS)
    return 0.0;

  return static_cast<double>(info.resident_size);
#else
  return 0.0;
#endif
}

double peak_resident_memory()
{
#if defined(__linux__)
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    return 0.0;

  return static_cast<double>(usage.ru_maxrss) * 1024.0;
#elif defined(__APPLE__) && defined(__MACH__)
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    return 0.0;

  return static_cast<double>(usage.ru_maxrss);
#elif defined(__unix__)
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    return 0.0;

  return static_cast<double>(usage.ru_maxrss) * 1024.0;
#else
  return 0.0;
#endif
}

} // namespace

void reset()
{
  resident_sum_ = 0.0;
  resident_samples_ = 0;
  summary_ = {};
}

void sample()
{
  double resident = current_resident_memory();
  if (resident > 0.0) {
    resident_sum_ += resident;
    ++resident_samples_;
  }
}

void collect()
{
  double average_resident =
    resident_samples_ > 0 ? resident_sum_ / resident_samples_ : 0.0;
  double peak_resident = peak_resident_memory();
  int resident_available = average_resident > 0.0 ? 1 : 0;
  int peak_available = peak_resident > 0.0 ? 1 : 0;

#ifdef OPENMC_MPI
  double average_resident_sum;
  double max_average_resident;
  double max_peak_resident;
  int n_resident_available;
  int n_peak_available;

  MPI_Allreduce(&average_resident, &average_resident_sum, 1, MPI_DOUBLE,
    MPI_SUM, mpi::intracomm);
  MPI_Allreduce(&average_resident, &max_average_resident, 1, MPI_DOUBLE,
    MPI_MAX, mpi::intracomm);
  MPI_Allreduce(
    &peak_resident, &max_peak_resident, 1, MPI_DOUBLE, MPI_MAX, mpi::intracomm);
  MPI_Allreduce(&resident_available, &n_resident_available, 1, MPI_INT,
    MPI_SUM, mpi::intracomm);
  MPI_Allreduce(
    &peak_available, &n_peak_available, 1, MPI_INT, MPI_SUM, mpi::intracomm);

  summary_.resident_available = n_resident_available > 0;
  summary_.peak_available = n_peak_available > 0;
  summary_.average_resident =
    summary_.resident_available ? average_resident_sum / n_resident_available :
                                  0.0;
  summary_.max_average_resident =
    summary_.resident_available ? max_average_resident : 0.0;
  summary_.peak_resident = summary_.peak_available ? max_peak_resident : 0.0;
#else
  summary_.resident_available = resident_available;
  summary_.peak_available = peak_available;
  summary_.average_resident = average_resident;
  summary_.max_average_resident = average_resident;
  summary_.peak_resident = peak_resident;
#endif
}

const Summary& summary()
{
  return summary_;
}

} // namespace memory_stats
} // namespace openmc
