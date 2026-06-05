import numpy as np
import matplotlib.pyplot as plt
import openmc


TALLY_NAME = "Flux spectrum"
REFERENCE_STATEPOINT = "statepoint.100_ref.h5"
TT_STATEPOINT = "statepoint.100_tt.h5"


def get_spectrum(filename):
    with openmc.StatePoint(filename) as sp:
        tally = sp.get_tally(name=TALLY_NAME)
        energy_filter = tally.filters[0]
        energy_bins = energy_filter.bins
        energies = np.r_[energy_bins[:, 0], energy_bins[-1, 1]]

        if tally.uses_tt:
            print(tally.tt_eps)
            mean = tally.get_tt_values(value="mean").ravel()
            std_dev = tally.get_tt_values(value="std_dev").ravel()
        else:
            mean = tally.get_values(value="mean").ravel()
            std_dev = tally.get_values(value="std_dev").ravel()

    return energies, mean, std_dev


def step_values(values):
    return np.r_[values, values[-1]]


energies, mean_ref, std_ref = get_spectrum(REFERENCE_STATEPOINT)
energies_tt, mean_tt, std_tt = get_spectrum(TT_STATEPOINT)

if not np.array_equal(energies, energies_tt):
    raise ValueError("Energy bins differ between the two statepoints.")

rel_mean_diff = np.divide(
    np.abs(mean_tt - mean_ref),
    np.abs(mean_ref),
    out=np.zeros_like(mean_ref),
    where=mean_ref != 0.0,
)

fig, (ax_mean, ax_rel, ax_std) = plt.subplots(
    3, 1, sharex=True, figsize=(7, 8), constrained_layout=True
)

ax_mean.loglog(energies, step_values(mean_ref), drawstyle="steps-post",
               label="Dense")
ax_mean.loglog(energies, step_values(mean_tt), drawstyle="steps-post",
               linestyle="--", label="TT")
ax_mean.set_ylabel("Mean flux")
ax_mean.grid(True, which="both")
ax_mean.legend()

ax_rel.semilogx(energies, step_values(rel_mean_diff), drawstyle="steps-post")
ax_rel.set_ylabel("|TT - dense| / |dense|")
ax_rel.grid(True, which="both")

ax_std.loglog(energies, step_values(std_ref), drawstyle="steps-post",
              label="Dense")
ax_std.loglog(energies, step_values(std_tt), drawstyle="steps-post",
              linestyle="--", label="TT")
ax_std.set_xlabel("Energy [eV]")
ax_std.set_ylabel("Std. dev.")
ax_std.grid(True, which="both")
ax_std.legend()

plt.show()
