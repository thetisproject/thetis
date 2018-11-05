# Example figure from output hdf5
import h5py
import matplotlib.pyplot as plt

df = h5py.File('outputs/diagnostic_lagoon_1.hdf5', 'r')

# Index library for output
index_conv = {"t": 0, "h_o": 1, "h_i": 2, "DZ": 3, "P": 4, "E": 5,
              "m": 6, "Q_t": 7, "Q_s": 8, "m_dt": 9, "m_t": 10, "f_r": 11}

fig, ax = plt.subplots(2, figsize=(10, 4), sharex="all")

# Plotting Elevations in time (black for outer water levels and red for inner)
ax[0].plot(df["operation_output"][:, index_conv["t"]], df["operation_output"][:, index_conv["h_o"]], color="black")
ax[0].plot(df["operation_output"][:, index_conv["t"]], df["operation_output"][:, index_conv["h_i"]], color="r")
ax[0].set_ylabel("$\\eta$ (m)")

# Plotting Power output in time
ax[1].plot(df["operation_output"][:, index_conv["t"]], df["operation_output"][:, index_conv["P"]], color="black")
ax[1].set_ylabel("$P$ (MW)")
ax[1].set_xlabel("$t$ (sec)")

fig.subplots_adjust(hspace=0)
plt.show()
