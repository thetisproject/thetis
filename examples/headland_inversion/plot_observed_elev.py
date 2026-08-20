"""Plot stored detector elevation time series from Thetis detector HDF5 files.

By default, scans the selected forward output directories and plots all
`diagnostic_timeseries_*.hdf5` files it finds.

Usage:
    python plot_detector_elev.py
    python plot_detector_elev.py --ensemble
"""

from __future__ import annotations
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Run the headland forward model in standard or ensemble mode.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--ensemble', action='store_true',
                    help='Enable ensemble mode with one station/time offset per ensemble member')
args = parser.parse_args()

plt.rc('font', size=14)

if args.ensemble:
    DEFAULT_OUTPUT_GLOB = "outputs/outputs_forward/member_*/diagnostic_timeseries_*.hdf5"
else:
    DEFAULT_OUTPUT_GLOB = "outputs/outputs_forward/diagnostic_timeseries_*.hdf5"


def find_hdf5_files(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob(DEFAULT_OUTPUT_GLOB))


def extract_time_and_elevation(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5file:
        if "time" in h5file:
            time = np.asarray(h5file["time"], dtype=float)
            station_keys = [key for key in h5file.keys() if key != "time"]
            if not station_keys:
                raise RuntimeError(f"No station datasets found in {h5_path}")
            station_data = np.asarray(h5file[station_keys[0]], dtype=float)
            if station_data.ndim != 2 or station_data.shape[1] < 1:
                raise RuntimeError(f"Unexpected station dataset shape in {h5_path}: {station_data.shape}")
            elev = station_data[:, 0]
            return time, elev

        dataset = None

        def visitor(_, obj):
            nonlocal dataset
            if dataset is None and isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                names = list(obj.dtype.names or [])
                if "time" in names and "elev_2d" in names:
                    dataset = obj[()]

        h5file.visititems(visitor)

        if dataset is None:
            raise RuntimeError(f"Could not find detector time/elevation data in {h5_path}")

        time = np.asarray(dataset["time"], dtype=float)
        elev = np.asarray(dataset["elev_2d"], dtype=float)
        return time, elev


def station_label(h5_path: Path) -> str:
    stem = h5_path.stem
    prefix = "diagnostic_timeseries_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    files = find_hdf5_files(script_dir)

    if not files:
        raise SystemExit("No detector HDF5 files found")

    plt.figure(figsize=(8, 6))
    for h5_path in files:
        time, elev = extract_time_and_elevation(h5_path)
        plt.plot(time, elev, label=station_label(h5_path))

    plt.xlabel("Time (s)")
    plt.ylabel("Elevation (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
