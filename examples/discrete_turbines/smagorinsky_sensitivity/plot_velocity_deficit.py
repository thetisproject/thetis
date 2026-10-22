"""Plot centreline wake recovery from the sensitivity transect CSV files."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', default='results')
parser.add_argument('--depth', type=float, default=None,
                    help='Water depth [m], shown in the plot title')
parser.add_argument('--average-steps', type=int, default=None,
                    help='Number of final timestep solutions in the mean')
args = parser.parse_args()

cases = [
    ('constant', 'constant viscosity', 'black'),
    ('cs_0p1', r'$C_s = 0.1$', 'C0'),
    ('cs_1', r'$C_s = 1$', 'C1'),
    ('cs_5', r'$C_s = 5$', 'C2'),
    ('cs_10', r'$C_s = 10$', 'C3'),
]

figure, axis = plt.subplots(figsize=(8, 4.5))
# The Physical Surface 100 refinement box is x = 300--650 m.  The turbine is
# at x = 425 m; the Gmsh Box field transitions over 30 m outside that patch.
array_start = (300.0 - 425.0)/20.0
array_end = (650.0 - 425.0)/20.0
transition_end = (680.0 - 425.0)/20.0
axis.axvspan(array_start, array_end, facecolor='C0', alpha=0.10, zorder=0)
axis.axvspan(array_end, transition_end, facecolor='0.5', alpha=0.12, zorder=0)
axis.axvspan(transition_end, 40.0, facecolor='C2', alpha=0.06, zorder=0)
for filename, label, colour in cases:
    path = os.path.join(args.results_dir, filename + '.csv')
    data = np.genfromtxt(path, delimiter=',', names=True)
    style = {'linestyle': '--', 'zorder': 10} if filename == 'constant' else {}
    axis.plot((data['x_m'] - 425.0)/20.0, data['velocity_deficit'], label=label,
              color=colour, linewidth=1.8, **style)

axis.axvline(0.0, color='0.5', linestyle='--', linewidth=1.0,
            label='turbine centre')
axis.axhline(0.0, color='0.7', linewidth=0.8)
axis.set(xlabel=r'distance from turbine, $(x-x_T)/D$', ylabel=r'velocity deficit, $1-u/U_0$')
axis.set_xlim(-5.0, 40.0)
if args.depth is not None:
    title = 'single-turbine wake recovery, depth = {:g} m'.format(args.depth)
    if args.average_steps is not None:
        title += ' (mean of final {:d} steps)'.format(args.average_steps)
    axis.set_title(title)
mesh_labels = axis.get_xaxis_transform()
axis.text(0.4, 0.04, r'array: $\Delta x = 4$ m', transform=mesh_labels,
          ha='center', va='bottom', fontsize=8, color='C0')
axis.text(7.0, 0.04, 'transition', transform=mesh_labels,
          ha='center', va='bottom', fontsize=7, color='0.35')
axis.text(23.5, 0.04, r'wake: $\Delta x = 20$ m', transform=mesh_labels,
          ha='center', va='bottom', fontsize=8, color='C2')
axis.legend(ncol=2)
axis.grid(True, alpha=0.25)
figure.tight_layout()
output = os.path.join(args.results_dir, 'velocity_deficit.png')
figure.savefig(output, dpi=200)
print('Wrote {}'.format(output))
