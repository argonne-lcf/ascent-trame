import yaml #pip install --user pyyaml
import pandas as pd
import matplotlib.pyplot as plt

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

session = []
with open(r'bin_data.yaml') as file:
  session = yaml.load(file, Loader=Loader)

binning = session['avg_z_velocity']
cycles = list(binning.keys())
bins = []

# loop through each cycle and grab the bins
for cycle in binning.values():
  bins.append((cycle['attrs']['value']['value']))

# create the coordinate axis using bin centers
z_axis = binning[cycles[0]]['attrs']['bin_axes']['value']['z']
z_min = z_axis['min_val']
z_max = z_axis['max_val']
z_bins = z_axis['num_bins']

z_delta = (z_max - z_min) / float(z_bins)
z_start = z_min + 0.5 * z_delta
z_vals = []
for b in range(0,z_bins):
  z_vals.append(b * z_delta + z_start)

# plot the curve from the last cycle
for i, bin in enumerate(bins):
  plt.plot(z_vals, bins[-1])
  plt.xlabel('z position')
  plt.ylabel('average velocity')
  plt.savefig(f"binning-{i + 1}.png")