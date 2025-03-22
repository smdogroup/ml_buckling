import numpy as np
import matplotlib.pyplot as plt

# Generate some log-normal distributed data
data = np.random.lognormal(mean=1, sigma=1, size=1000)

# Define log-spaced bins
bin_edges = np.logspace(np.log10(min(data)), np.log10(max(data)), num=30)

# Plot histogram
plt.figure(figsize=(8,6))
plt.hist(data, bins=bin_edges, log=True, edgecolor='k')

# Set x-axis to log scale
plt.xscale('log')

# Labels
plt.xlabel('Value')
plt.ylabel('Frequency (log scale)')
plt.title('Log-Scale Histogram')

plt.show()
