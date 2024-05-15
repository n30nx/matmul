import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Performance data
leaf_sizes = np.array([128, 256, 512, 1024, 2048])
execution_times = np.array([4.681795594, 5.076203895, 5.365837387, 5.465989967, 12.848879154])
cache_miss_rates = np.array([17.26, 10.22, 0.89, 0.43, 60.75])

# Normalize execution times and cache miss rates
normalized_times = (execution_times - np.mean(execution_times)) / np.std(execution_times)
normalized_miss_rates = (cache_miss_rates - np.mean(cache_miss_rates)) / np.std(cache_miss_rates)

# Combine normalized data
combined_signal = normalized_times + normalized_miss_rates

# Z-transform using frequency response
w, h = freqz(combined_signal)

# Plot the frequency response
plt.figure()
plt.plot(w, np.abs(h))
plt.title('Z-Transform of Combined Execution Times and Cache Miss Rates')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()

# Find the frequency of the peak magnitude
peak_freq_index = np.argmax(np.abs(h))
peak_freq = w[peak_freq_index]

# Map the peak frequency back to leaf sizes
optimal_leaf_size_index = int((peak_freq / np.pi) * len(leaf_sizes))
optimal_leaf_size = leaf_sizes[optimal_leaf_size_index]

print(f'Optimal Leaf Size: {optimal_leaf_size}')

