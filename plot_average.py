from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

results_folder = Path('results_adhoc_greedycorrected_1')
res_filename = str(results_folder / 'results_')
eacc_filename = str(results_folder / 'eaccuracy_')
eaccprey_filename = str(results_folder / 'eaccuracyprey_')
bacc_filename = str(results_folder / 'baccuracy_')

sample_size = 1
timesteps = 2**31
only_show_timesteps = False

for i in range(sample_size):
    timesteps = min(len(np.load(res_filename + str(i) + '.npy')), timesteps)

results = np.zeros((sample_size, timesteps))

eacc = np.zeros((sample_size, timesteps))
bacc = np.zeros((sample_size, timesteps))
eaccprey = np.zeros((sample_size, timesteps))


for i in range(sample_size):
    results[i] = np.load(res_filename + str(i) + '.npy')[:timesteps]
    if not only_show_timesteps:
        eacc[i] = np.load(eacc_filename + str(i) + '.npy')[:timesteps]
        bacc[i] = np.load(bacc_filename + str(i) + '.npy')[:timesteps]
        eaccprey[i] = np.load(eaccprey_filename + str(i) + '.npy')[:timesteps]

print(np.average(results))
fig, _ = plt.subplots(clear=True)
fig.clf()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

ax1.set_ylim([0.5, 1])
ax2.set_ylim([0, 70])
if not only_show_timesteps:
    ax1.plot(np.average(bacc, axis=0), label='Behavior')
    ax1.plot(np.average(eacc, axis=0), label='Environment')
    ax1.plot(np.average(eaccprey, axis=0), label='Environment (prey)')

ax2.plot(np.average(results, axis=0), 'red', label='Timesteps')
# ax2.plot(range(timesteps), [52]*timesteps, 'pink', label='baseline')
fig.legend()
plt.show()

print(','.join(str(a) for a in np.average(results, axis=0)))
print(','.join(str(a) for a in np.std(results, axis=0)))