from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

results_folder = Path('10x10_ta_k10')
res_filename = str(results_folder / 'results_eps')
eacc_filename = str(results_folder / 'eaccuracy_eps')
eaccprey_filename = str(results_folder / 'eaccuracyprey_eps')
bacc_filename = str(results_folder / 'baccuracy_eps')

episodes_range = (1, 50, 200)
only_show_timesteps = False

results = []

eacc = []
bacc = []
eaccprey = []


for i in episodes_range:
    results.append(np.array(sorted(np.load(res_filename + str(i) + '.npy'))))
    if not only_show_timesteps:
        eacc.append(np.array(np.load(eacc_filename + str(i) + '.npy')))
        eaccprey.append(np.array(np.load(eaccprey_filename + str(i) + '.npy')))
        bacc.append(np.array(np.load(bacc_filename + str(i) + '.npy')))

results = np.array(results)
eacc = np.array(eacc)
eaccprey = np.array(eaccprey)
bacc = np.array(bacc)
print(bacc)
print(results)
fig, _ = plt.subplots(clear=True)
fig.clf()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

ax1.set_ylim([0.2, 1])
ax2.set_ylim([0, 100])
if not only_show_timesteps:
    ax1.plot(episodes_range, np.average(bacc, axis=1), marker='o', label='Behavior')
    ax1.plot(episodes_range, np.average(eacc, axis=1), marker='o', label='Environment')
    ax1.plot(episodes_range, np.average(eaccprey, axis=1), marker='o', label='Environment (prey)')

ax2.plot(episodes_range, np.average(results, axis=1), 'red', marker='o', label='Timesteps')
ax2.plot(episodes_range, [7.6]*len(episodes_range), label='4 greedy agents')
# ax2.plot(range(timesteps), [52]*timesteps, 'pink', label='baseline')
fig.legend()
plt.show()

print(','.join(str(a) for a in np.average(results, axis=1)))
print(','.join(str(a) for a in np.std(results, axis=1)))