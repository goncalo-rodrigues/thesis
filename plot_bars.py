from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

filename = '20x20_ta_k10'
results_folder = Path(filename)
res_filename = str(results_folder / 'results_eps')
eacc_filename = str(results_folder / 'eaccuracy_eps')
eaccprey_filename = str(results_folder / 'eaccuracyprey_eps')
bacc_filename = str(results_folder / 'baccuracy_eps')

episodes_range = (1, 50, 200)
results = []

eacc = []
bacc = []
eaccprey = []


for i in episodes_range:
    results.append(np.array(sorted(np.load(res_filename + str(i) + '.npy'))))
    eacc.append(np.array(np.load(eacc_filename + str(i) + '.npy')))
    eaccprey.append(np.array(np.load(eaccprey_filename + str(i) + '.npy')))
    bacc.append(np.array(np.load(bacc_filename + str(i) + '.npy')))

results = np.array(results)
eacc = np.array(eacc)
eaccprey = np.array(eaccprey)
bacc = np.array(bacc)

conf_intervals = np.array([st.t.interval(0.9, len(r)-1, loc=np.mean(r), scale=st.sem(r)) for r in results])
x_positions = [0, 1, 2, 3.5, 4.5]
patterns = ['//', '--', 'o', '*', 'x']
colors = ['red', 'green', 'blue', 'gray', 'white']
legends = ['Ad-hoc after 1 ep.', 'Ad-hoc after 50 ep.', 'Ad-hoc after 200 ep.', 'All greedy', 'PLASTIC-Model']
baseline_val = 23.63
baseline_err = 0.57
pmodel_val = 20
pmodel_err = 0.5
values = np.average(results, axis=1)

fig, _ = plt.subplots(clear=True)
fig.clf()
ax1 = fig.add_subplot(1,1,1)
bars = ax1.bar(x_positions, np.append(np.average(results, axis=1), (baseline_val, pmodel_val)),
        yerr=np.append((conf_intervals.T - values)[1,:], (baseline_err, pmodel_err)),
        color=colors, edgecolor='black', capsize=3, width=1)

for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)


ax1.set_ylabel('Steps to capture')
ax1.axes.xaxis.set_ticks([])
ax1.legend(bars, legends)
plt.savefig(filename+'_plot.png', bbox_inches='tight')
plt.show()

######
# GREEDY
# 5x5 val - 7.6
# 5x5 err - 0.275
# 10x10 val - 42.12
# 10x10 err - 2.362
# 20x20 val - 62.163
# 20x20 err - 3.240
# TEAMAWARE
# 20x20 val - 23.63
# 20x20 err - 0.57