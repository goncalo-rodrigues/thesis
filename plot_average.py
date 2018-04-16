import numpy as np
import matplotlib.pyplot as plt

res_filename = 'results_adhoc_t500/results_'
eacc_filename = 'results_adhoc_t500/eaccuracy_'
eaccprey_filename = 'results_adhoc_t500/eaccuracyprey_'
bacc_filename = 'results_adhoc_t500/baccuracy_'

sample_size = 30
timesteps = 200

for i in range(sample_size):
    timesteps = min(len(np.load(res_filename + str(i) + '.npy')), timesteps)

results = np.zeros((sample_size, timesteps))
eacc = np.zeros((sample_size, timesteps))
bacc = np.zeros((sample_size, timesteps))
eaccprey = np.zeros((sample_size, timesteps))


for i in range(sample_size):
    results[i] = np.load(res_filename + str(i) + '.npy')[:timesteps]
    eacc[i] = np.load(eacc_filename + str(i) + '.npy')[:timesteps]
    bacc[i] = np.load(bacc_filename + str(i) + '.npy')[:timesteps]
    eaccprey[i] = np.load(eaccprey_filename + str(i) + '.npy')[:timesteps]

print(np.average(results))
fig, _ = plt.subplots(clear=True)
fig.clf()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

ax1.set_ylim([0.7, 1])
ax2.set_ylim([20, 200])
ax1.plot(np.average(bacc, axis=0), label='Behavior')
ax1.plot(np.average(eacc, axis=0), label='Environment')
ax1.plot(np.average(eaccprey, axis=0), label='Environment (prey)')

ax2.plot(np.average(results, axis=0), 'red', label='Timesteps')
ax2.plot(range(timesteps), [52]*timesteps, 'pink', label='baseline')
fig.legend()
plt.show()

print(','.join(str(a) for a in np.average(results, axis=0)))
print(','.join(str(a) for a in np.std(results, axis=0)))