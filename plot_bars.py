from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

BASELINE_VALS = {
    'greedy': {
        (5, 5): (7.5, 0.275),
        (10, 10): (42.12, 2.362),
        (20, 20): (62.163, 3.240)
    },
    'ta': {
        (5, 5): (7.43, 0.30),
        (10, 10): (14.16, 0.486),
        (20, 20): (23.63, 0.57)
    }
}

PLASTIC_VALS_K10 = {
    'greedy': {
        (5, 5): (5.55, 0.541),
        (10, 10): (17.38, 2.257),
        (20, 20): (42.03, 4.27)
    },
    'ta': {
        (5, 5): (5.42, 0.507),
        (10, 10): (12.17, 1.232),
        (20, 20): (26.83, 2.4)
    }

}

DEEPQLEARN_VALS = {
    'greedy': {
        (5, 5): (8.24, 9.032-8.24),
        (10, 10): (97.3, 107.75-97.3),
        (20, 20): (125.99, 142.03-125.99)
    },
    'ta': {
        (5, 5): (11.96, 13.272-11.96),
        (10, 10): (30.61, 33.587-30.61),
        (20, 20): (62.73, 70.32-62.73)
    }
}

PLASTIC_WITH_LEARNED_MODEL_VALS = {
    'greedy': {
        (5, 5): (5.39, 5.92-5.39),
        (10, 10): (24.18, 3.884),
        (20, 20): (47.84, 5.179)
        # 10x10: (14,9,161,9,59,9,28,21,48,6,12,42,28,11,11,10,7,5,12,102,8,8,15,31,17,35,10,9,34,9,33,19,13,28,21,8,38,16,96,6,8,42,8,21,7,12,9,8,30,26,12,21,16,8,10,50,25,12,15,9,9,36,4,15,49,23,42,7,7,7,45,21,28,30,62,29,76,4,7,51,25,14,47,24,42,13,7,16,51,8,8,7,44,20,32,12,29,42,8,10,)
        # 20x20: (32,47,44,39,24,18,56,19,26,13,39,89,57,33,52,20,64,20,13,80,32,24,29,106,28,37,45,71,65,109,13,66,53,18,113,27,63,103,48,45,33,55,51,24,19,26,20,91,16,62,26,195,16,60,47,81,47,50,26,16,33,103,70,101,84,19,30,33,18,64,47,68,43,88,33,50,33,106,14,100,31,20,43,70,53,24,20,27,48,23,56,32,45,44,14,44,130,18,32,32,)

    },
    'ta': {
        (5, 5): (7.19, 0.854),
        (10, 10): (15.92, 1.89),
        (20, 20): (33.66, 3.456)
        # 5x5: (5,12,5,6,3,3,24,5,7,3,2,9,20,5,7,4,6,4,4,8,10,9,8,4,5,4,4,13,8,3,3,2,4,10,27,9,5,5,5,4,6,19,13,4,3,5,8,2,7,22,6,14,5,3,16,11,4,5,4,7,2,3,3,12,4,13,8,2,2,8,3,2,12,4,9,7,6,2,5,2,12,9,5,5,5,3,5,10,4,5,8,11,12,3,4,12,10,23,12,4,)
        # 10x10: (5,17,10,40,9,79,9,12,10,10,4,12,7,31,11,10,7,11,29,17,7,20,6,19,9,24,7,7,36,11,16,16,10,42,23,11,26,12,8,11,8,12,8,9,12,38,9,2,11,19,6,25,9,14,10,7,7,13,16,11,28,19,43,12,26,8,31,11,8,9,14,10,13,13,25,13,36,10,8,26,15,36,7,17,14,10,41,11,10,19,21,13,8,14,26,8,5,13,20,14,)
        # 20x20: (38,66,25,18,22,48,12,24,27,11,36,51,33,12,75,89,64,39,27,49,17,48,38,37,45,24,17,28,21,41,16,14,14,42,43,33,42,22,45,44,19,25,28,27,56,55,30,20,26,39,12,150,25,15,19,32,50,15,24,13,12,72,22,30,29,19,9,19,16,44,67,34,31,40,28,46,24,34,26,68,26,20,36,18,12,46,25,76,20,23,31,37,25,15,47,22,18,16,27,79,)
    }
}

ADHOC_1EP = {
    'greedy': {
        (5, 5): (37.94, 4.80),
        (10, 10): (191.85, 21.562),
        (20, 20): (209.01, 23.99)
    },
    'ta': {
        (5, 5): (23.68, 4.55),
        (10, 10): (32.18, 7.475),
        (20, 20): (99.12, 11.23)
    }
}

ADHOC_50EP = {
    'greedy': {
        (5, 5): (11.07, 1.09),
        (10, 10): (42.04, 6.939),
        (20, 20): (65.51, 8.261)
    },
    'ta': {
        (5, 5): (22.16, 3.62),
        (10, 10): (24.21, 3.419),
        (20, 20): (44.38, 5.04)
    }
}

ADHOC_200EP = {
    'greedy': {
        (5, 5): (7.24, 0.9289),
        (10, 10): (41.68, 5.88),
        (20, 20): (63.47, 8.29)
    },
    'ta': {
        (5, 5): (11.93, 1.44),
        (10, 10): (24.23, 4.09),
        (20, 20): (39.21, 4.02)
    }
}
for world_size in ((5,5), (10,10), (20, 20)):
    agent_type = 'greedy'
    filename = f'{world_size[0]}x{world_size[1]}_{agent_type}_random'
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

    print("WORLD SIZE " + str(world_size))
    print(results)
    conf_intervals = np.array([st.t.interval(0.9, len(r) - 1, loc=np.mean(r), scale=st.sem(r)) for r in results])
    x_positions = [0, 1, 2, 3.5, 4.5, 5.5, 6.5]
    patterns = ['//', '--', 'o', '*', 'x', '\\', 'oo']
    colors = ['red', 'green', 'blue', 'gray', 'white', 'purple', 'yellow']
    legends = ['Ad-hoc after 1 ep.', 'Ad-hoc after 50 ep.', 'Ad-hoc after 200 ep.', f'All {agent_type}', 'PLASTIC-Model', 'PLASTIC-Model(NN)', 'Deep Q-learning']
    values = np.average(results, axis=1)

    fig, _ = plt.subplots(clear=True)
    fig.clf()
    ax1 = fig.add_subplot(1, 1, 1)
    bars = ax1.bar(x_positions, np.append(np.average(results, axis=1),
                                          (BASELINE_VALS[agent_type][world_size][0],
                                           PLASTIC_VALS_K10[agent_type][world_size][0],
                                           PLASTIC_WITH_LEARNED_MODEL_VALS[agent_type][world_size][0],
                                           DEEPQLEARN_VALS[agent_type][world_size][0])),
                   yerr=np.append((conf_intervals.T - values)[1, :],
                                  (BASELINE_VALS[agent_type][world_size][1],
                                   PLASTIC_VALS_K10[agent_type][world_size][1],
                                   PLASTIC_WITH_LEARNED_MODEL_VALS[agent_type][world_size][1],
                                   DEEPQLEARN_VALS[agent_type][world_size][1])),
                   color=colors, edgecolor='black', capsize=3, width=1)

    for bar, pattern in zip(bars, patterns):
        bar.set_hatch(pattern)


    ax1.set_ylabel('Steps to capture')
    ax1.axes.xaxis.set_ticks([])
    ax1.legend(bars, legends)
    plt.savefig(filename + '_plot.png', bbox_inches='tight')
    plt.show()
