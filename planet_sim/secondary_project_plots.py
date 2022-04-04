import pickle
import numpy as np
import matplotlib.pyplot as plt

from planet_sim.transit_toolbox import generate_sigma

# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_1_compact_retrieval.pkl'
# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_noise125_compact_retrieval.pkl'
# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_mixedcase_compact_retrieval.pkl'
filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_mixedcase_500smpl_compact_retrieval.pkl'

with open(filename, 'rb') as f:
    benneke_archive = pickle.load(f)

h2otrue_archive = benneke_archive['h2o_true']
ch4true_archive = benneke_archive['ch4_true']
mixedtrue_archive = benneke_archive['h2och4_true']


# labeling stuff
sigma_1 = 0.9  # sigma = 2
sigma_2 = 5.0  # sigma = 3.6
sigma_3 = 11.0  # sigma = 5
xs = np.array([-sigma_3, -sigma_2, -sigma_1, sigma_1, sigma_2, sigma_3])
ys = np.zeros(xs.shape) + 0.5
labels = ['5σ', '3.6σ', '2σ', '2σ', '3.6σ', '5σ']


'''Water detection'''

# water is present
delta_logz_water_mixedtrue = mixedtrue_archive['logz_h2och4'] - mixedtrue_archive['logz_ch4']

# water is not present
delta_logz_water_ch4true = ch4true_archive['logz_h2och4'] - ch4true_archive['logz_ch4']

# hist_fig, hist_ax = plt.subplots(1, figsize=(12, 6))
hist_fig, hist_ax = plt.subplots(1, 1, figsize=(8, 6))
hist_ax.hist(delta_logz_water_mixedtrue, bins=25, alpha=0.5)
hist_ax.hist(delta_logz_water_ch4true, bins=25, alpha=0.5)
hist_fig.suptitle('Water detection (CH4 vs H2OCH4 atmosphere)')
hist_ax.set_xlabel('Bayes Evidence Ratio (Δlog(z))')
hist_ax.axvline(0.9, label='2σ Positive Result', color='b')
hist_ax.axvline(5.0, color='b')
hist_ax.axvline(11.0, color='b')
hist_ax.axvline(-0.9, label='2σ Negative result', color='g')
hist_ax.axvline(-5.0, color='g')
hist_ax.axvline(-11.0, color='g')


for x, y, label in zip(xs, ys, labels):
    if x < 0:
        ha = 'right'
    else:
        ha = 'left'
    hist_ax.annotate(label,
                (x, y),
                ha=ha)

hist_ax.legend()




