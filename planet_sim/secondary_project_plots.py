import pickle
import numpy as np
import matplotlib.pyplot as plt

# from planet_sim.transit_toolbox import generate_sigma
from planet_sim.transit_toolbox import bayes_to_pvalue
from planet_sim.transit_toolbox import bayes_to_sigma
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
ys = np.zeros(xs.shape) + 0.001
labels = ['5σ', '3.6σ', '2σ', '2σ', '3.6σ', '5σ']


'''Water detection'''
# scratch space
# 'True Model: water-methane mix'
# 'True Model: methane-only (no water)'

# water is present
delta_logz_water_mixedtrue = mixedtrue_archive['logz_h2och4'] - mixedtrue_archive['logz_ch4']

threshhold = 3.6  # 3 sigma
water_falsenegative_rate = np.sum(delta_logz_water_mixedtrue < threshhold)/delta_logz_water_mixedtrue.size
print('False negative rate:', water_falsenegative_rate)
print('True positive rate:', 1-water_falsenegative_rate)

# water is not present
delta_logz_water_ch4true = ch4true_archive['logz_h2och4'] - ch4true_archive['logz_ch4']
water_falsepositive_rate = np.sum(delta_logz_water_ch4true > threshhold)/delta_logz_water_ch4true.size
print('False positive rate:', water_falsepositive_rate)


def calculate_pvalue(x, true_positive_data, false_positive_data):
    truepositive_rate = []
    falsepostive_rate = []
    for xvalue in x:
        truepositive_rate.append( np.sum(true_positive_data > xvalue)/true_positive_data.size )
        falsepostive_rate.append( np.sum(false_positive_data > xvalue)/false_positive_data.size )

    truepositive_rate = np.array(truepositive_rate)
    falsepostive_rate = np.array(falsepostive_rate)

    # calculate probability of true positive, while masking zero values
    return 1 - falsepostive_rate/(np.ma.masked_equal(falsepostive_rate + truepositive_rate, 0))


x_bayesfactors = np.linspace(0,
                             # np.concatenate((delta_logz_water_mixedtrue, delta_logz_water_ch4true)).max(),
                             delta_logz_water_ch4true.max(),
                             num=25)

pvalues = calculate_pvalue(x_bayesfactors, delta_logz_water_mixedtrue, delta_logz_water_ch4true)

fig, ax = plt.subplots()
ax.plot(x_bayesfactors, pvalues, 'o')
ax.set_xlabel('Bayes Factor values')
ax.set_ylabel('Numerically calculated p values')
fig.suptitle('P value distribution for Water detection')

# hist_fig, hist_ax = plt.subplots(1, figsize=(12, 6))
hist_fig, hist_ax = plt.subplots(1, 1, figsize=(8, 6))
hist_ax.hist(delta_logz_water_mixedtrue, bins=25, density=True, alpha=0.6, label='True Model: water-methane mix (H2O=0.8%, CH4=0.24%)')
hist_ax.hist(delta_logz_water_ch4true, bins=25, density=True, alpha=0.6, label='True Model: methane-only (H2O=0.0%, CH4=0.24%)')
hist_fig.suptitle('Water detection\n(Model Comparison: CH4 vs H2OCH4 atmosphere)')
hist_ax.set_xlabel('Bayes Evidence Ratio (Δlog(z))')
hist_ax.set_ylabel('Probability Density')
hist_ax.axvline(0.9, label='naive σ thresholds for Water detection', color='b')
hist_ax.axvline(5.0, color='b')
hist_ax.axvline(11.0, color='b')
hist_ax.axvline(-0.9, label='naive σ thresholds for Water rejection', color='g')
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
hist_ax.set_xlim(-2, 7)


'''Methane detection'''
# methane is present
delta_logz_methane_mixedtrue = mixedtrue_archive['logz_h2och4'] - mixedtrue_archive['logz_h2o']

# methane is not present
delta_logz_methane_h2otrue = h2otrue_archive['logz_h2och4'] - h2otrue_archive['logz_h2o']

x_bayesfactors = np.linspace(0,
                             # np.concatenate((delta_logz_water_mixedtrue, delta_logz_water_ch4true)).max(),
                             delta_logz_methane_h2otrue.max(),
                             num=25)

pvalues = calculate_pvalue(x_bayesfactors, delta_logz_methane_mixedtrue, delta_logz_methane_h2otrue)
fig, ax = plt.subplots()
ax.plot(x_bayesfactors, pvalues, 'o')
ax.set_xlabel('Bayes Factor values')
ax.set_ylabel('Numerically calculated p values')
fig.suptitle('P value distribution for Methane detection')

print(calculate_pvalue([3.6], delta_logz_methane_mixedtrue, delta_logz_methane_h2otrue))


# hist_fig, hist_ax = plt.subplots(1, figsize=(12, 6))
hist_fig, hist_ax = plt.subplots(1, 1, figsize=(8, 6))
# hist_ax.hist(delta_logz_methane_mixedtrue, bins=25, density=True, alpha=0.6, label='True Model: water-methane mix (H2O=0.8%, CH4=0.24%)')
hist_ax.hist(delta_logz_methane_h2otrue, bins=25, density=True, alpha=0.6, label='True Model: water-only (H2O=0.8%, CH4=0.0%)')
hist_fig.suptitle('Methane detection\n(Model Comparison: H2O vs H2OCH4 atmosphere)')
hist_ax.set_xlabel('Bayes Evidence Ratio (Δlog(z))')
hist_ax.set_ylabel('Probability Density')
hist_ax.axvline(0.9, label='naive σ thresholds for Methane detection', color='b')
hist_ax.axvline(5.0, color='b')
hist_ax.axvline(11.0, color='b')
hist_ax.axvline(-0.9, label='naive σ thresholds for Methane  rejection', color='g')
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
hist_ax.set_xlim(-2, 21)





'''degeneracy breaking?'''
# methane is present
delta_logz_ch4vsh2o_mixedtrue = mixedtrue_archive['logz_h2o'] - mixedtrue_archive['logz_ch4']

# methane is not present
delta_logz_ch4vsh2o_ch4true = ch4true_archive['logz_h2o'] - ch4true_archive['logz_ch4']

# hist_fig, hist_ax = plt.subplots(1, figsize=(12, 6))
hist_fig, hist_ax = plt.subplots(1, 1, figsize=(8, 6))
hist_ax.hist(delta_logz_ch4vsh2o_mixedtrue, bins=25, density=True, alpha=0.6, label='True Model: water-methane mix (H2O=0.8%, CH4=0.24%)')
hist_ax.hist(delta_logz_ch4vsh2o_ch4true, bins=25, density=True, alpha=0.6, label='True Model: water-only (H2O=0.0%, CH4=0.24%)')
hist_fig.suptitle('Water detection: degeneracy breaking\n(Model Comparison: H2O vs CH4 atmosphere)')
hist_ax.set_xlabel('Bayes Evidence Ratio (Δlog(z))')
hist_ax.set_ylabel('Probability Density')
hist_ax.axvline(0.9, label='naive σ thresholds for Methane detection', color='b')
hist_ax.axvline(5.0, color='b')
hist_ax.axvline(11.0, color='b')
hist_ax.axvline(-0.9, label='naive σ thresholds for Methane rejection σ', color='g')
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
hist_ax.set_xlim(-15.5, 21)


#
# theta = theta_h2och4
#
#
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(fine_wavelengths, fine_transit * 1e6, color='tab:grey', alpha=0.5, label='High resolution spectrum')
# ax.plot(fine_wavelengths, filtered_transit * 1e6, color='tab:blue', label='Smoothed spectrum')
# ax.errorbar(pixel_wavelengths, noisey_transit_depth_mix[0] * 1e6, yerr=err*1e6, color='tab:orange', label='sampled spectrum', fmt='o', capsize=3.0)
# ax.set_xlabel('Wavelength (μm)')
# ax.set_ylabel('transit depth (ppm)')
# fig.suptitle('Sample spectrum of water-methane mix atmosphere')
# ax.legend()
#
#

