import pickle
import numpy as np
import matplotlib.pyplot as plt

# from planet_sim.transit_toolbox import generate_sigma
from planet_sim.transit_toolbox import bayes_to_pvalue
from planet_sim.transit_toolbox import bayes_to_sigma
# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_1_compact_retrieval.pkl'
# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_noise125_compact_retrieval.pkl'
# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_mixedcase_compact_retrieval.pkl'
# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_mixedcase_500smpl_compact_retrieval.pkl'

filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/macdonald_H2OCH4NH3HCN_1_compact_retrieval.pkl'

with open(filename, 'rb') as f:
    ntrue_archive = pickle.load(f)

# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/macdonald_H2OCH4_0_compact_retrieval.pkl'

# updated result
filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/macdonald_H2OCH4_1_compact_retrieval.pkl'

with open(filename, 'rb') as f:
    nfalse_archive = pickle.load(f)


# labeling stuff
sigma_1 = 0.9  # sigma = 2
sigma_2 = 5.0  # sigma = 3.6
sigma_3 = 11.0  # sigma = 5
xs = np.array([sigma_1, sigma_2, sigma_3])
ys = np.zeros(xs.shape) + 0.001
labels = ['2σ', '3.6σ', '5σ']


'''Nitrogen detection'''
# scratch space
# 'True Model: water-methane mix'
# 'True Model: methane-only (no water)'

# nitrogen is present
delta_logz_water_nitrotrue = ntrue_archive['logz_full'] - ntrue_archive['logz_h2och4']

# nitrogen is not present
delta_logz_water_nitrofalse = nfalse_archive['logz_nitrogen'] - nfalse_archive['logz_h2och4']

# hist_fig, hist_ax = plt.subplots(1, figsize=(12, 6))
hist_fig, hist_ax = plt.subplots(1, 1, figsize=(8, 6))
hist_ax.hist(delta_logz_water_nitrotrue, bins=25, density=True, alpha=0.6, label='True Model: H2O-CH4-NH3-HCN')
hist_ax.hist(delta_logz_water_nitrofalse, bins=25, density=True, alpha=0.6, label='True Model: H2O-CH4')
hist_fig.suptitle('Nitrogen detection\n(Model Comparison: H2OCH4 vs H2OCH4-NH3HCN atmosphere)')
hist_ax.set_xlabel('Bayes Evidence Ratio (Δlog(z))')
hist_ax.set_ylabel('Probability Density')
hist_ax.axvline(0.9, label='σ thresholds for Nitrogen detection', color='b')
hist_ax.axvline(5.0, color='b')
hist_ax.axvline(11.0, color='b')
# hist_ax.axvline(-0.9, label='naive σ thresholds for Water rejection', color='g')
# hist_ax.axvline(-5.0, color='g')
# hist_ax.axvline(-11.0, color='g')


for x, y, label in zip(xs, ys, labels):
    if x < 0:
        ha = 'right'
    else:
        ha = 'left'
    hist_ax.annotate(label,
                     (x, y),
                     ha=ha)

hist_ax.legend()
hist_ax.set_xlim(-2.5, 20)




threshhold = 3.6  # 3 sigma
nitro_falsenegative_rate = np.sum(delta_logz_water_mixedtrue < threshhold)/delta_logz_water_mixedtrue.size
print('True positive rate:', 1-nitro_falsenegative_rate)

# water is not present
delta_logz_water_ch4true = nfalse_archive['logz_full'] - nfalse_archive['logz_h2och4']
nitro_falsepositive_rate = np.sum(delta_logz_water_ch4true > threshhold)/delta_logz_water_ch4true.size
print('False positive rate:', nitro_falsepositive_rate)


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

