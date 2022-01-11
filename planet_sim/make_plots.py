import pickle
import numpy as np
import matplotlib.pyplot as plt

filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/macdonald_H2OCH4NH3HCN_1_compact_retrieval.pkl'

with open(filename, 'rb') as f:
    macdonald_H2OCH4NH3HCN_archive = pickle.load(f)
#
# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/macdonald_H2OCH4NH3HCN_R140_compact_retrieval.pkl'
#
# with open(filename, 'rb') as f:
#     macdonald_H2OCH4NH3HCN_r140_archive = pickle.load(f)
#     # macdonald_H2OCH4NH3HCN_archive = macdonald_H2OCH4NH3HCN_r140_archive

# need to save this in compact archive
rad_planet = 1.35  # in jovian radii
T = 1071  # Kelvin
log_f_h2o = -5.24
log_fch4 = -7.84
log_fnh3 = -6.03
log_fhcn = -6.35


'''
Histograms of the logz results
'''

# analyze the standard data
logz_full = macdonald_H2OCH4NH3HCN_archive['logz_full']
logz_h2och4 = macdonald_H2OCH4NH3HCN_archive['logz_h2och4']

delta_logz = logz_full - logz_h2och4


# analyze the results for when R is doubled
# logz_r140_full = macdonald_H2OCH4NH3HCN_r140_archive['logz_full']
# logz_r140_h2och4 = macdonald_H2OCH4NH3HCN_r140_archive['logz_h2och4']
#
# delta_logz_r140 = logz_r140_full - logz_r140_h2och4

# hist_fig, hist_ax = plt.subplots(1, figsize=(12, 6))
hist_fig, hist_ax = plt.subplots(1)
hist_ax.hist(delta_logz, bins=25)
hist_fig.suptitle('Δlog(z)')
hist_ax.set_xlabel('H2O-CH4-NH3-HCN vs H2O-CH4, R=70')
hist_ax.axvline(0.9, label='2 σ', color='r')
hist_ax.axvline(5.0, label='3.6 σ', color='r')
hist_ax.axvline(11.0, label='5 σ', color='r')
hist_ax.legend()

false_negs = delta_logz < 3.6
print('Number of false negatives', false_negs.sum())

# plot of just the R70 stuff
# hist_fig, hist_ax = plt.subplots(1, 1, figsize=(12, 6))
# hist_ax.hist(delta_logz)
# hist_fig.suptitle('Delta log(z)')
# hist_ax.set_xlabel('H2O-CH4-NH3-HCN vs H2O-CH4, R=70')
# hist_ax.axvline(0.9, label='2 σ', color='r')
# hist_ax.axvline(5.0, label='3.6 σ', color='r')
# hist_ax.axvline(11.0, label='5 σ', color='r')
# hist_ax.legend()


# hist_ax[0].axvline(11)

# hist_ax[1].hist(delta_logz_r140)
# hist_ax[1].set_xlabel('H2O-CH4-NH3-HCN vs H2O-CH4, R=140')
# hist_ax[1].axvline(0.9, label='2 σ', color='r')
# hist_ax[1].axvline(3.0, label='3 σ', color='r')
# hist_ax[1].axvline(11.0, label='5 σ', color='r')
# hist_ax[1].legend()

'''
Plot of the retrieved parameters
'''

full_quantiles = np.array(macdonald_H2OCH4NH3HCN_archive['full_quantiles'])
h2och4_quantiles = np.array(macdonald_H2OCH4NH3HCN_archive['h2och4_quantiles'])
num = np.arange(0, full_quantiles.shape[0]) + 1  # shift to 1-index instead of 0-index


# radius
full_r = full_quantiles[:, 0, 1]
full_r_lower = np.abs(full_quantiles[:, 0, 0] - full_r)
full_r_upper = np.abs(full_quantiles[:, 0, 2] - full_r)
h2och4_r = h2och4_quantiles[:, 0, 1]
h2och4_r_lower = np.abs(h2och4_quantiles[:, 0, 0] - h2och4_r)
h2och4_r_upper = np.abs(h2och4_quantiles[:, 0, 2] - h2och4_r)
true_r = 1.35


r_fig, r_ax = plt.subplots(figsize=(8,12))
r_fig.suptitle('95% confidence error bars')
r_ax.errorbar(full_r, delta_logz, xerr=(full_r_lower, full_r_upper), label='H2O-CH4-NH3-HCN (true model)', capsize=2.0, fmt='o')
r_ax.errorbar(h2och4_r, delta_logz, xerr=(h2och4_r_lower, h2och4_r_upper), label='H2O-CH4 model', capsize=2.0, fmt='o')
r_ax.axvline(true_r, color='r', label='True radius')
r_ax.legend(loc='best')
r_ax.set_xlabel('Planet Radius')
r_ax.set_ylabel('Delta log(z)')

# temperature
full_T = full_quantiles[:, 1, 1]
full_T_lower = np.abs(full_quantiles[:, 1, 0] - full_T)
full_T_upper = np.abs(full_quantiles[:, 1, 2] - full_T)
h2och4_T = h2och4_quantiles[:, 1, 1]
h2och4_T_lower = np.abs(h2och4_quantiles[:, 1, 0] - h2och4_T)
h2och4_T_upper = np.abs(h2och4_quantiles[:, 1, 2] - h2och4_T)
true_T = 1071

T_fig, T_ax = plt.subplots(figsize=(8,12))
T_fig.suptitle('95% confidence error bars')
T_ax.errorbar(full_T, delta_logz, xerr=(full_T_lower, full_T_upper), label='H2O-CH4-NH3-HCN (true model)', capsize=2.0, fmt='o')
T_ax.errorbar(h2och4_T, delta_logz, xerr=(h2och4_T_lower, h2och4_T_upper), label='H2O-CH4 model', capsize=2.0, fmt='o')
T_ax.axvline(true_T, color='r', label='True Temperature')
T_ax.legend(loc='best')
T_ax.set_xlabel('Planet Temperature')
T_ax.set_ylabel('Delta log(z)')

# analyze the H2O
full_h2o = full_quantiles[:, 2, 1]
full_h2o_lower = np.abs(full_quantiles[:, 2, 0] - full_h2o)
full_h2o_upper = np.abs(full_quantiles[:, 2, 2] - full_h2o)
h2och4_h2o = h2och4_quantiles[:, 2, 1]
h2och4_h2o_lower = np.abs(h2och4_quantiles[:, 2, 0] - h2och4_h2o)
h2och4_h2o_upper = np.abs(h2och4_quantiles[:, 2, 2] - h2och4_h2o)
true_h2o = -5.24

h2o_fig, h2o_ax = plt.subplots(figsize=(8,12))
h2o_fig.suptitle('95% confidence error bars')
h2o_ax.errorbar(full_h2o, delta_logz, xerr=(full_h2o_lower, full_h2o_upper), label='H2O-CH4-NH3-HCN (true model)', capsize=2.0, fmt='o')
h2o_ax.errorbar(h2och4_h2o, delta_logz, xerr=(h2och4_h2o_lower, h2och4_h2o_upper), label='H2O-CH4 model', capsize=2.0, fmt='o')
h2o_ax.axvline(true_h2o, color='r', label='True H2O fraction')
h2o_ax.legend(loc='best')
h2o_ax.set_xlabel('Log H2O fraction (log ppm)')
h2o_ax.set_ylabel('Delta log(z)')


full_ch4 = full_quantiles[:, 3, 1]
full_ch4_lower = np.abs(full_quantiles[:, 3, 0] - full_h2o)
full_ch4_upper = np.abs(full_quantiles[:, 3, 2] - full_h2o)
h2och4_ch4 = h2och4_quantiles[:, 3, 1]
h2och4_ch4_lower = np.abs(h2och4_quantiles[:, 3, 0] - h2och4_h2o)
h2och4_ch4_upper = np.abs(h2och4_quantiles[:, 3, 2] - h2och4_h2o)
true_ch4 = -7.84

ch4_fig, ch4_ax = plt.subplots(figsize=(8,12))
ch4_fig.suptitle('95% confidence error bars')
ch4_ax.errorbar(full_ch4, delta_logz, xerr=(full_ch4_lower, full_ch4_upper), label='H2O-CH4-NH3-HCN (true model)', capsize=2.0, fmt='o')
ch4_ax.errorbar(h2och4_h2o, delta_logz, xerr=(h2och4_ch4_lower, h2och4_ch4_upper), label='H2O-CH4 model', capsize=2.0, fmt='o')
ch4_ax.axvline(true_h2o, color='r', label='True CH4 fraction')
ch4_ax.legend(loc='best')
ch4_ax.set_xlabel('Log CH4 fraction (log ppm)')
ch4_ax.set_ylabel('Delta log(z)')

print('mean methane', h2och4_ch4.mean())



