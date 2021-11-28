import pickle
import numpy as np
import matplotlib.pyplot as plt

filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/macdonald_H2OCH4NH3HCN_1_compact_retrieval.pkl'

with open(filename, 'rb') as f:
    macdonald_H2OCH4NH3HCN_archive = pickle.load(f)

filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/macdonald_H2OCH4_0_compact_retrieval.pkl'

with open(filename, 'rb') as f:
    macdonald_H2OCH4_archive = pickle.load(f)

# analyze the standard data
logz_N = macdonald_H2OCH4NH3HCN_archive['logz_full'], macdonald_H2OCH4NH3HCN_archive['logz_h2och4']
delta_logz_N = logz_N[0] - logz_N[1]

logz_noN = macdonald_H2OCH4_archive['logz_nitrogen'], macdonald_H2OCH4_archive['logz_h2och4']
delta_logz_noN = logz_noN[0] - logz_noN[1]


# hist_fig, hist_ax = plt.subplots(1, figsize=(12, 6))
hist_fig, hist_ax = plt.subplots(1)
hist_ax.hist(delta_logz_N, bins=25)
hist_fig.suptitle('Δlog(z), with nitrogen')
hist_ax.set_xlabel('H2O-CH4-NH3-HCN vs H2O-CH4, R=70')
hist_ax.axvline(0.9, label='2 σ', color='r')
hist_ax.axvline(5.0, label='3.6 σ', color='r')
hist_ax.axvline(11.0, label='5 σ', color='r')
hist_ax.legend()


hist_fig, hist_ax = plt.subplots(1, 1, figsize=(12, 6))
hist_ax.hist(delta_logz_noN)
hist_fig.suptitle('Δlog(z), without nitrogen')
hist_ax.set_xlabel('H2O-CH4-NH3-HCN vs H2O-CH4, R=70')
hist_ax.axvline(0.9, label='2 σ', color='r')
hist_ax.axvline(5.0, label='3.6 σ', color='r')
hist_ax.axvline(11.0, label='5 σ', color='r')
hist_ax.legend()


'''Make Receiver Operator Characteristic curves'''
dynamic_range = np.linspace(delta_logz_noN.min(), delta_logz_N.max(), num=50)

false_negs = np.empty(dynamic_range.shape)
for i, bayes_criteria in enumerate(dynamic_range):
    false_negs[i] = np.sum(delta_logz_N < bayes_criteria)
    # print(i, np.sum(delta_logz_N < bayes_criteria))

true_negs = np.empty(dynamic_range.shape)
for i, bayes_criteria in enumerate(dynamic_range):
    true_negs[i] = np.sum(delta_logz_noN < bayes_criteria)
    # print(i, np.sum(delta_logz_noN < bayes_criteria))

true_pos = np.empty(dynamic_range.shape)
for i, bayes_criteria in enumerate(dynamic_range):
    true_pos[i] = np.sum(delta_logz_N > bayes_criteria)

false_pos = np.empty(dynamic_range.shape)
for i, bayes_criteria in enumerate(dynamic_range):
    false_pos[i] = np.sum(delta_logz_noN > bayes_criteria)

roc_fig, (pos_ax, negs_ax) = plt.subplots(1, 2, figsize=(12, 6))

negs_ax.step(false_negs, true_negs)
negs_ax.set_xlabel('False Negative')
negs_ax.set_ylabel('True Negative')

pos_ax.step(false_pos, true_pos)
pos_ax.set_xlabel('False Positive')
pos_ax.set_ylabel('True Positive')
