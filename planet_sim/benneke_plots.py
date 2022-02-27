import pickle
import numpy as np
import matplotlib.pyplot as plt

# filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_1_compact_retrieval.pkl'
filename = '/home/lee/PycharmProjects/spectrum-transmission-sim/planet_sim/sim_results/benneke_h2o_vs_ch4_noise125_compact_retrieval.pkl'
with open(filename, 'rb') as f:
    benneke_archive = pickle.load(f)

h2otrue_archive = benneke_archive['h2o_true']
ch4true_archive = benneke_archive['ch4_true']

# analyze the standard data
delta_logz_h2otrue = h2otrue_archive['logz_h2o'] - h2otrue_archive['logz_ch4']


delta_logz_ch4true = ch4true_archive['logz_h2o'] - ch4true_archive['logz_ch4']


# hist_fig, hist_ax = plt.subplots(1, figsize=(12, 6))
hist_fig, hist_ax = plt.subplots(1, 1, figsize=(8, 6))
hist_ax.hist(delta_logz_h2otrue, bins=25)
hist_fig.suptitle('Δlog(z), with water')
hist_ax.set_xlabel('H2O vs CH4, R=70')
hist_ax.axvline(0.9, label='2 σ', color='r')
hist_ax.axvline(5.0, label='3.6 σ', color='r')
hist_ax.axvline(11.0, label='5 σ', color='r')
hist_ax.legend()


hist_fig, hist_ax = plt.subplots(1, 1, figsize=(8, 6))
hist_ax.hist(delta_logz_ch4true)
hist_fig.suptitle('Δlog(z), with methane')
hist_ax.set_xlabel('H2O vs CH4, R=70')
hist_ax.axvline(0.9, label='2 σ', color='r')
hist_ax.axvline(5.0, label='3.6 σ', color='r')
hist_ax.axvline(11.0, label='5 σ', color='r')
hist_ax.legend()


'''Make Receiver Operator Characteristic curves'''
dynamic_range = np.linspace(delta_logz_ch4true.min(), delta_logz_h2otrue.max(), num=50)

false_negs = np.empty(dynamic_range.shape)
for i, bayes_criteria in enumerate(dynamic_range):
    false_negs[i] = np.sum(delta_logz_h2otrue < bayes_criteria) / delta_logz_h2otrue.size
    # print(i, np.sum(delta_logz_h2otrue < bayes_criteria))

true_negs = np.empty(dynamic_range.shape)
for i, bayes_criteria in enumerate(dynamic_range):
    true_negs[i] = np.sum(delta_logz_ch4true < bayes_criteria) / delta_logz_ch4true.size
    # print(i, np.sum(delta_logz_ch4true < bayes_criteria))

true_pos = np.empty(dynamic_range.shape)
for i, bayes_criteria in enumerate(dynamic_range):
    true_pos[i] = np.sum(delta_logz_h2otrue > bayes_criteria) / delta_logz_h2otrue.size

false_pos = np.empty(dynamic_range.shape)
for i, bayes_criteria in enumerate(dynamic_range):
    false_pos[i] = np.sum(delta_logz_ch4true > bayes_criteria) / delta_logz_ch4true.size

# roc_fig, (pos_ax, negs_ax) = plt.subplots(1, 2, figsize=(12, 6))
roc_fig, pos_ax = plt.subplots(1, figsize=(6,6))
# negs_ax.step(false_negs, true_negs)
# negs_ax.set_xlabel('False Negative')
# negs_ax.set_ylabel('True Negative')

pos_ax.step(false_pos, true_pos)
pos_ax.set_xlabel('False Positive')
pos_ax.set_ylabel('True Positive')
diag = np.linspace(0, 1)
pos_ax.plot(diag, diag, color='r')

true_pos_2sigma = np.sum(delta_logz_h2otrue > 0.9) / delta_logz_h2otrue.size
false_pos_2sigma = np.sum(delta_logz_ch4true > 0.9) / delta_logz_ch4true.size
pos_ax.scatter(false_pos_2sigma, true_pos_2sigma, color='r', marker='d', label='Δlog(z) = 0.9 (2σ)')

true_pos_36sigma = np.sum(delta_logz_h2otrue > 5) / delta_logz_h2otrue.size
false_pos_36sigma = np.sum(delta_logz_ch4true > 5) / delta_logz_ch4true.size
pos_ax.scatter(false_pos_36sigma, true_pos_36sigma, color='purple', marker='d', label='Δlog(z) = 5 (3.6σ)')

true_pos_36sigma = np.sum(delta_logz_h2otrue > 11) / delta_logz_h2otrue.size
false_pos_36sigma = np.sum(delta_logz_ch4true > 11) / delta_logz_ch4true.size
pos_ax.scatter(false_pos_36sigma, true_pos_36sigma, color='black', marker='d', label='Δlog(z) = 11 (5σ)')

pos_ax.axvline(0.05, linestyle='--', color='r', label='5% false positive rate')
pos_ax.axhline(0.95, linestyle='--', color='b', label='95% true positive rate')
pos_ax.legend()
pos_ax.set_title('ROC curve for nitrogen detection')
