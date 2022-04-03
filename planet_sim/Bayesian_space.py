import numpy as np
import matplotlib.pyplot as plt

sigma_1 = 0.9  # sigma = 2
sigma_2 = 5.0  # sigma = 3.6
sigma_3 = 11.0  # sigma = 5

xs = np.array([-sigma_3, -sigma_2, -sigma_1, sigma_1, sigma_2, sigma_3])
ys = np.zeros(xs.shape) + 0.5

labels = ['5σ', '3.6σ', '2σ', '2σ', '3.6σ', '5σ']

fig, ax = plt.subplots(figsize=(8, 4))

ax.axvline(sigma_1, color='tab:blue')
ax.axvline(sigma_2, color='tab:blue')
ax.axvline(sigma_3, color='tab:blue')

ax.axvline(-sigma_1, color='tab:green')
ax.axvline(-sigma_2, color='tab:green')
ax.axvline(-sigma_3, color='tab:green')

xlims = ax.get_xlim()
ax.axvspan(-20, -sigma_3, alpha=1.0, color='tab:green')  # , label='Definitively favors Model A')
ax.axvspan(-sigma_3, -sigma_2, alpha=0.6, color='tab:green', label='Strongly favors Model A')
ax.axvspan(-sigma_2, -sigma_1, alpha=0.2, color='tab:green', label='Weakly favors Model A')
ax.axvspan(-sigma_1, sigma_1, alpha=0.25, color='tab:grey', label='No preference')
ax.axvspan(sigma_1, sigma_2, alpha=0.25, color='tab:blue', label='Weakly favors Model B')
ax.axvspan(sigma_2, sigma_3, alpha=0.6, color='tab:blue', label='Strongly favors Model B')
ax.axvspan(sigma_3, 20, alpha=1.0, color='tab:blue')

ax.set_xlim(xlims)

for x, y, label in zip(xs, ys, labels):
    if x < 0:
        ha = 'right'
    else:
        ha = 'left'
    ax.annotate(label,
                (x, y),
                ha=ha)

fig.suptitle('Model A vs Model B')
ax.set_xlabel('Bayesian Evidence Ratio (Δlog(z))')


ax.legend()



