"""Show aggregate early-window protocol effects without individual record profiles."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT

root = RUN_ROOT / 'results/record_reference_control'
assert (root / 'verification.json').exists()
folds = pd.read_csv(root / 'scores.csv')
summary = pd.read_csv(root / 'summary.csv').set_index('protocol')
names = ['trial_shared', 'record_disjoint']
fig, ax = plt.subplots(figsize=(7.2, 4.3), layout='constrained')
for j, name in enumerate(names):
    s = summary.loc[name]
    ax.bar(j, 100*s['mean'], width=.6, color=['#236a9b', '#bd6e3c'][j],
           yerr=100*s['fold_std'], capsize=5, alpha=.85)
    ax.scatter([j]*4, 100*folds.loc[folds.protocol == name, 'accuracy'],
               s=28, color='black', zorder=3)
    ax.text(j, 100*s['mean']+100*s['fold_std']+3,
            f"{100*s['mean']:.2f}%", ha='center')
ax.axhline(12.5, ls='--', lw=1, color='gray', label='Uniform 8-class reference')
ax.set(xticks=[0, 1], xticklabels=['Trial split\nrecords shared', 'Record split\nrecords disjoint'],
       ylim=(0, 100), ylabel='Early-window accuracy (%)',
       title='Same initial 200 ms, 905 features and XGB 400 rounds')
ax.legend(loc='upper right', fontsize=8)
ax.grid(axis='y', alpha=.2)
fig.savefig(root / 'protocol_comparison.png', dpi=170)
plt.close(fig)
