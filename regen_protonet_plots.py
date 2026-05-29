"""
Reconstruct ProtoNet training curves from the actual logged checkpoint values.
Produces clean publication-quality plots.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Actual logged checkpoint data (every 300 episodes) ---
episodes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]

data = {
    'CNN': {
        'loss': [0.7574, 0.7085, 0.6296, 0.4749, 0.5006, 0.3705, 0.3999, 0.3874, 0.2090, 0.3121],
        'acc':  [71.6,   78.6,   83.5,   85.7,   89.0,   90.6,   93.4,   95.0,   96.4,   96.6],
    },
    'CNN+Attention': {
        'loss': [0.8816, 0.5660, 0.4033, 0.5210, 0.6309, 0.4702, 0.3444, 0.2815, 0.2867, 0.2773],
        'acc':  [74.8,   81.8,   86.0,   87.7,   89.3,   92.1,   94.7,   96.6,   97.7,   98.5],
    },
    'CNN+Mamba': {
        'loss': [0.8370, 0.7715, 0.6561, 0.4097, 0.2778, 0.2275, 0.2027, 0.2157, 0.1986, 0.2008],
        'acc':  [64.6,   78.0,   87.8,   93.0,   97.7,   99.4,   99.8,  100.0,   99.9,   99.9],
    },
    'CNN+Attn+Mamba': {
        'loss': [0.7152, 0.6981, 0.4614, 0.4524, 0.2740, 0.2179, 0.1996, 0.2484, 0.1995, 0.1984],
        'acc':  [66.7,   76.3,   84.3,   90.3,   96.1,   99.5,   99.7,   99.9,  100.0,  100.0],
    },
}

holdout = {
    'CNN':            {'mean': 90.67, 'max': 96.67, 'min': 85.00},
    'CNN+Attention':  {'mean': 92.28, 'max': 98.33, 'min': 85.00},
    'CNN+Mamba':      {'mean': 91.23, 'max': 98.33, 'min': 83.33},
    'CNN+Attn+Mamba': {'mean': 92.77, 'max': 98.33, 'min': 85.00},
}

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# ─── Plot 1: Training Curves ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ProtoNet — Training Curves', fontsize=14, fontweight='bold')

for (name, d), color in zip(data.items(), colors):
    axes[0].plot(episodes, d['loss'], marker='o', markersize=4, label=name, color=color)
    axes[1].plot(episodes, d['acc'],  marker='o', markersize=4, label=name, color=color)

axes[0].set(title='Loss vs Episodes', xlabel='Episode', ylabel='Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.4)
axes[0].set_xticks(episodes)

axes[1].set(title='Accuracy vs Episodes', xlabel='Episode', ylabel='Accuracy (%)')
axes[1].legend(); axes[1].grid(True, alpha=0.4)
axes[1].set_xticks(episodes)
axes[1].set_ylim(55, 105)

plt.tight_layout()
plt.savefig('fsl_protonet_best_curves.png', dpi=150)
plt.close()
print("Saved fsl_protonet_best_curves.png")

# ─── Plot 2: Holdout bar chart ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('ProtoNet — 5-Shot Holdout Accuracy (100 Trials)', fontsize=14, fontweight='bold')

names  = list(holdout.keys())
means  = [holdout[n]['mean'] for n in names]
maxs   = [holdout[n]['max']  for n in names]
mins   = [holdout[n]['min']  for n in names]
errs_lo = [m - mn for m, mn in zip(means, mins)]
errs_hi = [mx - m  for m, mx in zip(means, maxs)]

bars = ax.bar(names, means, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7,
              yerr=[errs_lo, errs_hi], capsize=6, error_kw={'linewidth': 1.5})

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 0.3, f'{mean:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(80, 102)
ax.grid(True, axis='y', alpha=0.4)
ax.set_title('Mean ± (min/max) over 100 trials', fontsize=11)
plt.tight_layout()
plt.savefig('fsl_protonet_best_holdout.png', dpi=150)
plt.close()
print("Saved fsl_protonet_best_holdout.png")
print("Done!")
