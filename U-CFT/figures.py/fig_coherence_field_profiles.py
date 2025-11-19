import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 12

def C_field(r, r0=1.0, n=2):
    return 1.0 / (1.0 + (r/r0)**n)

r = np.linspace(0.01, 20, 2000)

horizons = [1.2, 3.5] 

fig, ax = plt.subplots(figsize=(10,6))

profiles = [(1, 2), (2, 2), (1, 4)]  
colors = ['C0','C1','C2']

for (r0, n), c in zip(profiles, colors):
    ax.plot(r, C_field(r, r0=r0, n=n),
            label=fr"$\mathcal{{C}}(r) = \frac{{1}}{{1+(r/r_0)^n}}, \ r_0={r0}, \ n={n}$",
            color=c)

for h in horizons:
    ax.axvline(h, color='k', linestyle='--', alpha=0.6)
    ax.text(h, 0.05, fr"$r_h={h:.1f}$",
            rotation=90, verticalalignment='bottom', horizontalalignment='right')

ax.set_xlim(0, 10)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$r$", fontsize=21)
ax.set_ylabel(r"$\mathcal{C}(r)$", fontsize=21)
ax.set_title("Illustrative coherence field profiles around a black hole", fontsize=24.5)
ax.grid(True, alpha=0.3)

plt.legend(loc="upper right", fontsize=15, frameon=True)
plt.tight_layout()
plt.show()
