import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 12

def f_coh(r, M=1.0, eps=0.5, lam=0.1, gamma=0.5):
    return 1.0 - (2.0*M*r)/(r**2 + eps**2) + (lam/6.0)*r**2*(1.0 - np.exp(-gamma*r))

M = 1.0
lam = 0.1
gamma = 0.5

eps_list = [0.1, 0.5, 1.0]

r_min = 1e-4
r_max = 20.0
N = 20000
r = np.linspace(r_min, r_max, N)

def find_roots_on_grid(x, y):
    roots = []
    for i in range(len(y)-1):
        y1, y2 = y[i], y[i+1]
        if np.isnan(y1) or np.isnan(y2):
            continue
        if y1 == 0.0:
            roots.append(x[i])
        elif y1*y2 < 0:
            a, b = x[i], x[i+1]
            for _ in range(30):
                m = 0.5*(a+b)
                fm = np.interp(m, x, y)
                if y1*fm <= 0:
                    b, y2 = m, fm
                else:
                    a, y1 = m, fm
            roots.append(0.5*(a+b))
    roots_sorted = []
    for r0 in roots:
        if not any(abs(r0 - r1) < 1e-3 for r1 in roots_sorted):
            roots_sorted.append(r0)
    return sorted(roots_sorted)

fig, ax = plt.subplots(figsize=(10,6))

colors = ['C0', 'C1', 'C2', 'C3']
for idx, eps in enumerate(eps_list):
    y = f_coh(r, M=M, eps=eps, lam=lam, gamma=gamma)
    ax.plot(r, y, label=fr"$f_{{\mathrm{{coh}}}}(r), \ \epsilon={eps}$", color=colors[idx%len(colors)])
    roots = find_roots_on_grid(r, y)
    roots = [root for root in roots if 0 < root < r_max]
    for j, root in enumerate(roots):
        ax.plot(root, f_coh(root, M=M, eps=eps, lam=lam, gamma=gamma),
                marker='o', markersize=6, color=colors[idx%len(colors)])
        if len(roots) == 2:
            label = r"$r_-$" if j == 0 else r"$r_+$"
        else:
            label = r"$r_h$"
        offset = -0.5 - idx*1.5 - j*0.5
        ax.text(root, offset, f"{label}={root:.2f}",
                fontsize=12, color=colors[idx%len(colors)],
                horizontalalignment='center')

ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.6)

ax.set_xlim(-0.3, 6)
ax.set_ylim(-10, 3)
ax.set_xlabel(r"$r$", fontsize=21)
ax.set_ylabel(r"$f_{\mathrm{coh}}(r)$", fontsize=21)
ax.set_title("Horizon structure of U--CFT metric", fontsize=24.5)
ax.grid(True, alpha=0.3)

plt.legend(loc="lower right", fontsize=15, frameon=True)

param_text = (r"$M=1.0$" "\n"
              r"$\lambda=0.1$" "\n"
              r"$\gamma=0.5$" "\n"
              r"(illustrative values)")
plt.gcf().text(0.75, 0.55, param_text, fontsize=12,
               bbox=dict(facecolor="white", edgecolor="black"))

plt.tight_layout()
plt.show()
