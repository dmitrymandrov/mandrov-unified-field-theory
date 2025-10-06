import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 12

def f_coh(r, M=1.0, eps=0.5, lam=0.1, gamma=0.5):
    return 1.0 - (2.0*M*r)/(r**2 + eps**2) + (lam/6.0)*r**2*(1.0 - np.exp(-gamma*r))

def find_roots(x, y):
    roots = []
    for i in range(len(y)-1):
        if np.isnan(y[i]) or np.isnan(y[i+1]): 
            continue
        if y[i] == 0: 
            roots.append(x[i])
        elif y[i]*y[i+1] < 0:
            a, b = x[i], x[i+1]
            for _ in range(30):
                m = 0.5*(a+b)
                fm = np.interp(m, x, y)
                if y[i]*fm <= 0:
                    b = m
                else:
                    a = m
            roots.append(0.5*(a+b))
    return sorted(set([round(r,3) for r in roots]))

M, lam, gamma = 1.0, 0.1, 0.5
eps_values = np.linspace(0.05, 2, 3000)
r_range = np.linspace(1e-3, 20, 20000)

r_minus, r_plus = [], []

for eps in eps_values:
    y = f_coh(r_range, M, eps, lam, gamma)
    roots = find_roots(r_range, y)
    if len(roots) == 2:
        r_minus.append(roots[0])
        r_plus.append(roots[1])
    else:
        r_minus.append(np.nan)
        r_plus.append(np.nan)

plt.figure(figsize=(10,6))
plt.plot(eps_values, r_plus, label=r"$r_+(\epsilon)$", color="blue")
plt.plot(eps_values, r_minus, label=r"$r_-(\epsilon)$", color="red")

plt.xlabel(r"$\epsilon$", fontsize=20)
plt.ylabel(r"Horizon radius $r_h$", fontsize=20)
plt.title("Bifurcation of Horizons in U-CFT Black Holes", fontsize=25)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14, frameon=True)

param_text = (r"$M=1.0$" "\n"
              r"$\lambda=0.1$" "\n"
              r"$\gamma=0.5$" "\n"
              r"(illustrative values)")
plt.gcf().text(0.3, 0.4, param_text, fontsize=12,
               bbox=dict(facecolor="white", edgecolor="black"))

plt.show()
