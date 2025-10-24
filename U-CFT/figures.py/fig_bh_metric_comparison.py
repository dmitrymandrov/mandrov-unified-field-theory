import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 12

M = 1.0
eps = 0.5   
lam = 0.1
gamma = 0.5

r = np.linspace(0, 10, 1000)

f_schw = 1 - 2*M/r
f_coh = 1 - (2*M*r)/(r**2 + eps**2) + (lam/6.0)*r**2*(1 - np.exp(-gamma*r))

plt.figure(figsize=(12,8))
plt.plot(r, f_coh, label=r"$f_{\mathrm{coh}}(r) = 1 - \frac{2Mr}{r^2+\epsilon^2} + \frac{\lambda}{6}r^2(1-e^{-\gamma r})$", color="blue")
plt.plot(r, f_schw, "--", label=r"$f_{\mathrm{Schw}}(r) = 1 - \frac{2M}{r}$", color="red")
plt.axhline(0, color="black", linewidth=0.8)

plt.legend(loc="lower right", fontsize=15, frameon=True)

param_text = (r"$M = 1.0$" "\n"
              r"$\epsilon = 0.5$" "\n"
              r"$\lambda = 0.1$" "\n"
              r"$\gamma = 0.5$" "\n"
              r"(illustrative values)")
plt.text(7.5, -10, param_text, fontsize=12, bbox=dict(facecolor="white", edgecolor="black"))

plt.title("Black hole metric functions: U-CFT and Schwarzschild", fontsize=24.5)
plt.xlabel(r"$r$", fontsize=21)
plt.ylabel(r"$f(r)$", fontsize=21)
plt.ylim(-20, 3)
plt.grid(True, alpha=0.3)

plt.show()
