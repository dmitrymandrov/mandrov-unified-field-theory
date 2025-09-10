import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 14

lam = 0.1
m_values = [0.5, 1.0, 2.0] 
a = np.linspace(0.01, 20, 1000)

plt.figure(figsize=(12,7))

colors = ["C0", "C1", "C2"]
for i, m in enumerate(m_values):
    C = 1/(1+a)
    Cp = -1/(1+a)**2
    
    rho_C = lam * (0.5 * Cp**2 + 0.5 * m**2 * C**2)
    p_C   = lam * (0.5 * Cp**2 - 0.5 * m**2 * C**2)
    w_C = p_C / rho_C
    
    plt.plot(a, w_C, label=fr"$w_\mathcal{{C}}(a), \ m={m}$", color=colors[i])

plt.axhline(-1, color="red", linestyle="--", label=r"$w=-1$")
plt.axhline(0, color="black", linestyle=":", alpha=0.6)

plt.legend(loc="upper right", fontsize=14, frameon=True)

param_text = (r"$\lambda = 0.1$" "\n"
              r"$V(\mathcal{C}(a)) = \frac{1}{2} m^2 (\mathcal{C}(a))^2$" "\n"
              r"$\mathcal{C}(a) = \frac{1}{1+a}$" "\n"
              r"(illustrative values)")
plt.gcf().text(0.72, 0.25, param_text, fontsize=12,
               bbox=dict(facecolor="white", edgecolor="black"))

plt.xlabel(r"Scale factor $a$", fontsize=20)
plt.ylabel(r"$w_\mathcal{C}(a)$", fontsize=20)
plt.title("Evolution of effective equation of state", fontsize=25)
plt.ylim(-1.2, 1.2)
plt.xlim(0, 20)
plt.grid(True, alpha=0.3)

plt.show()
