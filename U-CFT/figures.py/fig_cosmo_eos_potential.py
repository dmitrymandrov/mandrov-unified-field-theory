import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 14

lam = 0.1
m_values = [0.5, 1.0, 2.0]  
a = np.linspace(0, 20, 3000)  

H0 = 1.0 

def w_coherence(a, lam, m, H0=1.0):
    C = 1.0 / (1.0 + a)           
    dC_da = -1.0 / (1.0 + a)**2   
    
    H = H0 * a**(-1.5)            
    C_dot = dC_da * a * H          
    
    rho_C = 0.5 * lam * C_dot**2 + 0.5 * m**2 * C**2
    p_C   = 0.5 * lam * C_dot**2 - 0.5 * m**2 * C**2
    return p_C / rho_C

plt.figure(figsize=(12, 7))
colors = ["C0", "C1", "C2"]

for i, m in enumerate(m_values):
    w_C = w_coherence(a, lam, m, H0)
    plt.plot(
        a, w_C,
        label=fr"$w_\mathcal{{C}}(a), \ m={m}$",
        color=colors[i]
    )

plt.axhline(-1, color="red", linestyle="--", label=r"$w=-1$")
plt.axhline(0, color="black", linestyle=":", alpha=0.6)

plt.xlabel(r"Scale factor $a$", fontsize=21)
plt.ylabel(r"$w_\mathcal{C}(a)$", fontsize=21)
plt.title(r"Evolution of effective equation of state $w_\mathcal{C}(a)$", fontsize=24.5)

plt.xlim(-0.1, 4)
plt.ylim(-1.2, 1.2)
plt.grid(True, alpha=0.3)

param_text = (
    r"$\lambda = 0.1$" "\n"
    r"$\mathcal{C}(a) = \frac{1}{1+a}$" "\n"
    r"$\frac{d\mathcal{C}}{da} = -\frac{1}{(1+a)^2}$" "\n"
    r"$\mathcal{C}'(a) = \frac{d\mathcal{C}}{da} \cdot a H(a)$" "\n"
    r"$V(\mathcal{C}(a)) = \frac{1}{2} m^2 (\mathcal{C}(a))^2$" "\n" 
    r"$H(a) \propto a^{-\frac{3}{2}}$" "\n"
    r"(illustrative values)"
)
plt.gcf().text(
    0.68, 0.25, param_text, fontsize=12,
    bbox=dict(facecolor="white", edgecolor="black")
)

plt.legend(loc="upper right", fontsize=15, frameon=True)
plt.show()
