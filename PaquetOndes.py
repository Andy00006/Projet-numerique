import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import simpson

HBAR = 1.0
MASS = 1.0
SIGMA = 0.2
K0 = 1.0
X0 = -20.0

NX = 2000
X_MIN = -40.0
X_MAX = 60.0
x = np.linspace(X_MIN, X_MAX, NX)
dx = x[1] - x[0]

dt = 0.9 * (dx**2)
NT = 4000

V0 = 0.7
BARRIER_START = 0.0
BARRIER_END = 3.0

V = np.zeros(NX)
barrier_mask = (x >= BARRIER_START) & (x <= BARRIER_END)
V[barrier_mask] = V0

def Compute_gaussian(k, mean, std):
    return np.exp(-((k - mean) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

def Compute_gaussian_wp(x_arr, t, sigma=SIGMA, k0=K0, x0=X0):
    a = 1.0 / sigma
    a2 = a**2
    terme = MASS * a2 + 2j * HBAR * t
    inv_t = 1.0 / terme
    amp = np.sqrt(2 * MASS * a * inv_t / np.sqrt(2 * np.pi))
    
    psi_libre = amp * np.exp(inv_t * MASS * (a2 * k0 + 2j * (x_arr - x0))**2 / 4.0) * np.exp(-a2 * (k0**2) / 4.0)
    return psi_libre

def Check_normalization(prob_density, x_arr):
    return simpson(prob_density, x_arr)

psi_init = Compute_gaussian_wp(x, 0.0)
re_psi = np.real(psi_init)
im_psi = np.imag(psi_init)

re_history = np.zeros((NT, NX))
im_history = np.zeros((NT, NX))
prob_history = np.zeros((NT, NX))
norm_history = np.zeros(NT)
t_axis = np.arange(NT) * dt

s = dt / (2.0 * dx**2)

for n in range(NT):
    im_psi[1:-1] = im_psi[1:-1] + s * (re_psi[2:] + re_psi[:-2] - 2.0 * re_psi[1:-1]) - dt * V[1:-1] * re_psi[1:-1]
    re_psi[1:-1] = re_psi[1:-1] - s * (im_psi[2:] + im_psi[:-2] - 2.0 * im_psi[1:-1]) + dt * V[1:-1] * im_psi[1:-1]
    
    re_history[n, :] = re_psi
    im_history[n, :] = im_psi
    prob_history[n, :] = re_psi**2 + im_psi**2
    norm_history[n] = Check_normalization(prob_history[n, :], x)

idx_traversee = -1
for n in range(NT):
    idx_max = np.argmax(prob_history[n, :])
    if x[idx_max] >= BARRIER_END:
        idx_traversee = n
        break

if idx_traversee != -1:
    tau_t_num = t_axis[idx_traversee]
    print(f"Norme finale : {norm_history[-1]:.6f}")
    print(f"Temps de traversée numérique mesuré (tau_t) : {tau_t_num:.4f} s")
else:
    print("Le paquet d'ondes n'a pas franchi la barrière dans la fenêtre de temps de simulation.")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("Position (x)")
ax.set_ylabel("Amplitude / Potentiel")
ax.set_title("Simulation FDTD : Paquet d'ondes face à la barrière de potentiel")

line_prob, = ax.plot([], [], label=r'$|\Psi(x,t)|^2$', color='blue', lw=2)
line_re, = ax.plot([], [], label=r'$Re(\Psi)$', color='green', linestyle='--', alpha=0.6)
line_im, = ax.plot([], [], label=r'$Im(\Psi)$', color='orange', linestyle='--', alpha=0.6)
ax.fill_between(x, 0, V / V0 if V0 != 0 else 0, where=V > 0, color='red', alpha=0.3, label='Barrière de potentiel')
ax.legend(loc='upper right')

def init_anim():
    line_prob.set_data([], [])
    line_re.set_data([], [])
    line_im.set_data([], [])
    return line_prob, line_re, line_im

def update_anim(frame):
    line_prob.set_data(x, prob_history[frame, :])
    line_re.set_data(x, re_history[frame, :])
    line_im.set_data(x, im_history[frame, :])
    return line_prob, line_re, line_im

anim = FuncAnimation(fig, update_anim, frames=range(0, NT, 20), init_func=init_anim, blit=True, interval=30)
plt.show()
