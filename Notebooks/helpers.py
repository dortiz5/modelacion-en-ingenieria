import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def configure_plot_style():
    gray = '#5c5c5c'
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["text.color"] = gray
    plt.rcParams["xtick.color"] = gray
    plt.rcParams["ytick.color"] = gray
    plt.rcParams["axes.labelcolor"] = gray
    plt.rcParams["axes.edgecolor"] = gray
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 13

    plt.rcParams.update(
        {
            'text.usetex': False,
            'mathtext.fontset': 'stix',
        }
    )


def save_simulation(df, model_name, data_dir='../data'):
    data_path = Path(data_dir) / model_name
    data_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    existing = list(data_path.glob(f'{model_name}_{date_str}_*.csv'))
    counter = len(existing) + 1
    filename = f'{model_name}_{date_str}_{counter:03d}.csv'
    filepath = data_path / filename
    df.to_csv(filepath, index=False)
    return filepath

def generate_input(t, kind='none', **kwargs):
    t = np.asarray(t)                    # asegura que t sea array de numpy
    u = np.zeros_like(t, dtype=float)    # inicializa la señal en ceros

    if kind == 'none':
        return u                         # señal nula (sin excitación)

    elif kind == 'step':
        t_step = kwargs.get('t_step', t[len(t)//4])  # instante del salto (default: 1/4 del tiempo)
        u0 = kwargs.get('u0', 0.0)       # nivel antes del salto
        u1 = kwargs.get('u1', 1.0)       # nivel después del salto
        u[:] = u0                        # rellena toda la señal con u0
        u[t >= t_step] = u1              # cambia a u1 a partir de t_step
        return u

    elif kind == 'multisine':
        freqs = np.atleast_1d(kwargs.get('freqs', [0.1, 0.3, 0.7]))   # frecuencias [Hz]
        amps = np.atleast_1d(kwargs.get('amps', np.ones_like(freqs))) # amplitudes por componente
        phases = np.atleast_1d(kwargs.get('phases', np.zeros_like(freqs)))  # fases por componente
        for f, a, p in zip(freqs, amps, phases):
            u += a * np.sin(2*np.pi*f*t + p)   # suma cada sinusoide
        return u

    elif kind == 'prbs':
        n_bits = kwargs.get('n_bits', 8)             # tamaño del registro LFSR
        t_bit = kwargs.get('t_bit', (t[-1]-t[0])/50) # duración de cada bit [s]
        amplitude = kwargs.get('amplitude', 1.0)     # amplitud de la señal binaria
        seed = kwargs.get('seed', 0)                 # semilla para reproducibilidad
        rng = np.random.default_rng(seed)
        reg = rng.integers(1, 2**n_bits)             # estado inicial del registro (no cero)
        n_samples_per_bit = max(1, int(round(t_bit/(t[1]-t[0]))))  # muestras por bit
        n_bits_total = int(np.ceil(len(t)/n_samples_per_bit))      # cantidad de bits a generar
        bits = np.zeros(n_bits_total)
        for i in range(n_bits_total):
            b0 = reg & 1                             # bit menos significativo
            b1 = (reg >> 1) & 1                      # segundo bit
            new_bit = b0 ^ b1                        # realimentación XOR (taps en 0 y 1)
            bits[i] = 2*b0 - 1                       # mapea {0,1} a {-1,+1}
            reg = (reg >> 1) | (new_bit << (n_bits-1))  # desplaza e inserta el nuevo bit
        u = np.repeat(bits, n_samples_per_bit)[:len(t)] * amplitude  # expande a la longitud de t
        return u

    elif kind == 'chirp':
        f0 = kwargs.get('f0', 0.05)             # frecuencia inicial [Hz]
        f1 = kwargs.get('f1', 1.0)              # frecuencia final [Hz]
        amplitude = kwargs.get('amplitude', 1.0)
        T = t[-1] - t[0]                        # duración total
        k = (f1 - f0) / T                       # tasa de variación de frecuencia
        u = amplitude * np.sin(2*np.pi*(f0*t + 0.5*k*t**2))  # chirp lineal
        return u

    else:
        raise ValueError(f"kind '{kind}' no reconocido")
    
    
# ============================================================
# Filtro de Kalman lineal
# ============================================================
def KF(A, B, C, Q, R, u, y, x0, P0, Gamma=None):
    n_x = A.shape[0]
    n_y = C.shape[0]
    N   = y.shape[0]
    if Gamma is None:
        Gamma = np.eye(n_x)

    x_hist = np.zeros((N, n_x))
    P_hist = np.zeros((N, n_x, n_x))
    K_hist = np.zeros((N, n_x, n_y))

    x = x0.copy()
    P = P0.copy()
    x_hist[0] = x
    P_hist[0] = P

    for n in range(N - 1):
        # Predicción: n -> n+1
        x_pred = A @ x + (B @ u[n]).flatten()
        P_pred = A @ P @ A.T + Gamma @ Q @ Gamma.T

        # Corrección con y[n+1]
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        innov = y[n+1] - C @ x_pred
        x = x_pred + (K @ innov).flatten()
        P = (np.eye(n_x) - K @ C) @ P_pred

        x_hist[n+1] = x
        P_hist[n+1] = P
        K_hist[n+1] = K

    return x_hist, P_hist, K_hist


# ============================================================
# Función de graficación
# ============================================================
def plot_kf_results(x_true, y_meas, x_hist, P_hist, K_hist, dt, title=""):
    N = x_hist.shape[0]
    t = np.arange(N) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    # Posición
    axes[0, 0].plot(t, y_meas.flatten(), 'b-',  lw=1,   label='Medida')
    axes[0, 0].plot(t, x_hist[:, 0],     'g-',  lw=1.5, label='Estimada')
    axes[0, 0].plot(t, x_true[:, 0],     'k--', lw=1,   label='Real')
    axes[0, 0].set_title('Posición')
    axes[0, 0].set_xlabel('t [s]')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Velocidad
    axes[0, 1].plot(t, x_true[:, 1], 'b-', lw=1,   label='Real')
    axes[0, 1].plot(t, x_hist[:, 1], 'g-', lw=1.5, label='Estimada')
    axes[0, 1].set_title('Velocidad')
    axes[0, 1].set_xlabel('t [s]')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Traza de P
    trace_P = np.array([np.trace(P_hist[n]) for n in range(N)])
    axes[1, 0].plot(t, trace_P, 'b-', lw=1.5)
    axes[1, 0].set_title('trace(P)')
    axes[1, 0].set_xlabel('t [s]')
    axes[1, 0].grid(alpha=0.3)

    # Norma de K
    norm_K = np.array([np.linalg.norm(K_hist[n]) for n in range(N)])
    axes[1, 1].plot(t, norm_K, 'b-', lw=1.5)
    axes[1, 1].set_title('norm(K)')
    axes[1, 1].set_xlabel('t [s]')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# Filtro de Kalman Extendido
# ============================================================
def ExKF(f, h, df_dx, dh_dx, Q, R, u, y, x0, P0, Gamma=None):
    n_x = x0.shape[0]
    n_y = y.shape[1]
    N   = y.shape[0]
    if Gamma is None:
        Gamma = np.eye(n_x)

    x_hist = np.zeros((N, n_x))
    P_hist = np.zeros((N, n_x, n_x))
    K_hist = np.zeros((N, n_x, n_y))

    x = x0.copy()
    P = P0.copy()
    x_hist[0] = x
    P_hist[0] = P

    for n in range(N - 1):
        # Jacobiano de f evaluado en x[n|n]
        A = df_dx(x, u[n])

        # Predicción no lineal: n -> n+1
        x_pred = f(x, u[n])
        P_pred = A @ P @ A.T + Gamma @ Q @ Gamma.T

        # Jacobiano de h evaluado en x[n+1|n]
        C = dh_dx(x_pred, u[n+1])

        # Corrección con y[n+1]
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        innov = y[n+1] - h(x_pred, u[n+1])
        x = x_pred + (K @ innov).flatten()
        P = (np.eye(n_x) - K @ C) @ P_pred

        x_hist[n+1] = x
        P_hist[n+1] = P
        K_hist[n+1] = K

    return x_hist, P_hist, K_hist

# ============================================================
# Función de graficación
# ============================================================
def plot_ekf_results(x_true, y_meas, x_hist, P_hist, K_hist, dt, title=""):
    N = x_hist.shape[0]
    t = np.arange(N) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    # Presas (medidas)
    axes[0, 0].plot(t, y_meas.flatten(), 'b-',  lw=1,   label='Medida')
    axes[0, 0].plot(t, x_hist[:, 0],     'g-',  lw=1.5, label='Estimada')
    axes[0, 0].plot(t, x_true[:, 0],     'k--', lw=1,   label='Real')
    axes[0, 0].set_title('Presas $x_1$ (medidas)')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Depredadores (no medidos)
    axes[0, 1].plot(t, x_true[:, 1], 'b-', lw=1,   label='Real')
    axes[0, 1].plot(t, x_hist[:, 1], 'g-', lw=1.5, label='Estimada')
    axes[0, 1].set_title('Depredadores $x_2$ (no medidos)')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Traza de P
    trace_P = np.array([np.trace(P_hist[n]) for n in range(N)])
    axes[1, 0].plot(t, trace_P, 'b-', lw=1.5)
    axes[1, 0].set_title('trace(P)')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].grid(alpha=0.3)

    # Norma de K
    norm_K = np.array([np.linalg.norm(K_hist[n]) for n in range(N)])
    axes[1, 1].plot(t, norm_K, 'b-', lw=1.5)
    axes[1, 1].set_title('norm(K)')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def LS(Phi, y_vec):
	theta_hat = np.linalg.pinv(Phi) @ y_vec
	y_hat     = Phi @ theta_hat
	residuos  = y_vec - y_hat
	return theta_hat, y_hat, residuos


def RLS(Phi, y_vec, lam=1.0, alpha=1e6):
	M, p      = Phi.shape
	theta     = np.zeros(p)
	P         = alpha * np.eye(p)
	theta_hist = np.zeros((M, p))
	
	for n in range(M):
		phi_n = Phi[n, :]
		y_n   = y_vec[n]
		denom = lam + phi_n @ P @ phi_n
		K     = P @ phi_n / denom
		theta = theta + K * (y_n - phi_n @ theta)
		P     = (1/lam) * (np.eye(p) - np.outer(K, phi_n)) @ P
		theta_hist[n, :] = theta
	
	y_hat    = Phi @ theta
	residuos = y_vec - y_hat
	return theta, y_hat, residuos, theta_hist

def analisis_residual(residuos, u, t, lags=20, label=''):
    N     = len(residuos)
    banda = 2 / np.sqrt(N)
    u_    = u[:N] - np.mean(u[:N])
    eps   = residuos - np.mean(residuos)
    
    # Autocorrelación
    ac = np.array([np.dot(eps[:N-k], eps[k:]) / np.dot(eps, eps) 
                   for k in range(lags+1)])
    
    # Correlación cruzada residuos-entrada
    cc = np.array([np.dot(eps[:N-k], u_[k:]) / (np.linalg.norm(eps) * np.linalg.norm(u_))
                   for k in range(lags+1)])
    
    lag_eje = np.arange(lags+1)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Residuos en el tiempo
    axes[0].plot(t[:N], residuos, color='steelblue', lw=1.0)
    axes[0].axhline(0, color='black', lw=0.8, ls='--')
    axes[0].set_ylabel(r'$\varepsilon[n]$')
    axes[0].set_title(f'Residuos en el tiempo {label}')
    axes[0].grid(True, alpha=0.3)
    
    # Autocorrelación
    axes[1].stem(lag_eje, ac, linefmt='steelblue', markerfmt='o', basefmt='k')
    axes[1].axhline( banda, color='red', ls='--', lw=0.8, label=r'$\pm 2/\sqrt{N}$')
    axes[1].axhline(-banda, color='red', ls='--', lw=0.8)
    axes[1].set_ylabel(r'$R_\varepsilon(\tau)$')
    axes[1].set_title('Autocorrelación de residuos')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Correlación cruzada
    axes[2].stem(lag_eje, cc, linefmt='darkorange', markerfmt='s', basefmt='k')
    axes[2].axhline( banda, color='red', ls='--', lw=0.8, label=r'$\pm 2/\sqrt{N}$')
    axes[2].axhline(-banda, color='red', ls='--', lw=0.8)
    axes[2].set_ylabel(r'$R_{\varepsilon u}(\tau)$')
    axes[2].set_xlabel(r'$\tau$')
    axes[2].set_title('Correlación cruzada residuos-entrada')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAnálisis residual {label}")
    print(f"  Max |Rε(τ≠0)|  = {np.max(np.abs(ac[1:])):.4f}  (banda: {banda:.4f})")
    print(f"  Max |Rεu(τ)|   = {np.max(np.abs(cc)):.4f}  (banda: {banda:.4f})")
    fuera_ac  = np.sum(np.abs(ac[1:]) > banda)
    fuera_cc  = np.sum(np.abs(cc)     > banda)
    print(f"  Lags fuera de banda Rε:  {fuera_ac}/{lags}")
    print(f"  Lags fuera de banda Rεu: {fuera_cc}/{lags+1}")

