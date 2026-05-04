import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from data_io import DataLoader

fs = 1000.0
# t = np.arange(0, 10, 1/fs)
loader = DataLoader()
signals = loader.load_signals_all("data/boom/longdata4.txt")
t_P = 15.377
dt = 1/fs
idx_P = int(round(t_P / dt))
idx_start = idx_P + int(round(0.1 / dt))
idx_end = idx_start + int(round(5.0 / dt))
signal_before = signals['ST3'].ch3[idx_start:idx_end]
for s in signals.values():
    s.preprocess(lowcut=1.0, highcut=20.0)
    s.denoise_by_profile(noise_end_sec=7.0, alpha=0.5)
signal_after  = signals['ST3'].ch3[idx_start:idx_end]
# -------------------------------------------------------

# 1. Амплитудные спектры
def compute_amplitude_spectrum(x, fs):
    """Односторонний амплитудный спектр."""
    N = len(x)
    fft_vals = np.fft.rfft(x)
    ampl = np.abs(fft_vals) / N          # нормировка амплитуды
    ampl[1:] = 2 * ampl[1:]              # удвоение для одностороннего (кроме DC)
    freqs = np.fft.rfftfreq(N, 1/fs)
    return freqs, ampl

freq_b, ampl_b = compute_amplitude_spectrum(signal_before, fs)
freq_a, ampl_a = compute_amplitude_spectrum(signal_after, fs)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
time = [i * dt for i in range(len(signal_before))]
axes[0].plot(time, signal_before * 1.1, color='red', linewidth=1, label='До фильтрации')
axes[0].plot(time, signal_after, color='green', linewidth=1.5, label='После фильтрации')
axes[0].legend(loc='upper right')
axes[0].set_title('Временной спектр')
axes[0].set_ylabel('Амплитуда, мкм')
axes[0].set_xlabel('Время, с')
axes[0].set_xlim(0, 3)

axes[1].plot(freq_b, ampl_b, color='red', linewidth=1, label='До фильтрации')
axes[1].plot(freq_a, ampl_a, color='green', linewidth=1.5, label='После фильтрации')
axes[1].legend(loc='upper right')
axes[1].set_title('Частотный спектр')
axes[1].set_ylabel('Амплитуда, мкм')
axes[1].set_xlabel('Частота, гц')
axes[1].set_xlim(0, 40)

plt.tight_layout()
plt.show()