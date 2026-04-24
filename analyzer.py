import numpy as np
import matplotlib.pyplot as plt


class SignalAnalyzer:
    def __init__(self, signals_dict):
        """
        :param signals_dict: словарь с объектами SeismicSignal {имя_станции: объект}
        """
        self.signals = signals_dict

    def print_statistics(self):
        """Вычисляет и выводит базовую статистику по всем станциям (до фильтрации)."""
        print("\n--- СТАТИСТИЧЕСКИЙ АНАЛИЗ СЫРЫХ ДАННЫХ ---")
        for st_name, signal in self.signals.items():
            # Берем первый (вертикальный) канал для примера
            ch_data = signal.ch1

            mean_val = np.mean(ch_data)
            min_val = np.min(ch_data)
            max_val = np.max(ch_data)
            p2p = max_val - min_val

            print(f"Станция {st_name} (Канал 1):")
            print(f"  Среднее (дрейф нуля): {mean_val:.6f}")
            print(f"  Размах (Peak-to-Peak): {p2p:.6f}")
            print(f"  Мин: {min_val:.6f} | Макс: {max_val:.6f}\n")

    def _compute_spectrum(self, data, dt, n_samples):
        """Внутренний метод: вычисляет амплитудный спектр Фурье (БПФ)."""
        # Убираем среднее перед Фурье, иначе на 0 Гц будет гигантский пик, затеняющий всё остальное
        data_centered = data - np.mean(data)

        freqs = np.fft.rfftfreq(n_samples, d=dt)
        fft_values = np.fft.rfft(data_centered)
        amplitudes = np.abs(fft_values) / n_samples

        return freqs, amplitudes

    def plot_spectrum(self, station_name, channel=1, max_freq=100):
        """
        Строит амплитудный спектр для конкретной станции и канала.
        :param max_freq: ограничение по оси X (по умолчанию 100 Гц)
        """
        if station_name not in self.signals:
            print(f"[ОШИБКА] Станция {station_name} не найдена.")
            return

        signal = self.signals[station_name]

        if channel == 1:
            data = signal.ch1
        elif channel == 2:
            data = signal.ch2
        else:
            data = signal.ch3

        freqs, amplitudes = self._compute_spectrum(data, signal.dt, signal.n_samples)

        plt.figure(figsize=(10, 5))
        plt.plot(freqs, amplitudes, color='purple', linewidth=1)
        plt.title(f'Амплитудный спектр (до фильтрации) - Станция: {station_name}, Канал {channel}')
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда')
        plt.xlim(0, max_freq)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

    def show_plots(self):
        """Отображает все созданные графики."""
        plt.show()