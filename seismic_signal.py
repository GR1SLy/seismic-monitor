import numpy as np
from scipy.signal import butter, filtfilt, detrend


class SeismicSignal:
    def __init__(self, station_name, ch1, ch2, ch3, fs=1000.0):
        """
        Инициализация сигнала станции.
        :param fs: Частота дискретизации в Гц (по умолчанию 1000 Гц = 1 мс)
        """
        self.station_name = station_name
        self.fs = fs
        self.dt = 1.0 / fs

        # Переводим в numpy массивы для быстрых математических операций
        self.ch1 = np.array(ch1)
        self.ch2 = np.array(ch2)
        self.ch3 = np.array(ch3)

        # Длина сигнала
        self.n_samples = len(self.ch1)

        self.arrival_time = 0
        self.snr = 0
        self.peak_sta_lta = 0

    def preprocess(self, lowcut=1.0, highcut=25.0, order=4):
        """
        Предобработка сигнала: удаление тренда и полосовая фильтрация.
        Для взрывов часто берут диапазон от 1 до 20-40 Гц.
        """
        # 1. Удаление постоянной составляющей и линейного тренда (Mean/Trend removal)
        self.ch1 = detrend(self.ch1)
        self.ch2 = detrend(self.ch2)
        self.ch3 = detrend(self.ch3)

        # 2. Полосовая фильтрация (Butterworth bandpass)
        nyq = 0.5 * self.fs  # Частота Найквиста
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        # Применяем filtfilt для нулевого фазового сдвига (только при наличии данных НЕ в реальном времени)
        self.ch1 = filtfilt(b, a, self.ch1)
        self.ch2 = filtfilt(b, a, self.ch2)
        self.ch3 = filtfilt(b, a, self.ch3)

    def denoise_by_profile(self, noise_end_sec=2.0):
        """
        Подавляет шум, используя спектральный профиль начала записи.
        """
        # Определяем индекс конца шумового окна
        n_noise = int(noise_end_sec * self.fs)

        for attr in ['ch1', 'ch2', 'ch3']:
            data = getattr(self, attr)

            # Делаем БПФ всего сигнала и окна шума
            spec = np.fft.rfft(data)
            noise_spec = np.fft.rfft(data[:n_noise], n=len(data))  # дополняем нулями до длины данных

            # Считаем среднюю амплитуду шума
            noise_amplitude = np.abs(noise_spec)

            # Мягкое вычитание спектра (Spectral Subtraction)
            # Мы уменьшаем амплитуды частот, которые доминируют в шуме
            scale = 1.0 - (noise_amplitude / (np.abs(spec) + 1e-2))
            scale = np.clip(scale, 0.1, 1.0)  # Не даем упасть в ноль, чтобы не убить сигнал

            new_spec = spec * scale
            setattr(self, attr, np.fft.irfft(new_spec, n=len(data)))