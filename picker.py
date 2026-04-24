import numpy as np
import matplotlib.pyplot as plt


class PhasePicker:
    def __init__(self, signals_dict):
        """
        :param signals_dict: словарь с отфильтрованными объектами SeismicSignal
        """
        self.signals = signals_dict

    def _compute_sta_lta(self, data, n_sta, n_lta):
        """
        Внутренний метод: вычисляет вектор функции STA/LTA для сигнала.
        Используется метод быстрого скользящего среднего через кумулятивную сумму.
        """
        # Характеристическая функция: берем квадрат сигнала (энергию)
        cf = data ** 2

        sta = np.zeros(len(cf))
        lta = np.zeros(len(cf))

        # Быстрая кумулятивная сумма
        csum = np.cumsum(cf)

        # Считаем STA (короткое окно)
        sta[n_sta:] = (csum[n_sta:] - csum[:-n_sta]) / n_sta
        sta[:n_sta] = csum[:n_sta] / np.arange(1, n_sta + 1)

        # Считаем LTA (длинное окно)
        lta[n_lta:] = (csum[n_lta:] - csum[:-n_lta]) / n_lta
        lta[:n_lta] = csum[:n_lta] / np.arange(1, n_lta + 1)

        # Защита от деления на ноль
        epsilon = np.percentile(cf, 10) + 1e-10
        sta_lta = sta / (lta + epsilon)

        # Обнуляем самое начало графика, пока LTA окно еще не заполнилось (чтобы избежать ложных скачков)
        sta_lta[:n_lta] = 0

        return sta_lta

    def pick_arrivals(self, sta_sec=0.1, lta_sec=2.0, threshold=10.0):
        """
        Пробегает по всем станциям и ищет время вступления P-волны (на 1-м канале).

        :param sta_sec: длина короткого окна (сек). Обычно 0.1 - 0.2
        :param lta_sec: длина длинного окна (сек). Обычно 1.0 - 5.0
        :param threshold: порог срабатывания. Обычно 3.0 - 5.0
        """
        print("\n--- ЗАПУСК АВТОМАТИЧЕСКОГО ПИКИНГА (STA/LTA) ---")

        for st_name, signal in self.signals.items():
            # Переводим секунды в количество отсчетов
            n_sta = int(sta_sec * signal.fs)
            n_lta = int(lta_sec * signal.fs)

            # Считаем STA/LTA (обычно P-волна лучше всего видна на вертикальном Канале 1)
            sta_lta_curve = self._compute_sta_lta(signal.ch3, n_sta, n_lta)

            # Сохраняем кривую в объект, чтобы потом нарисовать
            signal.sta_lta_curve = sta_lta_curve

            # АВТО-ПОРОГ: используем медиану и MAD для оценки фона
            # Берем первые 2/3 сигнала или весь сигнал для оценки статистики
            median_val = np.median(sta_lta_curve[n_lta:])
            mad_val = np.median(np.abs(sta_lta_curve[n_lta:] - median_val))

            # Порог = Медиана + K * MAD
            auto_threshold = median_val + threshold * mad_val
            signal.used_threshold = auto_threshold  # сохраним для графика

            # Ищем ИНДЕКС, где STA/LTA впервые превысило порог
            trigger_indices = np.where(sta_lta_curve > auto_threshold)[0]

            if len(trigger_indices) > 0:
                first_trigger_idx = trigger_indices[0]
                arrival_time = first_trigger_idx * signal.dt
                # Сохраняем найденное время прямо в объект сигнала!
                signal.arrival_time = arrival_time

                # После того как нашли signal.arrival_time:
                if signal.arrival_time:
                    idx = int(signal.arrival_time * signal.fs)

                    # Считаем SNR:
                    # Амплитуда сигнала (берем окно 0.5с после вступления)
                    signal_window = signal.ch3[idx: idx + int(0.5 * signal.fs)]
                    # Амплитуда шума (берем окно 1с до вступления)
                    noise_window = signal.ch3[idx - int(1.0 * signal.fs): idx]

                    if len(noise_window) > 0 and len(signal_window) > 0:
                        snr = np.max(np.abs(signal_window)) / (np.std(noise_window) + 1e-10)
                        signal.snr = snr

                        status = "OK" if snr > 5.0 else "WEAK"
                        print(f"Станция {st_name}: Время {signal.arrival_time:.3f} | SNR: {snr:.3f} [{status}]")
                    else:
                        signal.snr = 0
                else:
                    signal.snr = 0

                print(f"Станция {st_name}: Взрыв обнаружен на {arrival_time:.3f} сек (Пик STA/LTA: {np.max(sta_lta_curve):.1f})")
                signal.peak_sta_lta = float(np.max(sta_lta_curve))
            else:
                signal.arrival_time = -1
                print(f"[WARN] Станция {st_name}: Взрыв не обнаружен (Порог {threshold} не пробит).")

    def plot_picking(self, station_name):
        """
        Рисует сигнал и график STA/LTA с линией срабатывания.
        """
        if station_name not in self.signals:
            return

        signal = self.signals[station_name]
        if not hasattr(signal, 'sta_lta_curve') or signal.arrival_time is None:
            print(f"Для станции {station_name} нет данных пикинга.")
            return

        time_axis = [i * signal.dt for i in range(signal.n_samples)]

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)
        fig.suptitle(f'Результат STA/LTA Пикинга - Станция {station_name}', fontsize=14)

        # 1. График самого сигнала (Канал 1)
        axes[0].plot(time_axis, signal.ch3, color='blue', linewidth=1)
        axes[0].axvline(x=signal.arrival_time, color='red', linestyle='--', linewidth=2,
                        label=f'Вступление: {signal.arrival_time:.3f} с')
        axes[0].set_ylabel('Амплитуда сигнала')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. График функции STA/LTA
        axes[1].plot(time_axis, signal.sta_lta_curve, color='orange', linewidth=1.5, label='STA/LTA Отношение')
        # Линия порога
        axes[1].axhline(y=5.0, color='gray', linestyle=':', linewidth=2, label='Порог срабатывания')
        axes[1].axvline(x=signal.arrival_time, color='red', linestyle='--', linewidth=2)
        axes[1].set_ylabel('STA / LTA')
        axes[1].set_xlabel('Время (секунды)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()