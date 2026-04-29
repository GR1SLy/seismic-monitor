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

    def pick_event_end(self, noise_win_sec=1.0, noise_factor=3.0,
                       coda_factor=0.05, hold_sec=0.5, smooth_win_sec=0.2,
                       max_dur_sec=15.0):
        """
        Адаптивный поиск окончания события.
        Конец фиксируется, когда сглаженная огибающая падает ниже порога
        и остаётся под ним hold_sec секунд.
        Порог = максимум из (noise_factor * std шума) и (coda_factor * пик огибающей).
        """
        print("\n--- ПОИСК КОНЦА СОБЫТИЯ (ADAPTIVE THRESHOLD) ---")

        for st_name, signal in self.signals.items():
            if signal.arrival_time is None or signal.arrival_time < 0:
                signal.end_time = -1
                signal.duration = -1
                continue

            fs = signal.fs
            dt = signal.dt
            data = signal.ch3
            idx_arr = int(signal.arrival_time * fs)

            # 1. Оценка шума перед вступлением
            n_noise = max(0, idx_arr - int(noise_win_sec * fs))
            noise_std = np.std(data[n_noise:idx_arr])
            noise_thresh = noise_factor * noise_std

            # 2. Сглаженная огибающая (энергия)
            win = max(1, int(smooth_win_sec * fs))
            envelope = np.sqrt(np.convolve((data**2), np.ones(win) / win, mode='same'))
            signal.envelope = envelope  # сохраним для графики

            # 3. Окно поиска пика: [вступление, вступление + max_dur_sec]
            idx_end_win = min(len(data), idx_arr + int(max_dur_sec * fs))
            envelope_win = envelope[idx_arr:idx_end_win]
            if len(envelope_win) == 0:
                signal.end_time = signal.arrival_time
                signal.duration = 0.0
                continue

            peak_val = np.max(envelope_win)
            coda_thresh = coda_factor * peak_val

            # 4. Итоговый адаптивный порог
            threshold = max(noise_thresh, coda_thresh)
            signal.used_end_threshold = threshold  # для отладки/визуализации

            # 5. Поиск устойчивого перехода ниже порога после пика
            hold_samples = int(hold_sec * fs)
            idx_peak = idx_arr + np.argmax(envelope_win)
            end_idx = idx_end_win  # по умолчанию конец окна
            found = False

            i = idx_peak
            while i < idx_end_win - hold_samples:
                if envelope[i] < threshold:
                    if np.all(envelope[i:i + hold_samples] < threshold):
                        end_idx = i
                        found = True
                        break
                    else:
                        # пропускаем ложное проседание
                        viol = np.where(envelope[i:i + hold_samples] >= threshold)[0]
                        i += viol[-1] + 1 if len(viol) > 0 else hold_samples
                else:
                    i += 1

            if found:
                signal.end_time = end_idx * dt
            else:
                # спада не нашли – берём границу поискового окна
                signal.end_time = idx_end_win * dt

            signal.duration = signal.end_time - signal.arrival_time

            print(f"Станция {st_name}: пик={peak_val:.3e}, порог={threshold:.3e} "
                  f"(шум*{noise_factor}={noise_thresh:.3e}, доля пика={coda_thresh:.3e})")
            print(f"  Конец: {signal.end_time:.3f} с, длительность: {signal.duration:.3f} с")

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
        axes[0].axvline(x=signal.arrival_time, color='green', linestyle='--', linewidth=2,
                        label=f'Вступление: {signal.arrival_time:.3f} с')
        axes[0].axvline(x=signal.end_time, color='red', linestyle='--', linewidth=2,
                        label=f'Затухание: {signal.end_time:.3f} с')
        axes[0].set_ylabel('Амплитуда сигнала')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. График функции STA/LTA
        axes[1].plot(time_axis, signal.sta_lta_curve, color='orange', linewidth=1.5, label='STA/LTA Отношение')
        # Линия порога
        axes[1].axhline(y=5.0, color='gray', linestyle=':', linewidth=2, label='Порог срабатывания')
        axes[1].axvline(x=signal.arrival_time, color='green', linestyle='--', linewidth=2)
        axes[1].axvline(x=signal.end_time, color='red', linestyle='--', linewidth=2)
        axes[1].set_ylabel('STA / LTA')
        axes[1].set_xlabel('Время (секунды)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

    def plot_picking_all(self):
        """
        Рисует сигнал и график STA/LTA для всех станций в одном окне.
        Каждая станция представлена парой subplot: сигнал (канал 3) и кривая STA/LTA.
        """
        # Отбираем станции, для которых есть необходимые данные пикинга
        valid_stations = [
            (name, sig) for name, sig in self.signals.items()
            if hasattr(sig, 'sta_lta_curve') and sig.arrival_time is not None
        ]

        if not valid_stations:
            print("Нет данных пикинга ни для одной станции.")
            return

        n = len(valid_stations)
        fig, axes = plt.subplots(nrows=2 * n, ncols=1, figsize=(12, 3 * n), sharex=True)

        # Если станция только одна, axes — это одномерный массив из двух осей
        if n == 1:
            axes = [axes[0], axes[1]]

        for i, (name, signal) in enumerate(valid_stations):
            time_axis = [j * signal.dt for j in range(signal.n_samples)]

            # Верхний график — сигнал (канал 3)
            ax_sig = axes[2 * i]
            ax_sig.plot(time_axis, signal.ch3, color='blue', linewidth=1)
            ax_sig.axvline(x=signal.arrival_time, color='green', linestyle='--',
                           linewidth=2, label=f'Вступление: {signal.arrival_time:.3f} с')
            if signal.end_time is not None:
                ax_sig.axvline(x=signal.end_time, color='red', linestyle='--',
                               linewidth=2, label=f'Затухание: {signal.end_time:.3f} с')
            ax_sig.set_ylabel(f'{name}\nАмплитуда', fontsize=9)
            ax_sig.legend(loc='upper left', fontsize=8)
            ax_sig.grid(True, linestyle='--', alpha=0.6)

            # Нижний график — отношение STA/LTA
            ax_sta = axes[2 * i + 1]
            ax_sta.plot(time_axis, signal.sta_lta_curve, color='orange', linewidth=1.5,
                        label='STA/LTA Отношение')
            ax_sta.axvline(x=signal.arrival_time, color='green', linestyle='--', linewidth=2)
            if signal.end_time is not None:
                ax_sta.axvline(x=signal.end_time, color='red', linestyle='--', linewidth=2)
            ax_sta.grid(True, linestyle='--', alpha=0.6)

        # Общая подпись оси X для нижнего графика
        axes[-1].set_xlabel('Время (секунды)')
        fig.suptitle('Результат STA/LTA Пикинга — Все станции', fontsize=14, y=1.01)
        mgn = fig.canvas.manager
        mgn.resize(3000, 260*n)
        plt.tight_layout(h_pad=0.05)