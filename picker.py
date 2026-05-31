from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time

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
        axes[0].legend(loc='upper right', fontsize='16')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 2. График функции STA/LTA
        axes[1].plot(time_axis, signal.sta_lta_curve, color='orange', linewidth=1.5, label='STA/LTA Отношение')
        # Линия порога
        axes[1].axhline(y=5.0, color='blue', linestyle=':', linewidth=3, label='Порог срабатывания')
        axes[1].axvline(x=signal.arrival_time, color='green', linestyle='--', linewidth=2)
        axes[1].axvline(x=signal.end_time, color='red', linestyle='--', linewidth=2)
        axes[1].set_ylabel('STA / LTA')
        axes[1].set_xlabel('Время (секунды)')
        axes[1].legend(loc='upper right', fontsize='16')
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

    def _stream_sta_lta(self, data, station, dt=0.001, sta_window=0.1, lta_window=2.0, history_sec=5.0, k=10, epsilon=1e-8):
        """
        Имитация потокового STA/LTA‑детектора с адаптивным порогом.

        Параметры:
            data        : одномерный numpy-массив отсчётов сигнала.
            dt          : период дискретизации в секундах (по умолчанию 0.001 с = 1 мс).
            sta_window  : длина короткого окна STA в секундах.
            lta_window  : длина длинного окна LTA в секундах.
            k           : множитель для порога (порог = медиана + k * MAD).
            epsilon     : малая константа для защиты от деления на ноль.
        """
        start = time.perf_counter()
        sta_lta_curve = []
        moments = []
        thresholds = {}
        n_sta = int(sta_window / dt)  # длина STA в отсчётах (100)
        n_lta = int(lta_window / dt)  # длина LTA в отсчётах (2000)
        n_history = int(history_sec / dt)
        # Кольцевые буферы для характеристической функции (cf)
        sta_cf = deque(maxlen=n_sta)  # хранит последние n_sta значений cf
        lta_cf = deque(maxlen=n_lta)  # хранит последние n_lta значений cf

        # Буфер для значений STA/LTA (длина n_lta)
        extended_buffer_len = n_lta + n_history
        sta_lta_buffer = deque(maxlen=extended_buffer_len)
        # sta_lta_buffer = deque(maxlen=n_lta)

        # Накопительные суммы для быстрого обновления средних
        sum_sta = 0.0
        sum_lta = 0.0

        # Флаг события, чтобы порог не насчитывал высокие амплитуды события
        event = False

        # Главный цикл – последовательная обработка каждого нового отсчёта
        for idx, sample in enumerate(data):
            t = idx * dt  # текущее время в секундах
            cf = sample ** 2  # характеристическая функция (энергия)

            # --- Обновление короткого окна STA ---
            oldest_sta = sta_cf[0] if len(sta_cf) == n_sta else 0.0
            sta_cf.append(cf)
            sum_sta += cf - oldest_sta
            sta = sum_sta / len(sta_cf)  # среднее в коротком окне

            # --- Обновление длинного окна LTA ---
            oldest_lta = lta_cf[0] if len(lta_cf) == n_lta else 0.0
            lta_cf.append(cf)
            sum_lta += cf - oldest_lta
            lta = sum_lta / len(lta_cf)  # среднее в длинном окне

            # Текущее значение STA/LTA с защитой от деления на ноль
            sta_lta_val = sta / (lta + epsilon)

            # Добавляем значение в кольцевой буфер STA/LTA
            if not event:
                sta_lta_buffer.append(sta_lta_val)
            sta_lta_curve.append(sta_lta_val)

            # Детекция возможна только после заполнения длинного буфера STA/LTA
            if len(sta_lta_buffer) < n_lta:
                continue  # недостаточно истории для надёжного порога

            # --- Адаптивный порог по содержимому STA/LTA буфера ---
            # Исключаем самое свежее значение (текущий отсчёт), чтобы событие
            # не завышало собственный порог.
            background = list(sta_lta_buffer)[:-1]  # n_lta - 1 элементов
            median_val = np.median(background)
            mad_val = np.median(np.abs(background - median_val))
            threshold = median_val + k * mad_val
            thresholds[t] = threshold

            # Проверка на превышение порога
            if sta_lta_val > threshold:
                moments.append(t)
                event = True
                print(f"Событие {station} на {t:.4f} с, порог: {threshold:.4f}, STA/LTA: {sta_lta_val:.4f}")
            else:
                event = False

        end = time.perf_counter()
        print(f"Time res: {end - start:.4f} s")

        time_x = [i * dt for i in range(len(data))]
        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)
        # axes[0] = (time_x, data, color='blue', linewidth=1)
        # axes[1] = (time_x, sta_lta_curve, color='yellow', linewidth=1)
        # plt.tight_layout()
        # plt.show()
        self.plotting(time_x, data, sta_lta_curve, moments, thresholds, station)

    def stream_sta_lta(self, signals, threshold=10, history_sec=5.0):
        for st_name, signal in signals.items():
            self._stream_sta_lta(signal.ch3, st_name, k=threshold, history_sec=history_sec)

    def plotting(self, time_axis, data, sta_lta_curve, moments, thresholds, station):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)
        fig.suptitle(f'Результат STA/LTA Пикинга - Станция {station}', fontsize=14)

        # 1. График самого сигнала
        axes[0].plot(time_axis, data, color='blue', linewidth=1)
        axes[0].set_ylabel('Амплитуда сигнала')
        axes[0].legend(loc='upper right', fontsize='16')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        for i in moments:
            axes[0].axvline(x=i, color='red', linestyle='--', linewidth=0.1)

        # 2. График функции STA/LTA
        axes[1].plot(time_axis, sta_lta_curve, color='orange', linewidth=1.5, label='STA/LTA Отношение')
        axes[1].set_ylabel('STA / LTA')
        axes[1].set_xlabel('Время (секунды)')
        axes[1].legend(loc='upper right', fontsize='16')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        # for i in moments:
        #     axes[1].axvline(x=i, color='red', linestyle='--', linewidth=0.1)
        xx = list(thresholds.keys())
        yy = list(thresholds.values())
        axes[1].plot(xx, yy, color='blue', linewidth=0.3)

        plt.tight_layout()
        plt.show()