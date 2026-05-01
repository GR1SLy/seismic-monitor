from unittest import signals

import numpy as np
from scipy.optimize import least_squares
from explosion import Explosion
from seismic_signal import SeismicSignal


class Calculator:
    def __init__(self):
        self.ml_median = 0.0
        self.ml_stations = []
        self.md_median = 0.0
        self.md_stations = []
        self.intensity_median = 0.0
        self.intensity_stations = []

    def get_fragment(self, signal, dt=0.001, start_delay=0.1, window_len=3.0):
        """
        Возвращает фрагмент данных после взрыва
        """
        if signal.arrival_time < 0:
            return [0], [0]
        t_P = signal.arrival_time
        idx_P = int(round(t_P / dt))
        idx_start = idx_P + int(round(start_delay / dt))
        idx_end = idx_start + int(round(window_len / dt))
        return signal.ch1[idx_start:idx_end], signal.ch2[idx_start:idx_end], signal.ch3[idx_start:idx_end]

    def calculate_displacement(self, ns_speed, ew_speed, ud_speed, dt=0.001, scale_to_um=1.0):
        """
            Интегрирует сигналы скорости (NS, EW) в смещение.

                Параметры:
            ----------
            ns_speed, ew_speed : array_like
                Фрагменты скорости (одинаковой длины) по каналам NS и EW.
            dt : float
                Шаг дискретизации (с).
            scale_to_um : float, optional
                Если задан, умножает полученное смещение на этот коэффициент.
                Например, если исходная скорость в м/с, а нужно смещение в микронах,
                то scale_to_um = 1e6. Если в мм/с - 1000.
                Если скорость уже в мкм/с, scale_to_um = 1.0.
                Если None, возвращает смещение в единицах скорость * секунда.
        """
        ns = np.asarray(ns_speed)
        ew = np.asarray(ew_speed)
        ud = np.asarray(ud_speed)

        # Интегрирование методом накопленной суммы (прямоугольники)
        # displacement = sum(v_i * dt) для каждого i
        ns = ns - np.mean(ns)
        ew = ew - np.mean(ew)
        ud = ud - np.mean(ud)
        disp_ns = np.cumsum(ns) * dt
        disp_ew = np.cumsum(ew) * dt
        disp_ud = np.cumsum(ud) * dt

        # Если нужно масштабирование в микроны
        if scale_to_um is not None:
            disp_ns *= scale_to_um
            disp_ew *= scale_to_um
            disp_ud *= scale_to_um

        vec_disp = np.sqrt(disp_ns ** 2 + disp_ew ** 2 + disp_ud ** 2)

        return vec_disp

    def calculate_max_displacement(self, signals):
        for signal in signals.values():
            ns, ew, ud = self.get_fragment(signal, window_len=3.0)
            vec_disp = self.calculate_displacement(ns, ew, ud)
            signal.a_max = np.max(vec_disp)

    def calculate_distances(self, signals, explosion):
        for signal in signals.values():
            dx = signal.x - explosion.x
            dy = signal.y - explosion.y
            signal.distance = np.hypot(dx, dy) / 1000.0

    def calculate_distance(self, signal, explosion, in_kilometers=True):
        """
        Вычисляет расстояние от станции до взрыва

        Возвращает:
        ----------
        distance : float
            Расстояние от станции до эпицентра (в км, если in_kilometers=True, иначе в м).
        """
        dx = signal.x - explosion.x
        dy = signal.y - explosion.y
        dist_m = np.hypot(dx, dy)  # евклидово расстояние в метрах

        if in_kilometers:
            return dist_m / 1000.0
        else:
            return dist_m

    def __locate_explosion_3(self, signals, speed=SeismicSignal.speed):
        """
        Определяет координаты взрыва (x0, y0) и время t0 по временам прихода сигнала
        на сейсмических станциях.
        Возвращает
        ----------
        explosion : Explosion
            Объект с атрибутами x, y, t0, speed, rms.
        """
        print("Поиск взрыва...")
        # Преобразуем signals в список для удобства
        stations = list(signals.values())
        if len(stations) < 3:
            raise ValueError(f"Недостаточно станций: {len(stations)}. Нужно минимум 3.")

        # Извлекаем массивы
        xs = np.array([s.x for s in stations])
        ys = np.array([s.y for s in stations])
        times = np.array([s.arrival_time for s in stations])

        # Функция невязки: для каждого станции разница между наблюдённым и предсказанным временем
        def residuals(params):
            x0, y0, t0 = params
            dists = np.hypot(xs - x0, ys - y0)
            t_pred = t0 + dists / speed
            return times - t_pred

        # Начальное приближение: центр тяжести станций, t0 = минимальное время - 0.5 с
        x0_init = np.mean(xs)
        y0_init = np.mean(ys)
        t0_init = np.min(times) - 0.5
        params_init = [x0_init, y0_init, t0_init]

        # Оптимизация методом наименьших квадратов
        result = least_squares(residuals, params_init, method='trf')
        x0_opt, y0_opt, t0_opt = result.x

        # Вычисляем RMS
        final_res = residuals(result.x)
        rms = np.sqrt(np.mean(final_res ** 2))

        explosion = Explosion(x0_opt, y0_opt, t0_opt, SeismicSignal.speed, rms)
        print(f"Взрыв локализован в {x0_opt:.1f}, {y0_opt:.1f}!")
        return explosion

    def __locate_explosion_4(self, signals):
        """
        Определяет координаты взрыва (x0, y0) и время t0 по временам прихода
        P-волны на станции при известной средней скорости v (случай В).
        Возвращает
        ----------
        explosion : Explosion
            Объект с атрибутами x, y, t0, speed, rms.
        """
        stations = list(signals.values())
        if len(stations) < 4:
            raise ValueError(
                f"Недостаточно станций: {len(stations)}. Требуется минимум 4."
            )

        # Известная скорость
        v = SeismicSignal.speed

        xs = np.array([s.x for s in stations])
        ys = np.array([s.y for s in stations])
        times = np.array([s.arrival_time for s in stations])

        def residuals(params):
            x0, y0, t0 = params
            dists = np.hypot(xs - x0, ys - y0)
            t_pred = t0 + dists / v
            return times - t_pred

        # Начальные приближения
        x0_init = np.mean(xs)
        y0_init = np.mean(ys)
        t0_init = np.min(times) - 0.5
        params_init = [x0_init, y0_init, t0_init]

        # Нелинейный МНК (алгоритм доверительной области)
        result = least_squares(residuals, params_init, method='trf')
        x0_opt, y0_opt, t0_opt = result.x

        final_res = residuals(result.x)
        rms = np.sqrt(np.mean(final_res ** 2))

        explosion = Explosion(x0_opt, y0_opt, t0_opt, v, rms)
        print(f"Взрыв локализован в {x0_opt:.1f}, {y0_opt:.1f}!")
        return explosion


    def __locate_explosion_5(self, signals, initial_speed=1.4):
        """
        Определяет координаты (x0, y0), время взрыва t0 и среднюю скорость v
        по временам прихода P-волны на станции (случай Г – неизвестны ни t0, ни v).
        Возвращает
        ----------
        explosion : Explosion
            Объект с атрибутами x, y, t0, speed, rms.
        """
        stations = list(signals.values())
        if len(stations) < 5:
            raise ValueError(
                f"Недостаточно станций: {len(stations)}. Требуется минимум 5."
            )

        xs = np.array([s.x for s in stations])
        ys = np.array([s.y for s in stations])
        times = np.array([s.arrival_time for s in stations])

        def residuals(params):
            x0, y0, t0, v = params
            dists = np.hypot(xs - x0, ys - y0)
            t_pred = t0 + dists / v
            return times - t_pred

        # Начальные приближения
        x0_init = np.mean(xs)
        y0_init = np.mean(ys)
        # t0 берём чуть раньше самого раннего вступления
        t0_init = np.min(times) - 0.5
        params_init = [x0_init, y0_init, t0_init, initial_speed]

        # Оптимизация (доверительная область)
        result = least_squares(residuals, params_init, method='trf')
        x0_opt, y0_opt, t0_opt, speed_opt = result.x

        final_res = residuals(result.x)
        rms = np.sqrt(np.mean(final_res ** 2))

        explosion = Explosion(x0_opt, y0_opt, t0_opt, speed_opt, rms)
        print(f"Взрыв локализован в {x0_opt:.1f}, {y0_opt:.1f}!")
        return explosion

    def locate_explosion(self, signals):
        match len(signals):
            case 3:
                return self.__locate_explosion_3(signals)
            case 4:
                return self.__locate_explosion_4(signals)
            case _:
                return self.__locate_explosion_5(signals)

    def calculate_local_magnitude(self, signals, explosion):
        """
        Вычисляет магнитуду

         Возвращает:
        ----------
        ml_median : float
            Медианная магнитуда по всем станциям.
        station_results : list of dict
            Список с результатами для каждой станции (станция, расстояние, амплитуда, период, магнитуда).
        """
        print("Вычисление локальной магнитуды...")
        ml_values = []
        self.ml_stations = []

        for signal in signals.values():
            if signal.a_max <= 0:
                continue
            distance = self.calculate_distance(signal, explosion)

            ml = np.log10(signal.a_max * 1e3) + 1.11 * np.log10(signal.distance) + 0.00189 * signal.distance - 2.09

            ml_values.append(ml)
            signal.ml = ml

            print(f"Магнитуда для {signal.station_name} найдена: {ml:.3f}!")
            self.ml_stations.append({
                'station_name': signal.station_name,
                'amplitude': signal.a_max,
                'magnitude': ml
            })

        if not ml_values:
            return np.nan, []
        self.ml_median = np.median(ml_values)
        return self.ml_median, self.ml_stations

    def calculate_code_magnitude(self, signals, explosion, coef_a=0.65, coef_b=-0.06, coef_c=3.35):
        md_values = []

        for signal in signals.values():
            distance = self.calculate_distance(signal, explosion)
            md = coef_a * np.log10(signal.duration) + coef_b * distance + coef_c
            md_values.append(md)

            print(f"Магнитуда для {signal.station_name} найдена: {md:.3f}!")
            self.md_stations.append({
                'station_name': signal.station_name,
                'magnitude': md
            })

        if not md_values:
            return np.nan, []
        self.md_median = np.median(md_values)
        print(md_values)
        return self.md_median, self.md_stations

    def calculate_intensity(self, signals):
        intensity_values = []
        self.intensity_stations = []

        print(f"Вычисляется интенсивность...")
        for signal in signals.values():
            if signal.a_max <= 0:
                continue

            intensity = 1.47 * np.log10(signal.a_max * 10e-4) + 6.26 - 0.7
            intensity_values.append(intensity)

            print(f"Интенсивность для {signal.station_name} найдена: {intensity:.3f}!")
            self.intensity_stations.append({
                'station_name': signal.station_name,
                'intensity': intensity
            })
        if not intensity_values:
            return np.nan, []
        self.intensity_median = np.median(intensity_values)
        return self.intensity_median, self.intensity_stations
