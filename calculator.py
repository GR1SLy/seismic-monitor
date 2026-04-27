import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from explosion import Explosion
from seismic_signal import SeismicSignal


class Calculator:
    def __init__(self):
        self.ml_median = 0.0
        self.ml_stations = []
        self.md_median = 0.0
        self.md_stations = []
        self.explosion = None

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
        return signal.ch1[idx_start:idx_end], signal.ch2[idx_start:idx_end]

    def calculate_displacement(self, ns_speed, ew_speed, dt=0.001, scale_to_um=1.0):
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

            Возвращает:
            ----------
            disp_ns : np.ndarray
                Смещение по каналу NS (в тех же единицах, что и скорость, умноженных на dt,
                или масштабированное).
            disp_ew : np.ndarray
                Смещение по каналу EW.
            horiz_disp : np.ndarray
                Горизонтальное смещение = sqrt(disp_ns**2 + disp_ew**2).
        """
        ns = np.asarray(ns_speed)
        ew = np.asarray(ew_speed)
        if len(ns) != len(ew):
            raise ValueError("Массивы NS и EW должны быть одинаковой длины")

        # Интегрирование методом накопленной суммы (прямоугольники)
        # displacement = sum(v_i * dt) для каждого i
        ns = ns - np.mean(ns)
        ew = ew - np.mean(ew)
        disp_ns = np.cumsum(ns) * dt
        disp_ew = np.cumsum(ew) * dt

        # Если нужно масштабирование в микроны
        if scale_to_um is not None:
            disp_ns *= scale_to_um
            disp_ew *= scale_to_um

        horiz_disp = np.sqrt(disp_ns ** 2 + disp_ew ** 2)

        return disp_ns, disp_ew, horiz_disp

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

    def locate_explosion(self, signals, speed=SeismicSignal.speed):
        """
        Определяет координаты взрыва (x0, y0) и время t0 по временам прихода сигнала
        на сейсмических станциях.

        Возвращает
        ----------
        dict
            {
                'x': float,        # координата эпицентра X (м)
                'y': float,        # координата эпицентра Y (м)
                't0': float,       # время взрыва (с)
                'rms': float,      # среднеквадратичная невязка (с)
                'success': bool    # успешна ли оптимизация
            }
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

        self.explosion = Explosion(x0_opt, y0_opt, t0_opt, rms)
        print(f"Взрыв локализован в {x0_opt:.1f}, {y0_opt:.1f}!")
        return self.explosion

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
        print("Вычисление магнитуды...")
        ml_values = []
        self.ml_stations = []

        for signal in signals.values():
            ns, ew = self.get_fragment(signal, window_len=signal.duration-0.5)
            disp_ns, disp_ew, horiz_disp = self.calculate_displacement(ns, ew)
            a_max = np.max(horiz_disp)
            if a_max <= 0:
                continue
            a_max *= 10e3
            distance = self.calculate_distance(signal, explosion)

            ml = np.log10(a_max) + 1.11 * np.log10(distance) + 0.00189 * distance - 2.09

            ml_values.append(ml)

            print(f"Магнитуда для {signal.station_name} найдена: {ml:.3f}!")
            self.ml_stations.append({
                'station_name': signal.station_name,
                'amplitude': a_max,
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
