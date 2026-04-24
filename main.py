import matplotlib.pyplot as plt
import numpy as np
from data_io import DataLoader
from picker import PhasePicker
from seismic_signal import SeismicSignal

import numpy as np
from scipy.optimize import least_squares



def locate_explosion(signals, speed=SeismicSignal.speed):
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
    rms = np.sqrt(np.mean(final_res**2))

    return {
        'x': x0_opt,
        'y': y0_opt,
        't0': t0_opt,
        'rms': rms,
        'success': result.success
    }

def check_arrivals(signals):
    ok_signals = {}
    for signal in signals.values():
        if signal.snr > 5:
            ok_signals[signal.station_name] = signal
    return find_outliers_mad(ok_signals)

def find_outliers_mad(signals, threshold=3.5):
    """
    Функция для нахождения выбросов при нормальном SNR
    """
    signal_list = list(signals.values())

    arrival_times = np.array([s.arrival_time for s in signal_list], dtype=float)

    # --- Робастное обнаружение выбросов (MAD) ---
    median = np.median(arrival_times)
    abs_dev = np.abs(arrival_times - median)
    mad = np.median(abs_dev)

    if mad == 0:
        return {s.station_name: s for s in signal_list}

    # Модифицированный Z-критерий
    modified_z = 0.6745 * (arrival_times - median) / mad

    # Оставляем только те, у которых |Z| <= threshold (не выбросы)
    inlier_mask = np.abs(modified_z) <= threshold

    # Формируем результат
    res = {}
    for i, is_ok in enumerate(inlier_mask):
        if is_ok:
            sig = signal_list[i]
            res[sig.station_name] = sig

    return res

def set_coor(signals):
    for signal in signals.values():
        match signal.station_name:
            case 'ST1':
                signal.x = 558599.12
                signal.y = 5941987.09
            case 'ST2':
                signal.x = 556918.94
                signal.y = 5941443.52
            case 'ST3':
                signal.x = 559903.40
                signal.y = 5945530.70
            case 'ST4':
                signal.x = 560192.49
                signal.y = 5947214.54
            case 'ST5':
                signal.x = 559491.55
                signal.y = 5944646.45
            case 'ST6':
                signal.x = 557619.37
                signal.y = 5940428.39
            case 'ST7':
                signal.x = 562683.10
                signal.y = 5948237.41

def visualize_seismic(signal, name):
    """
    Функция для отрисовки всех трех каналов одной сейсмической станции.
    """
    # Создаем ось времени: умножаем индексы массива на шаг дискретизации (dt)
    time_axis = [i * signal.dt for i in range(signal.n_samples)]

    # Создаем окно с тремя графиками (3 строки, 1 колонка), ось X (время) - общая
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6), sharex=True)
    fig.suptitle(f'Сейсмограмма станции: {signal.station_name} ({name})', fontsize=14)

    # Отрисовка Канала 1 (обычно Z - вертикальный)
    axes[0].plot(time_axis, signal.ch1, color='blue', linewidth=0.7)
    axes[0].set_ylabel('Канал 1')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Отрисовка Канала 2 (обычно N-S - север-юг)
    axes[1].plot(time_axis, signal.ch2, color='green', linewidth=0.7)
    axes[1].set_ylabel('Канал 2')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Отрисовка Канала 3 (обычно E-W - запад-восток)
    axes[2].plot(time_axis, signal.ch3, color='red', linewidth=0.7)
    axes[2].set_ylabel('Канал 3')
    axes[2].set_xlabel('Время (секунды)')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    # Компактное расположение графиков
    plt.tight_layout()


def main():
    # 1. Указываем путь к файлу
    data_file = 'data/longdata1.txt'

    # 2. Инициализируем загрузчик и читаем данные
    print("Загрузка данных...")
    loader = DataLoader(data_file, fs=1000.0)
    signals = loader.load_signals()

    set_coor(signals)

    if not signals:
        print("Ошибка: Сигналы не загружены. Проверьте данные.")
        return

    print(f"Успешно загружены данные для {len(signals)} станций: {list(signals.keys())}")

    # for st_name, signal in signals.items():
    #     if st_name == 'ST1':
    #         visualize_seismic(signal, 'До фильтрации')

    # 3. Предобработка сигнала (Detrend + Bandpass filter)
    print("Запуск предобработки (Detrend + Bandpass filter 1-30 Гц)...")
    for st_name, signal in signals.items():
        signal.preprocess(lowcut=1.0, highcut=25.0)
        signal.denoise_by_profile(noise_end_sec=5.0)


    print("Данные готовы!")

    picker = PhasePicker(signals)
    picker.pick_arrivals(sta_sec=0.1, lta_sec=2.0, threshold=15)

    # picker.plot_picking('ST1')
    # picker.plot_picking('ST2')
    # picker.plot_picking('ST3')
    # picker.plot_picking('ST5')
    # picker.plot_picking('ST6')
    # picker.plot_picking('ST7')


    # 4. Вызов функции визуализации для каждой станции
    # for st_name, signal in signals.items():
    #     if st_name == 'ST1':
    #         visualize_seismic(signal, 'После фильтрации')

    plt.show()

    # for signal in signals.values():
    #     print(signal.get_info())
    #
    # print("================GOOD SIGNALS================")
    #
    # for signal in ok_signal.values():
    #     print(signal.get_info())

    # ok_signals = check_arrivals(signals)
    # result = locate_explosion(ok_signals)
    # print(f"Эпицентр: X={result['x']:.1f} м, Y={result['y']:.1f} м")
    # print(f"Время взрыва t0 = {result['t0']:.3f} с")
    # print(f"RMS = {result['rms']:.4f} с")

if __name__ == '__main__':
    main()