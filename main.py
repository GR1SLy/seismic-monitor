import matplotlib.pyplot as plt
from calculator import Calculator
from data_io import DataLoader
from picker import PhasePicker
import numpy as np
from pathlib import Path

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

def load_old(filename):
    print("Загрузка данных...")
    loader = DataLoader(filename, fs=1000.0)
    signals = loader.load_signals()
    print(f"Успешно загружены данные для {len(signals)} станций: {list(signals.keys())}")
    return signals

def preprocess(signals):
    print("Запуск предобработки (Detrend + Bandpass filter 1-30 Гц)...")
    for st_name, signal in signals.items():
        signal.preprocess(lowcut=1.0, highcut=30.0)
        signal.denoise_by_profile(noise_end_sec=5.0)
    print("Данные готовы!")

def pick_signals(signals, graphics):
    picker = PhasePicker(signals)
    print("Начинается пикирование данных...")
    picker.pick_arrivals(threshold=15)
    picker.pick_event_end(coda_factor=0.08)
    print("Взрывы найдены!")
    if graphics == 1:
        for st_name in signals:
            picker.plot_picking(st_name)
        plt.show()
    elif graphics == 2:
        picker.plot_picking_all()
        plt.show()

def old():

    signals = load_old("data/longdata1.txt")

    set_coor(signals)

    preprocess(signals)

    check = int(input("Для вывода отдельных графиков введите 1\nДля просмотра всех графиков введите 2:\n"))
    pick_signals(signals, check)

    calc = Calculator()

    ok_signals = check_arrivals(signals)
    explosion = calc.locate_explosion(ok_signals)

    calc.calculate_max_displacement(ok_signals)

    calc.calculate_local_magnitude(ok_signals, explosion)
    ml = calc.ml_median
    print(f"Median ML: {ml:.3f}")

    calc.calculate_intensity(ok_signals)
    intensity = calc.intensity_median
    print(f"Median intensity: {intensity:.3f}")

    # calc.calculate_code_magnitude(ok_signals, explosion)
    # md = calc.md_median
    # print(f"Median MD: {md:.3f}")

def new():
    folder = Path("data/dirname")

    for file in folder.iterdir():
        if file.is_file():
            filename = file.name
            full_path = str(file)


if __name__ == '__main__':
    old()