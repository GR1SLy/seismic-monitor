import matplotlib.pyplot as plt
from calculator import Calculator
from data_io import DataLoader
from picker import PhasePicker
import numpy as np
from pathlib import Path
import re
import os

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

def load(filename, type="ALL"):
    print("Загрузка данных...")
    loader = DataLoader(fs=1000.0)
    if type == "ALL":
        signals = loader.load_signals_all(filename)
    elif type == "PER_STATION":
        signals = loader.load_signals_per_station(filename)
    else:
        raise TypeError("Load type must be ALL or PER_STATION")
    print(f"Успешно загружены данные для {len(signals)} станций: {list(signals.keys())}")
    return signals

def preprocess(signals):
    print("Запуск предобработки (Detrend + Bandpass filter 1-30 Гц + denoise by profile)...")
    for st_name, signal in signals.items():
        signal.preprocess(lowcut=1.0, highcut=25.0)
        signal.denoise_by_profile(noise_end_sec=5.0, alpha=1.5)
    print("Данные готовы!")

def pick_signals(signals, graphics):
    picker = PhasePicker(signals)
    print("Начинается пикирование данных...")
    picker.pick_arrivals(threshold=20)
    picker.pick_event_end(coda_factor=0.08)
    print("Взрывы найдены!")
    if graphics == 1:
        for st_name in signals:
            picker.plot_picking(st_name)
        plt.show()
    elif graphics == 2:
        picker.plot_picking_all()
        plt.show()

def all_stations_in():

    # signals = load("data/boom1/data2.txt", type="ALL")
    signals = load("data/boom/longdata2.txt", type="ALL")

    set_coor(signals)

    preprocess(signals)

    check = int(input("Для вывода отдельных графиков введите 1\nДля просмотра всех графиков введите 2:\n"))
    pick_signals(signals, check)

    calc = Calculator()

    ok_signals = check_arrivals(signals)
    explosion = calc.locate_explosion(ok_signals)
    print(str(explosion))
    calc.calculate_max_displacement(ok_signals)

    calc.calculate_distances(signals, explosion)
    for signal in ok_signals.values():
        print(f"{signal.station_name} a_max = {signal.a_max}; distance = {calc.calculate_distance(signal, explosion)}; distance2 = {signal.distance}")
    calc.calculate_local_magnitude(ok_signals, explosion)
    ml = calc.ml_median
    print(f"Median ML: {ml:.3f}")

    calc.calculate_intensity(ok_signals)
    intensity = calc.intensity_median
    print(f"Median intensity: {intensity:.3f}")

def per_station():
    folder = Path("data/new/ST7")

    for file in folder.iterdir():
        if file.is_file():
            filename = file.name
            pattern = r'^ST(\d+) \(mode_normal\) (.+)\.txt$'
            match = re.match(pattern, filename)
            if not match:
                raise ValueError(f"Не удалось разобрать имя файла: {filename}")
            datetime_part = match.group(2)
            count = 7
            st_names = ['ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7']
            for st_num in range(1, 8):
                st_name = f"ST{st_num}"
                file_name = f"{st_name} (mode_normal) {datetime_part}.txt"
                fullpath = f"data/new/ST{st_num}/{file_name}"
                if not os.path.isfile(fullpath):
                    # print(f"[WARN] Файл {fullpath} не найден. Станция {st_name} будет пропущена.")
                    count -= 1
                    st_names.remove(st_name)
                    continue
            print(filename, count, st_names)

def test():
    folder = Path("data/boom/")
    result = []
    for inner_folder in folder.iterdir():
        for file in inner_folder.iterdir():
            if file.is_file():
                print(f"Анализ файла {inner_folder.name}/{file.name}...")
                dir = f"data/boom/{inner_folder.name}/{file.name}"
                signals = load(dir, type="ALL")
                set_coor(signals)
                preprocess(signals)
                pick_signals(signals, 0)
                calc = Calculator()
                good_signals = check_arrivals(signals)
                explosion = calc.locate_explosion(good_signals)
                calc.calculate_distances(good_signals, explosion)
                calc.calculate_max_displacement(good_signals)
                calc.calculate_local_magnitude(good_signals, explosion)
                calc.calculate_intensity(good_signals)
                res = f"{dir:<40} {len(good_signals):>8} {calc.ml_median:>8.2f} {calc.intensity_median:>6.1f}"
                result.append(res)
    header = f"{'Файл':<40} {'Станций':>8} {'ML':>8} {'I':>6}"
    separator = "─" * len(header)
    print(header)
    print(separator)
    for res in result:
        print(res)

def all_booms():
    result = []
    for i in range(1, 6):
        dir = f"data/boom/longdata{i}.txt"
        print(f"Анализ файла {dir}")
        signals = load(dir, type="ALL")
        set_coor(signals)
        preprocess(signals)
        pick_signals(signals, 0)
        calc = Calculator()
        good_signals = check_arrivals(signals)
        explosion = calc.locate_explosion(good_signals)
        calc.calculate_distances(good_signals, explosion)
        calc.calculate_max_displacement(good_signals)
        calc.calculate_local_magnitude(good_signals, explosion)
        calc.calculate_intensity(good_signals)
        res = f"{dir:<40}|{len(good_signals):>8} |{min(s.ml for s in good_signals.values()):>8.2f} |{calc.ml_median:>10.2f} |{calc.intensity_median:>8.2f} |"
        result.append(res)
    header = f"{'Файл':<40}|{'Станций':>8} |{'M_far':>8} |{'M_median':>10} |{'I':>8} |"
    # separator = "─" * len(header)
    separator = "─" * 40 + "+" + "─" * 8 + "─+" + "─" * 8 + "─+" + "─" * 10 + "─+" + "─" * 8 + "─|"
    print(header)
    print(separator)
    for res in result:
        print(res)
if __name__ == '__main__':
    all_stations_in()
    # test()
    # all_booms()

"""
Файл                                    | Станций |   M_far |  M_median |       I |
────────────────────────────────────────+─────────+─────────+───────────+─────────|
data/boom/longdata1.txt                 |       6 |    2.17 |      2.80 |    2.51 |
data/boom/longdata2.txt                 |       6 |    2.21 |      2.54 |    2.36 |
data/boom/longdata3.txt                 |       4 |    2.42 |      2.56 |    2.90 |
data/boom/longdata4.txt                 |       4 |    1.71 |      2.80 |    1.74 |
data/boom/longdata5.txt                 |       5 |    2.44 |      2.92 |    3.57 |
"""