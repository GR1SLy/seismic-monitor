import matplotlib.pyplot as plt
from data_io import DataLoader
from picker import PhasePicker


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
    data_file = 'data/longdata4.txt'

    # 2. Инициализируем загрузчик и читаем данные
    print("Загрузка данных...")
    loader = DataLoader(data_file, fs=1000.0)
    signals = loader.load_signals()

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

    picker.plot_picking('ST7')
    picker.plot_picking('ST1')
    picker.plot_picking('ST2')
    picker.plot_picking('ST3')
    picker.plot_picking('ST5')
    picker.plot_picking('ST6')

    # 4. Вызов функции визуализации для каждой станции
    # for st_name, signal in signals.items():
    #     if st_name == 'ST1':
    #         visualize_seismic(signal, 'После фильтрации')

    # for st_name, signal in signals.items():
    #     visualize_seismic(signal)

    # Отображаем все созданные окна с графиками разом
    print("Графики выведены на экран. Закройте окна графиков, чтобы завершить программу.")
    plt.show()


    for st_name, signal in signals.items():
        print(st_name, signal.snr, signal.arrival_time, signal.peak_sta_lta)

if __name__ == '__main__':
    main()