from data_io import DataLoader
from analyzer import SignalAnalyzer


def main():
    data_file = 'data/longdata4.txt'

    # 1. Загрузка данных
    print("Загрузка данных...")
    loader = DataLoader(data_file, fs=1000.0)
    signals = loader.load_signals()

    if not signals:
        print("Данные не загружены.")
        return

    # 2. Инициализация анализатора
    analyzer = SignalAnalyzer(signals)

    # 3. Вывод статистики смещения нуля
    analyzer.print_statistics()

    # 4. Построение спектра для первой доступной станции
    for i in range(6):
        sample_station = list(signals.keys())[i]
        print(f"Строим спектр для станции {sample_station}...")
        analyzer.plot_spectrum(sample_station, channel=1, max_freq=100)

    # Выводим графики на экран
    analyzer.show_plots()

    # --- ЗАДЕЛ НА БУДУЩЕЕ ---
    # Когда мы подберем фильтр, мы раскомментируем этот код:
    # print("Запуск фильтрации...")
    # for st_name, sig in signals.items():
    #     sig.preprocess(lowcut=2.0, highcut=25.0) # Частоты поменяем на основе спектра!


if __name__ == '__main__':
    main()