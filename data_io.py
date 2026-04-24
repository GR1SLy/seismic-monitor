import pandas as pd
from seismic_signal import SeismicSignal


class DataLoader:
    def __init__(self, data_filepath, fs=1000.0):
        self.data_filepath = data_filepath
        self.fs = fs

    def load_signals(self):
        print("--- ЗАПУСК ОТЛАДКИ ДАННЫХ ---")
        # Читаем файл
        df = pd.read_csv(self.data_filepath, sep='\t')

        print(f"Всего колонок в файле: {len(df.columns)}")

        # 1. ДЕБАГ: Ищем мусор руками по всем колонкам
        for col in df.columns:
            for idx, val in enumerate(df[col]):
                if pd.isna(val):
                    continue  # Обычные пустоты пропускаем, это нормально
                try:
                    # Пытаемся привести к float (на случай запятых тоже проверяем)
                    float(str(val).replace(',', '.'))
                except ValueError:
                    # idx + 2, потому что в DataFrame нумерация с 0, плюс 1 строка заголовков
                    print(
                        f"[НАЙДЕН МУСОР] Строка в файле: ~{idx + 2} | Колонка: '{col}' | Значение: '{val}' | Тип: {type(val)}")

        print("--- ОКОНЧАНИЕ ОТЛАДКИ ---\n")

        # 2. Оставляем ТОЛЬКО нужные колонки станций, удаляем весь остальной мусор
        seismic_cols = [c for c in df.columns if str(c).startswith('#ST')]
        df = df[seismic_cols]

        # 3. Безопасная обработка: работаем строго по колонкам
        for col in df.columns:
            # Превращаем в числа, убивая оставшийся мусор (он станет NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Теперь интерполируем только числовые колонки
        df = df.interpolate(method='linear').fillna(0.0)

        # Собираем имена станций
        station_names = set()
        for col in df.columns:
            st_name = col.split('(')[0].replace('#', '')
            station_names.add(st_name)

        signals = {}
        for st in sorted(station_names):
            col_ch1 = f'#{st}(ch1)'
            col_ch2 = f'#{st}(ch2)'
            col_ch3 = f'#{st}(ch3)'

            if col_ch1 in df.columns and col_ch2 in df.columns and col_ch3 in df.columns:
                signal_obj = SeismicSignal(
                    station_name=st,
                    ch1=df[col_ch1].values,
                    ch2=df[col_ch2].values,
                    ch3=df[col_ch3].values,
                    fs=self.fs
                )
                signals[st] = signal_obj
            else:
                print(f"[WARN] Не найдены все 3 канала для станции {st}. Пропуск.")

        return signals