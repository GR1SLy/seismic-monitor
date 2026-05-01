import pandas as pd
import os
import re
from seismic_signal import SeismicSignal


class DataLoader:
    def __init__(self, fs=1000.0):
        self.data_filepath = ""
        self.fs = fs

    def load_signals_all(self, filename):
        print("--- ЗАПУСК ОТЛАДКИ ДАННЫХ ---")
        self.data_filepath = filename
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

    def load_signals_per_station(self, file_path):
        """
        Загружает сейсмические сигналы для всех станций, синхронизированных по дате и времени.

        Параметры:
        file_path : str
            Путь к любому файлу станции в новой структуре, например
            "data/new/ST1/ST1 (mode normal) 2024-01-01 12-00-00.txt".
            Дата и время будут извлечены из имени файла, и по ним найдутся соответствующие
            файлы для остальных станций ST2..ST7.

        Возвращает:
        signals : dict
            Словарь {station_name: SeismicSignal} по всем успешно загруженным станциям.
        """
        # --- 1. Извлечение даты/времени из имени файла ---
        base_name = os.path.basename(file_path)
        # Ожидаемый шаблон: ST<номер> (mode normal) <дата_время>.txt
        pattern = r'^ST(\d+) \(mode_normal\) (.+)\.txt$'
        match = re.match(pattern, base_name)
        if not match:
            raise ValueError(f"Не удалось разобрать имя файла: {base_name}")

        datetime_part = match.group(2)  # например, "2024-01-01 12-00-00"

        # Базовая директория, где лежат папки ST1..ST7
        # Предполагаем, что self.base_dir = "data/new" (можно передать или заменить на self.data_filepath)
        base_dir = getattr(self, 'base_dir', 'data/new')

        signals = {}
        print("--- ЗАПУСК ЗАГРУЗКИ ПО НОВОЙ СТРУКТУРЕ ---")
        print(f"Ориентируемся по дате/времени: {datetime_part}")

        # --- 2. Обход всех возможных станций ST1..ST7 ---
        for st_num in range(1, 8):
            st_name = f"ST{st_num}"
            file_name = f"{st_name} (mode_normal) {datetime_part}.txt"
            full_path = os.path.join(base_dir, st_name, file_name)

            if not os.path.isfile(full_path):
                print(f"[WARN] Файл {full_path} не найден. Станция {st_name} будет пропущена.")
                continue

            # --- 3. Чтение файла станции ---
            # Структура: три столбца (ch1, ch2, ch3) с табуляцией
            df = pd.read_csv(full_path, sep='\t', header=None)

            if df.shape[1] < 3:
                print(f"[WARN] В файле {full_path} менее трёх колонок. Станция {st_name} пропущена.")
                continue

            # Выделяем первые три колонки
            ch1_raw = df.iloc[:, 0]
            ch2_raw = df.iloc[:, 1]
            ch3_raw = df.iloc[:, 2]

            # --- 4. Отладка: поиск мусора (аналогично старому методу) ---
            print(f"--- Отладка для {st_name} ---")
            for ch_name, series in [('ch1', ch1_raw), ('ch2', ch2_raw), ('ch3', ch3_raw)]:
                for idx, val in enumerate(series):
                    if pd.isna(val):
                        continue
                    try:
                        float(str(val).replace(',', '.'))
                    except ValueError:
                        print(
                            f"[НАЙДЕН МУСОР] Строка в файле: ~{idx + 2} | Канал: {ch_name} | "
                            f"Значение: '{val}' | Тип: {type(val)}"
                        )
            print(f"--- Завершение отладки для {st_name} ---\n")

            # --- 5. Безопасное преобразование в числа ---
            ch1 = pd.to_numeric(ch1_raw, errors='coerce')
            ch2 = pd.to_numeric(ch2_raw, errors='coerce')
            ch3 = pd.to_numeric(ch3_raw, errors='coerce')

            # --- 6. Интерполяция и заполнение нулями (как в старом методе) ---
            ch1 = ch1.interpolate(method='linear').fillna(0.0)
            ch2 = ch2.interpolate(method='linear').fillna(0.0)
            ch3 = ch3.interpolate(method='linear').fillna(0.0)

            # --- 7. Создание объекта SeismicSignal ---
            signal_obj = SeismicSignal(
                station_name=st_name,
                ch1=ch1.values,
                ch2=ch2.values,
                ch3=ch3.values,
                fs=self.fs
            )
            signals[st_name] = signal_obj

        print("--- ЗАГРУЗКА ЗАВЕРШЕНА ---")
        return signals