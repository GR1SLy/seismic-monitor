import pyproj


class Explosion:
    def __init__(self, x, y, t0, speed, rms):
        self.x = x
        self.y = y
        self.t0 = t0
        self.speed = speed
        self.rms = rms
        self.wgs_x, self.wgs_y = self.__utm45n_to_wgs84(x, y)

    def __utm45n_to_wgs84(self, x, y):
        """
        Перевод из UTM зона 45N (WGS84) в географические координаты WGS84.

        Параметры:
        ----------
        x : float
            Восточное смещение (Easting) в метрах.
        y : float
            Северное смещение (Northing) в метрах.

        Возвращает:
        -----------
            Широта и долгота в десятичных градусах (lat, lon).
        """
        # Создаём трансформер: из UTM zone 45N (EPSG:32645) в WGS84 (EPSG:4326)
        transformer = pyproj.Transformer.from_crs(
            "EPSG:32645",  # UTM zone 45N
            "EPSG:4326",  # WGS84 geographic
            always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return lat, lon

    def __str__(self):
        return f"""
---------------------EXPLOSION---------------------
Mercator: ({self.x:.5f};\t{self.y:.5f})
   WGS84: ({self.wgs_x:.5f};\t{self.wgs_y:.5f})
      t0: {self.t0:.3f} s
   speed: {self.speed:.3f} m/s
     rms: {self.rms:.3f} s
----------------------------------------------------"""