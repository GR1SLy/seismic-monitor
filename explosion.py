class Explosion:
    def __init__(self, x, y, t0, speed, rms):
        self.x = x
        self.y = y
        self.t0 = t0
        self.speed = speed
        self.rms = rms

    def __str__(self):
        return f"""
        Explosion
        ({self.x}; {self.y})
        t0: {self.t0}
        speed: {self.speed}
        rms: {self.rms}"""
