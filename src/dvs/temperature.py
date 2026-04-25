from __future__ import annotations


class TemperatureProvider:
    def __init__(self, initial_celsius: float) -> None:
        self._temperature_celsius = float(initial_celsius)

    @property
    def current_celsius(self) -> float:
        return self._temperature_celsius

    def increase(self, amount: float = 1.0) -> float:
        self._temperature_celsius += amount
        return self._temperature_celsius

    def decrease(self, amount: float = 1.0) -> float:
        self._temperature_celsius -= amount
        return self._temperature_celsius
