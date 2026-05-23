"""
Weather data module.
Tries Open-Meteo (free, no key needed) and falls back to random simulation.
"""

import math
import random
import datetime
import requests
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WeatherSnapshot:
    timestamp: datetime.datetime
    temperature_c: float        # outdoor temp in Celsius
    humidity_pct: float         # outdoor relative humidity 0-100
    wind_speed_kmh: float       # km/h
    cloud_cover_pct: float      # 0-100
    precipitation_mm: float     # mm/h
    uv_index: float             # 0-11+
    is_daytime: bool
    condition: str              # clear / cloudy / rain / storm / snow

    # derived helpers
    @property
    def feels_like(self) -> float:
        """Wind-chill / heat-index approximation."""
        t = self.temperature_c
        w = self.wind_speed_kmh
        if t <= 10 and w > 4.8:
            return (13.12 + 0.6215 * t - 11.37 * (w ** 0.16) + 0.3965 * t * (w ** 0.16))
        elif t >= 27:
            h = self.humidity_pct
            hi = (-8.78469475556 + 1.61139411 * t + 2.33854883889 * h
                  - 0.14611605 * t * h - 0.012308094 * t ** 2
                  - 0.0164248277778 * h ** 2 + 0.002211732 * t ** 2 * h
                  + 0.00072546 * t * h ** 2 - 0.000003582 * t ** 2 * h ** 2)
            return hi
        return t

    @property
    def solar_irradiance(self) -> float:
        """Rough W/m² estimate from cloud cover and UV."""
        if not self.is_daytime:
            return 0.0
        base = 1000.0 * (1.0 - self.cloud_cover_pct / 100.0)
        return max(0.0, base)


# ---------------------------------------------------------------------------
# Open-Meteo API fetcher
# ---------------------------------------------------------------------------

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Kyiv, Ukraine as default location
DEFAULT_LAT = 50.4501
DEFAULT_LON = 30.5234


def fetch_open_meteo(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> Optional[WeatherSnapshot]:
    """
    Fetch current weather from Open-Meteo (free, no API key).
    Returns None on any failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "precipitation",
            "uv_index",
            "is_day",
            "weather_code",
        ],
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()["current"]

        wcode = data.get("weather_code", 0)
        condition = _wmo_to_condition(wcode)
        is_day = bool(data.get("is_day", 1))

        return WeatherSnapshot(
            timestamp=datetime.datetime.now(),
            temperature_c=float(data["temperature_2m"]),
            humidity_pct=float(data["relative_humidity_2m"]),
            wind_speed_kmh=float(data["wind_speed_10m"]),
            cloud_cover_pct=float(data["cloud_cover"]),
            precipitation_mm=float(data["precipitation"]),
            uv_index=float(data.get("uv_index", 0)),
            is_daytime=is_day,
            condition=condition,
        )
    except Exception:
        return None


def _wmo_to_condition(code: int) -> str:
    if code == 0:
        return "clear"
    if code in range(1, 4):
        return "cloudy"
    if code in range(51, 68) or code in range(80, 83):
        return "rain"
    if code in range(71, 78) or code in range(85, 87):
        return "snow"
    if code in range(95, 100):
        return "storm"
    return "cloudy"


# ---------------------------------------------------------------------------
# Random / synthetic weather generator
# ---------------------------------------------------------------------------

class SyntheticWeatherGenerator:
    """
    Generates realistic synthetic weather that evolves over time using
    smooth noise and seasonal/diurnal patterns.
    """

    def __init__(self, seed: int = 42, latitude: float = 50.0):
        self._rng = random.Random(seed)
        self._lat = latitude
        self._base_temp = 15.0
        self._base_humidity = 60.0
        self._cloud_trend = self._rng.uniform(0.3, 0.7)
        self._precip_active = False
        self._step = 0

    def next(self, hour_of_day: Optional[int] = None, day_of_year: Optional[int] = None) -> WeatherSnapshot:
        now = datetime.datetime.now()
        h = hour_of_day if hour_of_day is not None else now.hour
        doy = day_of_year if day_of_year is not None else now.timetuple().tm_yday

        # Seasonal base temperature (northern hemisphere)
        seasonal_offset = 15.0 * math.cos(2 * math.pi * (doy - 200) / 365.0)
        # Diurnal variation
        diurnal_offset = 5.0 * math.sin(2 * math.pi * (h - 6) / 24.0)

        temp = (self._base_temp + seasonal_offset + diurnal_offset
                + self._rng.gauss(0, 0.8))
        temp = max(-30.0, min(50.0, temp))

        # Humidity anti-correlates with temperature somewhat
        humidity = self._base_humidity - 0.3 * seasonal_offset + self._rng.gauss(0, 3.0)
        humidity = max(10.0, min(100.0, humidity))

        # Cloud cover slow random walk
        self._cloud_trend += self._rng.uniform(-0.05, 0.05)
        self._cloud_trend = max(0.0, min(1.0, self._cloud_trend))
        cloud = self._cloud_trend * 100.0

        # Precipitation
        precip = 0.0
        if cloud > 70 and self._rng.random() < 0.15:
            self._precip_active = True
        if cloud < 40:
            self._precip_active = False
        if self._precip_active:
            precip = self._rng.uniform(0.1, 5.0)

        wind = max(0.0, self._rng.gauss(15.0, 8.0))
        uv = max(0.0, (1.0 - cloud / 100.0) * 8.0) if 6 <= h <= 20 else 0.0
        is_day = 6 <= h <= 20

        condition = "clear"
        if cloud > 60 and precip > 2.0 and temp > 0:
            condition = "rain"
        elif cloud > 60 and precip > 2.0 and temp <= 0:
            condition = "snow"
        elif cloud > 80 and precip > 4.0:
            condition = "storm"
        elif cloud > 40:
            condition = "cloudy"

        self._step += 1
        return WeatherSnapshot(
            timestamp=now,
            temperature_c=round(temp, 1),
            humidity_pct=round(humidity, 1),
            wind_speed_kmh=round(wind, 1),
            cloud_cover_pct=round(cloud, 1),
            precipitation_mm=round(precip, 2),
            uv_index=round(uv, 1),
            is_daytime=is_day,
            condition=condition,
        )


# ---------------------------------------------------------------------------
# Unified weather provider
# ---------------------------------------------------------------------------

class WeatherProvider:
    """
    Tries real API first, falls back to synthetic.
    Caches real data for 15 minutes.
    """

    CACHE_TTL_SECONDS = 900

    def __init__(self, use_api: bool = True, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON):
        self._use_api = use_api
        self._lat = lat
        self._lon = lon
        self._cache: Optional[WeatherSnapshot] = None
        self._cache_time: Optional[datetime.datetime] = None
        self._synthetic = SyntheticWeatherGenerator()
        self._api_available = True

    def get(self) -> WeatherSnapshot:
        if self._use_api and self._api_available:
            now = datetime.datetime.now()
            if (self._cache is None or
                    (now - self._cache_time).total_seconds() > self.CACHE_TTL_SECONDS):
                result = fetch_open_meteo(self._lat, self._lon)
                if result is not None:
                    self._cache = result
                    self._cache_time = now
                    return result
                else:
                    self._api_available = False
            elif self._cache is not None:
                return self._cache

        return self._synthetic.next()

    @property
    def source(self) -> str:
        if self._use_api and self._api_available and self._cache is not None:
            return "Open-Meteo API"
        return "Synthetic Generator"
