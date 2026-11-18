"""Historical hourly weather for recorded readings (Open-Meteo, no API key).

Readings analysed offline used to assume fixed outdoor weather (15 °C, 60 %), which made
winter or summer recommendations unrealistic. ``ArchiveWeather`` looks up the hour of each
reading: the archive API for older dates, the forecast API (which also serves the recent
past) for the last days. Results are cached in a JSON file per location and day, so repeated
imports are reproducible offline. Any failure falls back to the fixed assumption and the
reading records which source was used.
"""
import datetime as dt
import json
import threading
from pathlib import Path

from smart_house.environment.weather import DEFAULT_LAT, DEFAULT_LON, WeatherSnapshot

ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
HOURLY = ('temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'cloud_cover', 'precipitation', 'is_day')
ARCHIVE_DELAY_DAYS = 5
MAX_DAYS_PER_REQUEST = 366


def fixed_weather(timestamp):
    """The documented offline assumption: 15 °C, 60 %, calm, half cloud, day 06–20 h."""
    return WeatherSnapshot(timestamp, 15.0, 60.0, 0.0, 50.0, 0.0, 0.0, 6 <= timestamp.hour < 20, 'cloudy')


def _utc_hour(timestamp):
    """Naive timestamps are interpreted as UTC."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.astimezone(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)


def _condition(temperature, precipitation, cloud):
    if precipitation >= 0.2:
        return 'snow' if temperature <= 0 else ('storm' if precipitation >= 7.5 else 'rain')
    return 'clear' if cloud < 30 else 'cloudy'


class ArchiveWeather:
    """Hourly outdoor weather at a location; ``session`` is ``requests``-compatible (mockable)."""

    def __init__(self, lat=DEFAULT_LAT, lon=DEFAULT_LON, cache_path=None, session=None, timeout=15, today=None):
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError('Latitude must be -90..90 and longitude -180..180')
        self.lat, self.lon = round(float(lat), 4), round(float(lon), 4)
        self.cache_path = Path(cache_path) if cache_path else None
        self.timeout = timeout
        self._today = today
        self._lock = threading.Lock()
        if session is None:
            import requests
            session = requests
        self._session = session
        self._cache = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                self._cache = {}
        self.errors = []

    def _key(self, day):
        return f'{self.lat},{self.lon},{day.isoformat()}'

    def _fetch(self, url, start, end):
        response = self._session.get(url, timeout=self.timeout, params={
            'latitude': self.lat, 'longitude': self.lon, 'start_date': start.isoformat(),
            'end_date': end.isoformat(), 'hourly': ','.join(HOURLY), 'timezone': 'UTC',
            'wind_speed_unit': 'kmh'})
        response.raise_for_status()
        hourly = response.json()['hourly']
        rows = {}
        for index, stamp in enumerate(hourly['time']):
            values = {name: hourly[name][index] for name in HOURLY}
            if all(value is not None for value in values.values()):
                rows[stamp] = values
        return rows

    def prefetch(self, timestamps):
        """Download every missing day in at most a few requests; returns the number of failures."""
        today = self._today or dt.datetime.now(dt.timezone.utc).date()
        days = sorted({_utc_hour(t).date() for t in timestamps} - {
            dt.date.fromisoformat(key.rsplit(',', 1)[1]) for key in self._cache
            if key.startswith(f'{self.lat},{self.lon},')})
        days = [day for day in days if day <= today + dt.timedelta(days=15)]
        failures = 0
        with self._lock:
            for recent in (False, True):
                group = [d for d in days if (d > today - dt.timedelta(days=ARCHIVE_DELAY_DAYS)) == recent]
                while group:
                    start = group[0]
                    chunk = [d for d in group if (d - start).days < MAX_DAYS_PER_REQUEST]
                    group = group[len(chunk):]
                    try:
                        rows = self._fetch(FORECAST_URL if recent else ARCHIVE_URL, chunk[0], chunk[-1])
                    except Exception as exc:  # network, HTTP, malformed JSON: fall back per reading
                        failures += 1
                        self.errors.append(f'{chunk[0]}..{chunk[-1]}: {type(exc).__name__}: {exc}')
                        continue
                    for day in chunk:
                        prefix = day.isoformat()
                        self._cache[self._key(day)] = {stamp[11:13]: values for stamp, values in rows.items()
                                                       if stamp.startswith(prefix)}
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.cache_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(self._cache, sort_keys=True), encoding='utf-8')
                temporary.replace(self.cache_path)
        return failures

    def get(self, timestamp):
        """(WeatherSnapshot, source); source is 'open-meteo' or 'fixed' when the hour is unknown."""
        hour = _utc_hour(timestamp)
        values = self._cache.get(self._key(hour.date()), {}).get(f'{hour.hour:02d}')
        if values is None:
            return fixed_weather(timestamp), 'fixed'
        temperature, precipitation = float(values['temperature_2m']), float(values['precipitation'])
        cloud = float(values['cloud_cover'])
        return WeatherSnapshot(timestamp, temperature, float(values['relative_humidity_2m']),
                               float(values['wind_speed_10m']), cloud, precipitation, 0.0,
                               bool(values['is_day']), _condition(temperature, precipitation, cloud)), 'open-meteo'
