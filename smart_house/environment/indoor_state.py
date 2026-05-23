"""
Indoor environment state simulator.
Models realistic physics of how indoor conditions evolve based on
outdoor weather, device states, occupancy and time.
"""

import math
import random
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from smart_house.environment.weather import WeatherSnapshot


# ---------------------------------------------------------------------------
# Device action space (what the RL agent can control)
# ---------------------------------------------------------------------------

@dataclass
class DeviceState:
    """Current state of all controllable smart home devices."""

    # HVAC  0=off, 1=heating low, 2=heating high, 3=cooling low, 4=cooling high
    hvac_mode: int = 0
    # Lights per room: 0=off, 1-10 brightness level
    lights: Dict[str, int] = field(default_factory=lambda: {
        "living_room": 0, "bedroom": 0, "kitchen": 0,
        "bathroom": 0, "hallway": 0, "office": 0
    })
    # Blinds per room: 0=closed, 100=fully open
    blinds: Dict[str, float] = field(default_factory=lambda: {
        "living_room": 50.0, "bedroom": 50.0, "kitchen": 50.0, "office": 50.0
    })
    # Air purifier: 0=off, 1=low, 2=medium, 3=high
    air_purifier: int = 0
    # Humidifier: 0=off, 1=humidify, 2=dehumidify
    humidifier_mode: int = 0
    # Cleaner robot: 0=docked, 1=cleaning, 2=returning
    cleaner_bot: int = 0
    # Ventilation fan: 0-3 speed
    ventilation: int = 0
    # Security system: 0=off, 1=armed_home, 2=armed_away
    security: int = 0
    # Smart curtains: same as blinds but separate
    curtains_closed: Dict[str, bool] = field(default_factory=lambda: {
        "bedroom": False, "living_room": False
    })

    def to_vector(self) -> List[float]:
        v = [float(self.hvac_mode)]
        for room in sorted(self.lights):
            v.append(float(self.lights[room]))
        for room in sorted(self.blinds):
            v.append(self.blinds[room] / 100.0)
        v += [float(self.air_purifier), float(self.humidifier_mode),
              float(self.cleaner_bot), float(self.ventilation)]
        return v


# ---------------------------------------------------------------------------
# Room definition
# ---------------------------------------------------------------------------

ROOMS = {
    "living_room": {"volume_m3": 60.0, "windows": 3, "insulation": 0.7},
    "bedroom":     {"volume_m3": 30.0, "windows": 1, "insulation": 0.8},
    "kitchen":     {"volume_m3": 20.0, "windows": 1, "insulation": 0.6},
    "bathroom":    {"volume_m3": 8.0,  "windows": 0, "insulation": 0.9},
    "hallway":     {"volume_m3": 10.0, "windows": 0, "insulation": 0.85},
    "office":      {"volume_m3": 20.0, "windows": 2, "insulation": 0.75},
}


# ---------------------------------------------------------------------------
# Indoor state
# ---------------------------------------------------------------------------

@dataclass
class IndoorState:
    """Full snapshot of the indoor environment."""

    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

    # Thermal
    temperature_c: float = 21.0          # average indoor temp
    room_temps: Dict[str, float] = field(default_factory=lambda: {
        r: 21.0 for r in ROOMS
    })

    # Air quality
    humidity_pct: float = 50.0           # indoor relative humidity
    co2_ppm: float = 450.0               # CO2 concentration (fresh: 400, stuffy: >1500)
    voc_ppb: float = 100.0               # volatile organic compounds
    pm25_ugm3: float = 5.0               # fine particulate matter (good: <12)
    pm10_ugm3: float = 10.0
    air_quality_index: float = 50.0      # composite 0-500

    # Light
    lux_per_room: Dict[str, float] = field(default_factory=lambda: {
        r: 0.0 for r in ROOMS
    })

    # Cleanliness
    dirt_level: float = 0.0              # 0-100 (100 = very dirty)
    dust_level: float = 0.0              # 0-100
    last_cleaned: Optional[datetime.datetime] = None

    # Occupancy
    occupancy: Dict[str, bool] = field(default_factory=lambda: {
        r: False for r in ROOMS
    })
    total_occupants: int = 0
    occupant_activity: str = "idle"      # idle / cooking / exercising / sleeping

    # Energy
    power_consumption_kw: float = 0.0
    energy_today_kwh: float = 0.0

    # Comfort score (0-100, computed externally)
    comfort_score: float = 50.0

    def to_vector(self) -> List[float]:
        v = [
            self.temperature_c / 40.0,
            self.humidity_pct / 100.0,
            self.co2_ppm / 2000.0,
            self.voc_ppb / 500.0,
            self.pm25_ugm3 / 75.0,
            self.air_quality_index / 500.0,
            self.dirt_level / 100.0,
            self.dust_level / 100.0,
            float(self.total_occupants) / 6.0,
            self.power_consumption_kw / 10.0,
        ]
        for room in sorted(self.occupancy):
            v.append(float(self.occupancy[room]))
        for room in sorted(self.lux_per_room):
            v.append(min(1.0, self.lux_per_room[room] / 1000.0))
        for room in sorted(self.room_temps):
            v.append(self.room_temps[room] / 40.0)
        return v


# ---------------------------------------------------------------------------
# Occupancy simulator
# ---------------------------------------------------------------------------

class OccupancySimulator:
    """Simulates realistic human presence patterns."""

    def __init__(self, n_occupants: int = 2):
        self._n = n_occupants
        self._rng = random.Random()

    def update(self, hour: int) -> tuple:
        """Returns (occupancy_dict, total, activity)."""
        # Probability of being home by hour
        home_prob = self._home_probability(hour)
        total = sum(1 for _ in range(self._n) if self._rng.random() < home_prob)

        occupancy = {r: False for r in ROOMS}
        if total > 0:
            if 22 <= hour or hour < 7:
                occupancy["bedroom"] = True
                activity = "sleeping"
            elif 7 <= hour < 9:
                occupancy["kitchen"] = True
                occupancy["bathroom"] = True
                activity = "cooking"
            elif 12 <= hour < 14:
                occupancy["kitchen"] = True
                occupancy["living_room"] = True
                activity = "cooking"
            elif 18 <= hour < 22:
                occupancy["living_room"] = True
                occupancy["kitchen"] = total > 1
                activity = "idle" if hour >= 20 else "cooking"
            else:
                occupancy["office"] = True
                activity = "idle"
        else:
            activity = "idle"

        return occupancy, total, activity

    def _home_probability(self, hour: int) -> float:
        if 0 <= hour < 7:
            return 0.95
        if 7 <= hour < 9:
            return 0.7
        if 9 <= hour < 17:
            return 0.3
        if 17 <= hour < 22:
            return 0.85
        return 0.95


# ---------------------------------------------------------------------------
# Indoor physics simulator
# ---------------------------------------------------------------------------

class IndoorPhysicsSimulator:
    """
    Simulates how indoor conditions change each time step (default: 10 min).
    Uses simplified physics / heuristic models.
    """

    TIMESTEP_MINUTES = 10
    COMFORT_TEMP_MIN = 19.0
    COMFORT_TEMP_MAX = 25.0
    COMFORT_HUM_MIN = 35.0
    COMFORT_HUM_MAX = 65.0
    TARGET_CO2 = 600.0
    TARGET_LUX = {
        "living_room": 300,
        "bedroom": 100,
        "kitchen": 500,
        "bathroom": 200,
        "hallway": 150,
        "office": 500,
    }

    def __init__(self, n_occupants: int = 2):
        self._occupancy_sim = OccupancySimulator(n_occupants)
        self._rng = random.Random()
        self.state = IndoorState()
        self._step = 0
        self._cleaning_progress = 0.0

    def reset(self) -> IndoorState:
        self.state = IndoorState(
            temperature_c=self._rng.uniform(18.0, 24.0),
            humidity_pct=self._rng.uniform(40.0, 60.0),
            co2_ppm=self._rng.uniform(400.0, 600.0),
            dirt_level=self._rng.uniform(5.0, 30.0),
            dust_level=self._rng.uniform(5.0, 20.0),
        )
        self._step = 0
        self._cleaning_progress = 0.0
        return self.state

    def step(self, weather: WeatherSnapshot, devices: DeviceState) -> IndoorState:
        dt = self.TIMESTEP_MINUTES / 60.0  # hours
        h = datetime.datetime.now().hour

        # -- Occupancy --
        occ, n_occ, activity = self._occupancy_sim.update(h)
        self.state.occupancy = occ
        self.state.total_occupants = n_occ
        self.state.occupant_activity = activity

        # -- Temperature dynamics --
        self._update_temperature(weather, devices, dt, n_occ, activity)

        # -- Humidity dynamics --
        self._update_humidity(weather, devices, dt, n_occ, activity)

        # -- Air quality dynamics --
        self._update_air_quality(devices, dt, n_occ, activity)

        # -- Lighting --
        self._update_lighting(weather, devices, h)

        # -- Cleanliness --
        self._update_cleanliness(devices, dt, n_occ, activity)

        # -- Power consumption --
        self._update_power(devices)

        # -- Comfort score --
        self.state.comfort_score = self._compute_comfort()

        self.state.timestamp = datetime.datetime.now()
        self._step += 1
        return self.state

    # ------------------------------------------------------------------
    # Private update methods
    # ------------------------------------------------------------------

    def _update_temperature(self, weather: WeatherSnapshot, devices: DeviceState,
                             dt: float, n_occ: int, activity: str):
        indoor_t = self.state.temperature_c
        outdoor_t = weather.temperature_c
        avg_insulation = sum(r["insulation"] for r in ROOMS.values()) / len(ROOMS)

        # Heat exchange with outside
        heat_loss = (outdoor_t - indoor_t) * (1.0 - avg_insulation) * 0.4 * dt

        # HVAC contribution
        hvac_delta = 0.0
        if devices.hvac_mode == 1:    hvac_delta = +1.5 * dt
        elif devices.hvac_mode == 2:  hvac_delta = +3.5 * dt
        elif devices.hvac_mode == 3:  hvac_delta = -1.5 * dt
        elif devices.hvac_mode == 4:  hvac_delta = -3.5 * dt

        # Occupant body heat (~80W per person → ~0.05°C/h per person in avg house)
        body_heat = n_occ * 0.08 * dt
        if activity == "exercising":
            body_heat *= 3.0
        elif activity == "cooking":
            body_heat += 0.5 * dt

        # Solar gain through windows
        solar_gain = (weather.solar_irradiance / 1000.0) * 0.3 * dt

        # Ventilation cooling/heating
        vent_delta = 0.0
        if devices.ventilation > 0:
            vent_delta = (outdoor_t - indoor_t) * 0.1 * devices.ventilation * dt

        new_temp = indoor_t + heat_loss + hvac_delta + body_heat + solar_gain + vent_delta
        new_temp += self._rng.gauss(0, 0.05)
        self.state.temperature_c = round(max(-5.0, min(40.0, new_temp)), 2)

        # Per-room variation
        for room, cfg in ROOMS.items():
            base = self.state.temperature_c
            variation = self._rng.gauss(0, 0.3)
            occupied_bonus = 0.3 if self.state.occupancy.get(room) else 0.0
            room_t = base + variation + occupied_bonus
            self.state.room_temps[room] = round(room_t, 2)

    def _update_humidity(self, weather: WeatherSnapshot, devices: DeviceState,
                          dt: float, n_occ: int, activity: str):
        h = self.state.humidity_pct
        outdoor_h = weather.humidity_pct

        # Exchange with outside
        h += (outdoor_h - h) * 0.05 * dt

        # Occupant moisture (breathing, sweating)
        h += n_occ * 1.5 * dt
        if activity == "cooking":
            h += 3.0 * dt
        elif activity in ("exercising", "showering"):
            h += 5.0 * dt

        # Humidifier
        if devices.humidifier_mode == 1:
            h += 4.0 * dt
        elif devices.humidifier_mode == 2:
            h -= 4.0 * dt

        # Ventilation helps equalize
        if devices.ventilation > 0:
            h += (outdoor_h - h) * 0.08 * devices.ventilation * dt

        h += self._rng.gauss(0, 0.2)
        self.state.humidity_pct = round(max(10.0, min(100.0, h)), 2)

    def _update_air_quality(self, devices: DeviceState, dt: float,
                             n_occ: int, activity: str):
        # CO2
        co2 = self.state.co2_ppm
        co2 += n_occ * 20.0 * dt               # each person adds CO2
        if activity == "exercising":
            co2 += n_occ * 40.0 * dt
        if devices.ventilation > 0:
            co2 -= devices.ventilation * 80.0 * dt
        co2 = max(400.0, co2 + self._rng.gauss(0, 5.0))
        self.state.co2_ppm = round(min(5000.0, co2), 1)

        # VOC
        voc = self.state.voc_ppb
        if activity == "cooking":
            voc += 50.0 * dt
        voc -= devices.air_purifier * 30.0 * dt
        voc += self.state.dust_level * 0.1 * dt
        voc = max(10.0, voc + self._rng.gauss(0, 2.0))
        self.state.voc_ppb = round(min(800.0, voc), 1)

        # Particulates
        pm25 = self.state.pm25_ugm3
        pm25 += self.state.dust_level * 0.05 * dt
        if activity == "cooking":
            pm25 += 15.0 * dt
        pm25 -= devices.air_purifier * 5.0 * dt
        pm25 = max(0.5, pm25 + self._rng.gauss(0, 0.3))
        self.state.pm25_ugm3 = round(min(250.0, pm25), 2)
        self.state.pm10_ugm3 = round(min(400.0, self.state.pm25_ugm3 * 1.8), 2)

        # Composite AQI (simplified US EPA linear)
        self.state.air_quality_index = round(
            min(500.0, self.state.pm25_ugm3 * 4.0 + self.state.co2_ppm * 0.05
                + self.state.voc_ppb * 0.2), 1)

    def _update_lighting(self, weather: WeatherSnapshot, devices: DeviceState, hour: int):
        solar_lux = weather.solar_irradiance * 0.12  # rough indoor daylight factor

        for room in ROOMS:
            blind_openness = devices.blinds.get(room, 50.0) / 100.0
            windows = ROOMS[room]["windows"]
            natural = solar_lux * blind_openness * windows * 0.4
            artificial = devices.lights.get(room, 0) * 35.0
            self.state.lux_per_room[room] = round(natural + artificial + self._rng.gauss(0, 2), 1)

    def _update_cleanliness(self, devices: DeviceState, dt: float,
                             n_occ: int, activity: str):
        dirt = self.state.dirt_level
        dust = self.state.dust_level

        # Dirt accumulates from occupancy and activity
        dirt += n_occ * 0.5 * dt
        if activity == "cooking":
            dirt += 1.5 * dt

        # Dust settles slowly
        dust += 0.3 * dt
        if devices.ventilation > 0:
            dust += devices.ventilation * 0.1 * dt  # movement stirs dust

        # Cleaner bot reduces dirt
        if devices.cleaner_bot == 1:
            self._cleaning_progress += dt
            reduction = min(dirt, 8.0 * dt)
            dirt = max(0.0, dirt - reduction)
            dust = max(0.0, dust - reduction * 0.5)
            if dirt < 2.0 and dust < 2.0:
                self._cleaning_progress = 0.0

        # Air purifier reduces dust
        if devices.air_purifier > 0:
            dust = max(0.0, dust - devices.air_purifier * 1.0 * dt)

        self.state.dirt_level = round(min(100.0, dirt + self._rng.gauss(0, 0.1)), 2)
        self.state.dust_level = round(min(100.0, dust + self._rng.gauss(0, 0.05)), 2)

        if devices.cleaner_bot == 1 and dirt < 1.0:
            self.state.last_cleaned = datetime.datetime.now()

    def _update_power(self, devices: DeviceState):
        kw = 0.0
        hvac_watts = [0, 800, 1800, 900, 2000]
        kw += hvac_watts[devices.hvac_mode] / 1000.0
        kw += sum(devices.lights.values()) * 5.0 / 1000.0     # 50W max per room lamp
        kw += devices.air_purifier * 50.0 / 1000.0
        kw += devices.humidifier_mode * 30.0 / 1000.0
        kw += devices.ventilation * 80.0 / 1000.0
        if devices.cleaner_bot == 1:
            kw += 0.04   # ~40W robot vacuum
        kw += 0.3       # baseline always-on (fridge, router, etc.)
        self.state.power_consumption_kw = round(kw, 3)
        self.state.energy_today_kwh = round(
            self.state.energy_today_kwh + kw * (self.TIMESTEP_MINUTES / 60.0), 4)

    def _compute_comfort(self) -> float:
        score = 100.0

        # Temperature comfort
        t = self.state.temperature_c
        if t < self.COMFORT_TEMP_MIN:
            score -= (self.COMFORT_TEMP_MIN - t) * 6.0
        elif t > self.COMFORT_TEMP_MAX:
            score -= (t - self.COMFORT_TEMP_MAX) * 6.0

        # Humidity comfort
        h = self.state.humidity_pct
        if h < self.COMFORT_HUM_MIN:
            score -= (self.COMFORT_HUM_MIN - h) * 0.8
        elif h > self.COMFORT_HUM_MAX:
            score -= (h - self.COMFORT_HUM_MAX) * 0.8

        # Air quality
        if self.state.co2_ppm > 1000:
            score -= (self.state.co2_ppm - 1000) * 0.02
        if self.state.pm25_ugm3 > 35:
            score -= (self.state.pm25_ugm3 - 35) * 0.3

        # Lighting (check occupied rooms)
        for room, is_occ in self.state.occupancy.items():
            if is_occ:
                target = self.TARGET_LUX.get(room, 300)
                actual = self.state.lux_per_room.get(room, 0)
                lux_diff = abs(actual - target)
                score -= min(15.0, lux_diff * 0.02)

        # Cleanliness
        score -= self.state.dirt_level * 0.15
        score -= self.state.dust_level * 0.1

        return round(max(0.0, min(100.0, score)), 2)
