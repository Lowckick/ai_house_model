"""Disjoint, deterministic train/validation/test scenarios with a simulation clock."""
import datetime as dt
import hashlib
import random
from dataclasses import dataclass, asdict

from smart_house.environment.indoor_state import IndoorPhysicsSimulator
from smart_house.environment.weather import SyntheticWeatherGenerator


@dataclass(frozen=True)
class Scenario:
    split: str
    index: int
    seed: int
    start: dt.datetime
    occupants: int
    temperature_c: float
    humidity_pct: float
    co2_ppm: float
    pm25_ugm3: float
    dirt_level: float
    weather_offset_c: float

    def describe(self):
        return {**asdict(self), 'start': self.start.isoformat()}

    def environment(self):
        physics = IndoorPhysicsSimulator(n_occupants=self.occupants, seed=self.seed, start_time=self.start)
        state = physics.reset()
        for field in ('temperature_c', 'humidity_pct', 'co2_ppm', 'pm25_ugm3', 'dirt_level'):
            setattr(state, field, getattr(self, field))
        state.room_temps = {room: self.temperature_c for room in state.room_temps}
        state.occupancy, state.total_occupants, state.occupant_activity = physics._occupancy_sim.update(self.start.hour)
        state.comfort_score = physics._compute_comfort()
        return physics

    def weather_trace(self, steps):
        generator = SyntheticWeatherGenerator(seed=self.seed)
        trace = []
        for step in range(steps):
            time = self.start + dt.timedelta(minutes=step * IndoorPhysicsSimulator.TIMESTEP_MINUTES)
            weather = generator.next(hour_of_day=time.hour, day_of_year=time.timetuple().tm_yday)
            weather.timestamp = time
            weather.temperature_c = round(weather.temperature_c + self.weather_offset_c, 1)
            trace.append(weather)
        return trace


def make_scenario(split, index, seed=42):
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('Unknown scenario split')
    digest = hashlib.sha256(f'{split}:{seed}:{index}'.encode()).digest()
    scenario_seed = int.from_bytes(digest[:4], 'big')
    rng = random.Random(scenario_seed)
    # Cycle across seasons even in a short experiment. Years and seeds never overlap splits.
    month = (index * 5) % 12 + 1
    year = {'train': 2023 + (index // 12) % 2, 'validation': 2025, 'test': 2026}[split]
    day = rng.randint(1, 27)
    start = dt.datetime(year, month, day, rng.choice((0, 6, 12, 18)), tzinfo=dt.timezone.utc)
    return Scenario(split, index, scenario_seed, start, rng.randint(0, 4),
                    rng.uniform(16, 29), rng.uniform(25, 80), rng.uniform(400, 2200),
                    rng.uniform(3, 70), rng.uniform(0, 70), rng.uniform(-6, 6))
