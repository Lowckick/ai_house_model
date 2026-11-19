"""Demo data for presentations: a filled panel without sensors or manual setup.

``seed_demo`` builds a fresh history database with the last 24 hours of two simulated rooms
pushed through the live (shadow-mode) channel, one imported example file and one stored
model evaluation. The rooms follow daily routines and realistic CO₂ generation under the
simple controller, so the sensor tab shows alerts and recommendations worth discussing.
Everything is simulated and labelled as such in the readings (``sensor`` names end in -demo).
"""
import datetime as dt
from pathlib import Path

from smart_house.analytics.history import History
from smart_house.analytics.ingestion import ImportService
from smart_house.analytics.live import LiveService
from smart_house.environment.indoor_state import DeviceState, IndoorPhysicsSimulator, PhysicsParams
from smart_house.environment.weather import SyntheticWeatherGenerator
from smart_house.ml.controller import deterministic_devices

ROOT = Path(__file__).resolve().parents[2]
DEMO_PHYSICS = PhysicsParams(co2_per_person_ppm_h=250.0, co2_infiltration_per_h=0.5, occupancy='schedule',
                             source='demo')
ROOMS = (('bedroom-demo', 2, 11), ('kitchen-demo', 3, 12))   # sensor id, occupants, seed
FIELDS = ('temperature_c', 'humidity_pct', 'co2_ppm', 'pm25_ugm3', 'voc_ppb', 'dirt_level', 'total_occupants')


def simulated_day(occupants, seed, end, hours=24):
    """Readings every 10 minutes for ``hours`` up to ``end`` from a simulated room under simple rules."""
    steps = hours * 6
    start = (end - dt.timedelta(minutes=10 * steps)).replace(second=0, microsecond=0)
    sim = IndoorPhysicsSimulator(n_occupants=occupants, seed=seed, start_time=start, params=DEMO_PHYSICS)
    state = sim.reset()
    weather = SyntheticWeatherGenerator(seed=seed)
    readings = []
    for step in range(steps):
        now = start + dt.timedelta(minutes=10 * step)
        w = weather.next(hour_of_day=now.hour, day_of_year=now.timetuple().tm_yday)
        state = sim.step(w, deterministic_devices(state, w) if step else DeviceState())
        reading = {name: round(float(getattr(state, name)), 2) for name in FIELDS}
        reading['total_occupants'] = int(state.total_occupants)
        reading['timestamp'] = (now + dt.timedelta(minutes=10)).isoformat()
        readings.append(reading)
    return readings


def seed_demo(db_path, model, now=None):
    """Create (or replace) a demo database and return a short summary of what it contains."""
    db_path = Path(db_path)
    for suffix in ('', '-wal', '-shm', '-journal'):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    now = (now or dt.datetime.now(dt.timezone.utc)).replace(second=0, microsecond=0)
    imports = ImportService(model, History(db_path))
    summary = {'database': str(db_path), 'sensors': {}}
    clock_time = [now]   # each reading is 'received' at its own timestamp
    live = LiveService(imports, clock=lambda: clock_time[0])
    for sensor, occupants, seed in ROOMS:
        readings = simulated_day(occupants, seed, now)
        for reading in readings:
            clock_time[0] = dt.datetime.fromisoformat(reading['timestamp'])
            live.ingest(sensor, reading)
        summary['sensors'][sensor] = len(readings)
    example = ROOT / 'examples' / 'readings.json'
    summary['import_id'] = imports.import_text(example.name, example.read_text(encoding='utf-8'))['id']
    summary['evaluation_id'] = imports.evaluate(episodes=3, steps=144, seed=42)['id']
    return summary
