import datetime
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from smart_house.environment.indoor_state import DeviceState
from smart_house.ml.agent import action_index_to_devices, encode_state
from smart_house.ml.trainer import Trainer


SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "screenshots")

HVAC_LABELS = {0: "OFF", 1: "HEAT LOW", 2: "HEAT HIGH", 3: "COOL LOW", 4: "COOL HIGH"}
CLEANER_LABELS = {0: "Docked", 1: "Cleaning", 2: "Returning"}
SECURITY_LABELS = {0: "Disarmed", 1: "Armed Home", 2: "Armed Away"}
PURIFIER_LABELS = {0: "OFF", 1: "Low", 2: "Medium", 3: "High"}
HUMIDIFIER_LABELS = {0: "OFF", 1: "Humidify", 2: "Dehumidify"}
VENT_LABELS = {0: "OFF", 1: "Low", 2: "Medium", 3: "High"}


def _status_color(value: float, good_min: float, good_max: float, warn_min: float, warn_max: float) -> str:
    if good_min <= value <= good_max:
        return "#2ca02c"
    if warn_min <= value <= warn_max:
        return "#ffbf00"
    return "#d62728"


def _draw_panel(ax, x: float, y: float, w: float, h: float, title: str, lines: List[str], accent: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.5, edgecolor=accent, facecolor="#f8fbff"))
    ax.add_patch(Rectangle((x, y + h - 0.06), w, 0.06, linewidth=0, facecolor=accent))
    ax.text(x + 0.015, y + h - 0.038, title, fontsize=12, color="white", weight="bold", va="center")
    line_y = y + h - 0.105
    for line in lines:
        ax.text(x + 0.02, line_y, line, fontsize=10, color="#1b1b1b", va="top", family="DejaVu Sans Mono")
        line_y -= 0.045


def render_dashboard_snapshot(path: str, step: int, indoor, devices, weather, reward: float, epsilon: float, source: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(16, 9), facecolor="#e9eef6")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.add_patch(Rectangle((0.02, 0.92), 0.96, 0.06, linewidth=0, facecolor="#163a5f"))
    ax.text(0.04, 0.95, "Smart House AI Control System", fontsize=18, color="white", weight="bold", va="center")
    ax.text(0.96, 0.95, f"{ts} | Weather: {source}", fontsize=11, color="white", ha="right", va="center")

    weather_lines = [
        f"Condition: {weather.condition.upper()}",
        f"Temperature: {weather.temperature_c:+.1f} C",
        f"Feels like: {weather.feels_like:+.1f} C",
        f"Humidity: {weather.humidity_pct:.0f}%",
        f"Wind: {weather.wind_speed_kmh:.1f} km/h",
        f"Cloud cover: {weather.cloud_cover_pct:.0f}%",
        f"Precipitation: {weather.precipitation_mm:.1f} mm/h",
        f"Solar: {weather.solar_irradiance:.0f} W/m2",
    ]
    indoor_lines = [
        f"Avg temperature: {indoor.temperature_c:.1f} C",
        f"Humidity: {indoor.humidity_pct:.1f}%",
        f"CO2: {indoor.co2_ppm:.0f} ppm",
        f"PM2.5: {indoor.pm25_ugm3:.1f} ug/m3",
        f"AQI: {indoor.air_quality_index:.0f}",
        f"Dirt: {indoor.dirt_level:.1f}%",
        f"Dust: {indoor.dust_level:.1f}%",
        f"Occupants: {indoor.total_occupants} ({indoor.occupant_activity})",
        f"Comfort: {indoor.comfort_score:.1f}%",
        f"Power: {indoor.power_consumption_kw:.3f} kW",
    ]
    device_lines = [
        f"HVAC: {HVAC_LABELS.get(devices.hvac_mode, '?')}",
        f"Ventilation: {VENT_LABELS.get(devices.ventilation, '?')}",
        f"Air purifier: {PURIFIER_LABELS.get(devices.air_purifier, '?')}",
        f"Humidifier: {HUMIDIFIER_LABELS.get(devices.humidifier_mode, '?')}",
        f"Cleaner bot: {CLEANER_LABELS.get(devices.cleaner_bot, '?')}",
        f"Security: {SECURITY_LABELS.get(devices.security, '?')}",
        f"Lights on: {sum(1 for v in devices.lights.values() if v > 0)}/{len(devices.lights)} rooms",
    ]
    room_lines = [
        f"{room.replace('_', ' ').title()[:16]:16} {indoor.room_temps[room]:5.1f} C  {indoor.lux_per_room[room]:5.0f} lx  {'YES' if indoor.occupancy[room] else 'no'}"
        for room in sorted(indoor.room_temps)
    ]
    ml_lines = [
        f"Episode: demo snapshot",
        f"Step: {step}/144 ({step * 10} min)",
        f"Reward: {reward:+.4f}",
        f"Epsilon: {epsilon:.4f}",
        f"State dim: 54",
        f"Actions: 1920",
    ]

    _draw_panel(ax, 0.03, 0.53, 0.29, 0.35, "Outdoor Weather", weather_lines, "#d49b00")
    _draw_panel(ax, 0.355, 0.53, 0.29, 0.35, "Indoor Environment", indoor_lines, "#008c9e")
    _draw_panel(ax, 0.68, 0.53, 0.29, 0.35, "Devices", device_lines, "#2ca02c")
    _draw_panel(ax, 0.03, 0.13, 0.615, 0.34, "Room Status", room_lines, "#8a2be2")
    _draw_panel(ax, 0.68, 0.13, 0.29, 0.34, "ML Agent", ml_lines, "#333333")

    comfort_color = _status_color(indoor.comfort_score, 75, 100, 55, 100)
    temp_color = _status_color(indoor.temperature_c, 19, 25, 16, 28)
    co2_color = _status_color(indoor.co2_ppm, 0, 800, 0, 1200)
    ax.add_patch(Rectangle((0.02, 0.03), 0.96, 0.065, linewidth=0, facecolor="#ffffff"))
    ax.text(0.05, 0.062, f"Comfort: {indoor.comfort_score:.1f}%", fontsize=13, color=comfort_color, weight="bold", va="center")
    ax.text(0.30, 0.062, f"Temperature: {indoor.temperature_c:.1f} C", fontsize=13, color=temp_color, weight="bold", va="center")
    ax.text(0.55, 0.062, f"CO2: {indoor.co2_ppm:.0f} ppm", fontsize=13, color=co2_color, weight="bold", va="center")
    ax.text(0.78, 0.062, f"Energy today: {indoor.energy_today_kwh:.2f} kWh", fontsize=13, color="#163a5f", weight="bold", va="center")

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_simulation_screenshots(output_dir: str = None, count: int = 4, use_api_weather: bool = False, steps: int = 144) -> List[str]:
    output_dir = output_dir or SCREENSHOT_DIR
    os.makedirs(output_dir, exist_ok=True)
    trainer = Trainer(use_api_weather=use_api_weather)
    trainer.agent.epsilon = 0.0
    indoor = trainer.physics.reset()
    devices = DeviceState()
    weather = trainer.weather_provider.get()
    count = max(1, min(count, steps))
    if count == 1:
        target_steps = {steps // 2}
    else:
        target_steps = {round(i * (steps - 1) / (count - 1)) for i in range(count)}
    created = []

    for step in range(steps):
        state = encode_state(indoor, weather, devices)
        action_idx = trainer.agent.select_action(state)
        raw_devices = action_index_to_devices(action_idx, devices)
        devices = trainer.rule_layer.apply(raw_devices, indoor, weather)
        if step % 30 == 0:
            weather = trainer.weather_provider.get()
        indoor = trainer.physics.step(weather, devices)
        reward = trainer.agent.reward_fn.compute(indoor, devices)
        if step in target_steps:
            name = f"dashboard_step_{step:03d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = os.path.join(output_dir, name)
            created.append(render_dashboard_snapshot(path, step, indoor, devices, weather, reward, trainer.agent.epsilon, trainer.weather_provider.source))

    return created
