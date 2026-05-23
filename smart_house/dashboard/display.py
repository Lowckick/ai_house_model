"""
Rich terminal dashboard for the Smart House ML system.
Displays real-time indoor/outdoor state, device status, and ML metrics.
"""

import os
import time
import datetime
import sqlite3
import threading
from typing import Optional, List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.columns import Columns
from rich import box

from smart_house.environment.indoor_state import DeviceState, IndoorState, ROOMS
from smart_house.environment.weather import WeatherSnapshot


console = Console()

HVAC_LABELS = {0: "OFF", 1: "HEAT LOW", 2: "HEAT HIGH", 3: "COOL LOW", 4: "COOL HIGH"}
CLEANER_LABELS = {0: "Docked", 1: "Cleaning", 2: "Returning"}
SECURITY_LABELS = {0: "Disarmed", 1: "Armed Home", 2: "Armed Away"}
PURIFIER_LABELS = {0: "OFF", 1: "Low", 2: "Medium", 3: "High"}
HUMIDIFIER_LABELS = {0: "OFF", 1: "Humidify", 2: "Dehumidify"}
VENT_LABELS = {0: "OFF", 1: "Low", 2: "Med", 3: "High"}

# Color thresholds
def temp_color(t: float) -> str:
    if t < 16: return "blue"
    if t < 19: return "cyan"
    if t <= 25: return "green"
    if t <= 28: return "yellow"
    return "red"

def co2_color(co2: float) -> str:
    if co2 < 800: return "green"
    if co2 < 1200: return "yellow"
    return "red"

def air_quality_color(aqi: float) -> str:
    if aqi < 50: return "bright_green"
    if aqi < 100: return "green"
    if aqi < 150: return "yellow"
    if aqi < 200: return "orange1"
    return "red"

def comfort_color(score: float) -> str:
    if score >= 80: return "bright_green"
    if score >= 60: return "green"
    if score >= 40: return "yellow"
    return "red"

def dirt_color(d: float) -> str:
    if d < 20: return "green"
    if d < 50: return "yellow"
    return "red"


def bar(value: float, maximum: float = 100.0, width: int = 15) -> str:
    filled = int(min(1.0, value / maximum) * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def build_weather_panel(weather: WeatherSnapshot) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="bold", width=18)
    table.add_column("Value")

    cond_icons = {"clear": "☀", "cloudy": "☁", "rain": "🌧", "storm": "⛈", "snow": "❄"}
    icon = cond_icons.get(weather.condition, "?")

    table.add_row("Condition", f"{icon} {weather.condition.upper()}")
    table.add_row("Temperature",
                  Text(f"{weather.temperature_c:+.1f}°C  (feels {weather.feels_like:+.1f}°C)",
                       style=temp_color(weather.temperature_c)))
    table.add_row("Humidity", f"{weather.humidity_pct:.0f}%")
    table.add_row("Wind", f"{weather.wind_speed_kmh:.1f} km/h")
    table.add_row("Cloud Cover", f"{bar(weather.cloud_cover_pct)} {weather.cloud_cover_pct:.0f}%")
    table.add_row("Precipitation", f"{weather.precipitation_mm:.1f} mm/h")
    table.add_row("UV Index", f"{weather.uv_index:.1f}")
    table.add_row("Solar",
                  f"{weather.solar_irradiance:.0f} W/m²  {'[Day]' if weather.is_daytime else '[Night]'}")

    return Panel(table, title="[bold yellow]🌤 Outdoor Weather[/bold yellow]",
                 border_style="yellow")


def build_indoor_panel(indoor: IndoorState) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="bold", width=18)
    table.add_column("Value")

    table.add_row("Temp (avg)",
                  Text(f"{indoor.temperature_c:.1f}°C",
                       style=temp_color(indoor.temperature_c)))
    table.add_row("Humidity",
                  f"{bar(indoor.humidity_pct)} {indoor.humidity_pct:.1f}%")
    table.add_row("CO₂",
                  Text(f"{bar(indoor.co2_ppm, 2000)} {indoor.co2_ppm:.0f} ppm",
                       style=co2_color(indoor.co2_ppm)))
    table.add_row("PM2.5",
                  f"{bar(indoor.pm25_ugm3, 75)} {indoor.pm25_ugm3:.1f} µg/m³")
    table.add_row("Air Quality",
                  Text(f"{bar(indoor.air_quality_index, 300)} AQI {indoor.air_quality_index:.0f}",
                       style=air_quality_color(indoor.air_quality_index)))
    table.add_row("Dirt Level",
                  Text(f"{bar(indoor.dirt_level)} {indoor.dirt_level:.1f}%",
                       style=dirt_color(indoor.dirt_level)))
    table.add_row("Dust Level",
                  Text(f"{bar(indoor.dust_level)} {indoor.dust_level:.1f}%",
                       style=dirt_color(indoor.dust_level)))
    table.add_row("Occupants",
                  f"{indoor.total_occupants} ({indoor.occupant_activity})")
    table.add_row("Comfort",
                  Text(f"{bar(indoor.comfort_score)} {indoor.comfort_score:.1f}%",
                       style=comfort_color(indoor.comfort_score)))
    table.add_row("Power",
                  f"{indoor.power_consumption_kw:.3f} kW  ({indoor.energy_today_kwh:.2f} kWh today)")

    return Panel(table, title="[bold cyan]🏠 Indoor Environment[/bold cyan]",
                 border_style="cyan")


def build_rooms_panel(indoor: IndoorState) -> Panel:
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("Room", style="bold", width=14)
    table.add_column("Temp °C", width=9)
    table.add_column("Lux", width=8)
    table.add_column("Occupied", width=10)

    room_icons = {
        "living_room": "🛋", "bedroom": "🛏", "kitchen": "🍳",
        "bathroom": "🚿", "hallway": "🚪", "office": "💻",
    }

    for room in sorted(ROOMS.keys()):
        icon = room_icons.get(room, "")
        rt = indoor.room_temps.get(room, 0.0)
        lux = indoor.lux_per_room.get(room, 0.0)
        occ = "✓ YES" if indoor.occupancy.get(room) else "  no"
        table.add_row(
            f"{icon} {room.replace('_', ' ').title()}",
            Text(f"{rt:.1f}", style=temp_color(rt)),
            f"{lux:.0f}",
            Text(occ, style="bright_green" if indoor.occupancy.get(room) else "dim"),
        )

    return Panel(table, title="[bold magenta]🏘 Room Status[/bold magenta]",
                 border_style="magenta")


def build_devices_panel(devices: DeviceState) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Device", style="bold", width=18)
    table.add_column("Status")

    hvac_label = HVAC_LABELS.get(devices.hvac_mode, "?")
    hvac_color = "red" if devices.hvac_mode in (2, 4) else ("yellow" if devices.hvac_mode in (1, 3) else "dim")

    table.add_row("🌡 HVAC", Text(hvac_label, style=hvac_color))
    table.add_row("💨 Ventilation", VENT_LABELS.get(devices.ventilation, "?"))
    table.add_row("🌬 Air Purifier", PURIFIER_LABELS.get(devices.air_purifier, "?"))
    table.add_row("💧 Humidifier", HUMIDIFIER_LABELS.get(devices.humidifier_mode, "?"))
    table.add_row("🤖 Cleaner Bot",
                  Text(CLEANER_LABELS.get(devices.cleaner_bot, "?"),
                       style="green" if devices.cleaner_bot == 1 else "dim"))
    table.add_row("🔒 Security",
                  Text(SECURITY_LABELS.get(devices.security, "?"),
                       style="bright_red" if devices.security else "dim"))

    # Lights
    lights_str = "  ".join(
        f"[bold]{r.replace('_',' ')[:3].title()}:[/bold]{v}"
        for r, v in sorted(devices.lights.items()) if v > 0
    ) or "[dim]all off[/dim]"
    table.add_row("💡 Lights", lights_str)

    return Panel(table, title="[bold green]⚡ Devices[/bold green]",
                 border_style="green")


def build_ml_panel(episode: int, step: int, reward: float,
                   epsilon: float, recent_losses: List[float]) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="bold", width=18)
    table.add_column("Value")

    table.add_row("Episode", str(episode))
    table.add_row("Step", f"{step} / 144  ({step * 10} min sim)")
    table.add_row("Last Reward", Text(f"{reward:+.4f}",
                                     style="green" if reward >= 0 else "red"))
    table.add_row("Epsilon (ε)", f"{bar(epsilon * 100)} {epsilon:.4f}")
    avg_loss = sum(recent_losses[-20:]) / max(1, len(recent_losses[-20:]))
    table.add_row("Avg Loss (20)", f"{avg_loss:.6f}")
    table.add_row("Replay Size",
                  f"{'see logs' if not recent_losses else f'{len(recent_losses)} updates'}")

    return Panel(table, title="[bold white]🧠 ML Agent[/bold white]",
                 border_style="white")


# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------

class LiveDashboard:
    """Displays real-time dashboard using Rich Live."""

    def __init__(self):
        self._indoor = IndoorState()
        self._weather = WeatherSnapshot(
            timestamp=datetime.datetime.now(),
            temperature_c=15.0, humidity_pct=60.0, wind_speed_kmh=10.0,
            cloud_cover_pct=50.0, precipitation_mm=0.0, uv_index=3.0,
            is_daytime=True, condition="cloudy"
        )
        self._devices = DeviceState()
        self._episode = 0
        self._step = 0
        self._reward = 0.0
        self._epsilon = 1.0
        self._losses: List[float] = []
        self._source = "Synthetic"
        self._lock = threading.Lock()

    def update(self, episode: int, step: int, indoor: IndoorState,
               devices: DeviceState, weather: WeatherSnapshot,
               reward: float, epsilon: float = 1.0, loss: Optional[float] = None):
        with self._lock:
            self._indoor = indoor
            self._devices = devices
            self._weather = weather
            self._episode = episode
            self._step = step
            self._reward = reward
            self._epsilon = epsilon
            if loss is not None:
                self._losses.append(loss)

    def build_layout(self) -> Layout:
        with self._lock:
            indoor = self._indoor
            weather = self._weather
            devices = self._devices

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="middle"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="weather"),
            Layout(name="ml"),
        )
        layout["middle"].split_column(
            Layout(name="indoor"),
            Layout(name="rooms"),
        )
        layout["right"].split_column(
            Layout(name="devices"),
        )

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text(
            f"  🏠 Smart House AI Control System   |   {ts}   |   Weather: {self._source}",
            style="bold white on dark_blue"
        )
        layout["header"].update(Panel(header_text, style="dark_blue"))

        layout["weather"].update(build_weather_panel(weather))
        layout["ml"].update(build_ml_panel(
            self._episode, self._step, self._reward, self._epsilon, self._losses
        ))
        layout["indoor"].update(build_indoor_panel(indoor))
        layout["rooms"].update(build_rooms_panel(indoor))
        layout["devices"].update(build_devices_panel(devices))

        comfort = indoor.comfort_score
        comfort_bar = bar(comfort, 100, 30)
        foot = Text(
            f"  Overall Comfort: {comfort_bar} {comfort:.1f}%   |   "
            f"Energy today: {indoor.energy_today_kwh:.2f} kWh   |   "
            f"[Q to quit]",
            style="bold"
        )
        layout["footer"].update(Panel(foot, style="dim"))

        return layout

    def run_with_trainer(self, trainer, n_episodes: int = 200):
        """Attach dashboard to a trainer and run."""

        last_loss = [None]

        def callback(ep, step, indoor, devices, weather, reward):
            loss = trainer.agent.losses[-1] if trainer.agent.losses else None
            self.update(ep, step, indoor, devices, weather, reward,
                        trainer.agent.epsilon, loss)
            self._source = trainer.weather_provider.source

        trainer.progress_callback = callback

        with Live(self.build_layout(), refresh_per_second=4, screen=True) as live:
            def train_thread():
                trainer.train(n_episodes=n_episodes, verbose=False)

            t = threading.Thread(target=train_thread, daemon=True)
            t.start()

            while t.is_alive():
                live.update(self.build_layout())
                time.sleep(0.25)

            live.update(self.build_layout())

        console.print("\n[bold green]Training complete![/bold green]")
