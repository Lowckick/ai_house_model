"""
Rule-based override layer.
Applies safety rules and smart heuristics ON TOP of the RL agent's decisions.
This prevents physically dangerous or obviously wrong actions.
"""

from smart_house.environment.indoor_state import DeviceState, IndoorState, ROOMS
from smart_house.environment.weather import WeatherSnapshot


class RuleLayer:
    """
    Post-processes the RL agent action with hard safety rules and
    intelligent per-room adjustments that the agent doesn't control directly.
    """

    # Safety thresholds
    MAX_CO2 = 1500.0           # ppm — force ventilation
    MIN_TEMP_SAFETY = 5.0      # °C — force heating
    MAX_TEMP_SAFETY = 32.0     # °C — force cooling
    HIGH_DIRT = 60.0           # % — force cleaner bot
    HIGH_PM25 = 55.0           # µg/m³ — force air purifier

    # Comfort targets
    IDEAL_TEMP = 22.0
    IDEAL_HUM_MIN = 40.0
    IDEAL_HUM_MAX = 60.0

    # Lux targets per room
    LUX_TARGETS = {
        "living_room": 300,
        "bedroom": 80,
        "kitchen": 500,
        "bathroom": 200,
        "hallway": 120,
        "office": 500,
    }
    SLEEP_LUX = 10              # near dark for sleeping

    def apply(self, devices: DeviceState, indoor: IndoorState,
              weather: WeatherSnapshot) -> DeviceState:
        """Return a (possibly modified) DeviceState after applying rules."""

        # Work on a copy
        d = DeviceState(
            hvac_mode=devices.hvac_mode,
            lights=dict(devices.lights),
            blinds=dict(devices.blinds),
            air_purifier=devices.air_purifier,
            humidifier_mode=devices.humidifier_mode,
            cleaner_bot=devices.cleaner_bot,
            ventilation=devices.ventilation,
            security=devices.security,
            curtains_closed=dict(devices.curtains_closed),
        )

        self._apply_safety_rules(d, indoor, weather)
        self._apply_smart_lights(d, indoor, weather)
        self._apply_smart_blinds(d, indoor, weather)
        self._apply_smart_security(d, indoor)
        self._apply_sleep_mode(d, indoor)

        return d

    # ------------------------------------------------------------------
    # Safety rules (hard overrides)
    # ------------------------------------------------------------------

    def _apply_safety_rules(self, d: DeviceState, indoor: IndoorState,
                             weather: WeatherSnapshot):
        # Dangerously cold → force heating
        if indoor.temperature_c < self.MIN_TEMP_SAFETY:
            d.hvac_mode = 2     # heating high

        # Dangerously hot → force cooling
        if indoor.temperature_c > self.MAX_TEMP_SAFETY:
            d.hvac_mode = 4     # cooling high

        # CO2 too high → force ventilation
        if indoor.co2_ppm > self.MAX_CO2:
            d.ventilation = max(d.ventilation, 2)

        # Very dirty → launch cleaner bot (if no one is home or sleeping)
        if indoor.dirt_level > self.HIGH_DIRT and indoor.occupant_activity != "sleeping":
            d.cleaner_bot = 1

        # High PM2.5 → boost air purifier
        if indoor.pm25_ugm3 > self.HIGH_PM25:
            d.air_purifier = max(d.air_purifier, 2)

        # Don't run cleaner while sleeping (noisy)
        if indoor.occupant_activity == "sleeping":
            d.cleaner_bot = 0

        # Humidity too high → dehumidify
        if indoor.humidity_pct > 75:
            d.humidifier_mode = 2
        elif indoor.humidity_pct < 25:
            d.humidifier_mode = 1

    # ------------------------------------------------------------------
    # Smart per-room lighting
    # ------------------------------------------------------------------

    def _apply_smart_lights(self, d: DeviceState, indoor: IndoorState,
                             weather: WeatherSnapshot):
        for room in ROOMS:
            occupied = indoor.occupancy.get(room, False)
            current_lux = indoor.lux_per_room.get(room, 0.0)

            if not occupied:
                d.lights[room] = 0
                continue

            # How much artificial light is needed on top of natural?
            target_lux = self.LUX_TARGETS.get(room, 300)
            if indoor.occupant_activity == "sleeping":
                target_lux = self.SLEEP_LUX

            deficit = max(0.0, target_lux - current_lux)
            # Each brightness unit ≈ 35 lux
            needed_brightness = min(10, int(deficit / 35.0) + 1) if deficit > 20 else 0
            d.lights[room] = needed_brightness

    # ------------------------------------------------------------------
    # Smart blinds (control solar gain + privacy)
    # ------------------------------------------------------------------

    def _apply_smart_blinds(self, d: DeviceState, indoor: IndoorState,
                             weather: WeatherSnapshot):
        for room in d.blinds:
            # Summer daytime: close blinds to avoid overheating
            if (indoor.temperature_c > 25 and weather.is_daytime
                    and weather.solar_irradiance > 400):
                d.blinds[room] = 20.0
            # Cold day: open to get solar heat
            elif indoor.temperature_c < 18 and weather.is_daytime:
                d.blinds[room] = 90.0
            # Night: close for privacy
            elif not weather.is_daytime:
                d.blinds[room] = 10.0
            else:
                d.blinds[room] = max(d.blinds.get(room, 50.0), 50.0)

        # Bedroom: close curtains during sleep
        if indoor.occupant_activity == "sleeping":
            d.curtains_closed["bedroom"] = True
            d.blinds["bedroom"] = 0.0
        else:
            d.curtains_closed["bedroom"] = False

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    def _apply_smart_security(self, d: DeviceState, indoor: IndoorState):
        if indoor.total_occupants == 0:
            d.security = 2       # armed away
        elif indoor.total_occupants > 0 and indoor.occupant_activity == "sleeping":
            d.security = 1       # armed home (perimeter only)
        else:
            d.security = 0       # disarmed

    # ------------------------------------------------------------------
    # Sleep mode
    # ------------------------------------------------------------------

    def _apply_sleep_mode(self, d: DeviceState, indoor: IndoorState):
        if indoor.occupant_activity == "sleeping":
            # Quieter ventilation
            d.ventilation = min(d.ventilation, 1)
            # Slightly cooler for sleep
            if indoor.temperature_c > 20:
                d.hvac_mode = 3   # cooling low → cool down gently
