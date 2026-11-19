"""Two-zone residential electricity tariff (day / night).

Assumption for the simulator: 4.32 UAH/kWh by day and the usual Ukrainian two-zone
coefficient 0.5 at night (23:00–07:00). Prices only scale results; the reward uses
relative weights normalised to a mean of 1 over the day, so the overall energy weight
of ``comfort_safety_tou_v3`` matches ``comfort_safety_v2``.
"""
import datetime as dt

DAY_PRICE_UAH = 4.32
NIGHT_COEFFICIENT = 0.5
NIGHT_START_HOUR, NIGHT_END_HOUR = 23, 7
NIGHT_HOURS = (24 - NIGHT_START_HOUR) + NIGHT_END_HOUR
MEAN_PRICE_UAH = DAY_PRICE_UAH * ((24 - NIGHT_HOURS) + NIGHT_HOURS * NIGHT_COEFFICIENT) / 24


def is_night(hour):
    """True during the night tariff zone (23:00–07:00)."""
    return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR


def price_uah_per_kwh(hour):
    """Electricity price for an hour of the day."""
    return DAY_PRICE_UAH * (NIGHT_COEFFICIENT if is_night(hour) else 1.0)


def price_weight(hour):
    """Relative price with a daily mean of 1 (day 1.2, night 0.6 for the default tariff)."""
    return price_uah_per_kwh(hour) / MEAN_PRICE_UAH


def interval_hour(end_time, minutes):
    """Hour in which a simulation interval ending at ``end_time`` started."""
    return (end_time - dt.timedelta(minutes=minutes)).hour
