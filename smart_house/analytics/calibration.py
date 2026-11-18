"""Estimate simulator parameters from recorded sensor readings.

Consecutive readings (2–30 min apart) give rates of change. Three robust linear fits:

* temperature: dT/dt = k · (T_out − T_in) + c      → heat_loss_per_h = k
* CO₂:         dC/dt = −a · (C − 400) + g · n + c0  → co2_infiltration_per_h = a,
                                                     co2_per_person_ppm_h = g (if occupants recorded)
* humidity:    dH/dt = e · (H_out − H) + c          → humidity_exchange_per_h = e

Intercepts absorb average internal gains (people, cooking, any heating that was running).
A slope is only accepted when the data cover a wide enough range and the fit is significant;
otherwise the simulator default is kept and the reason is reported. Heating or ventilation that
was switched on during the recording biases the estimates: prefer periods without them.
"""
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from smart_house.environment.indoor_state import ROOMS, PhysicsParams

MIN_STEP_MIN, MAX_STEP_MIN = 2.0, 30.0
MIN_PAIRS = 24
DEFAULT_HEAT_LOSS = (1.0 - sum(r['insulation'] for r in ROOMS.values()) / len(ROOMS)) * 0.4


def _robust_fit(X, y):
    """Least squares, then refit without residuals beyond 3.5 robust standard deviations."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    keep = np.ones(len(y), bool)
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        residual = y - X @ coef
        mad = np.median(np.abs(residual[keep] - np.median(residual[keep]))) * 1.4826
        if mad == 0:
            break
        keep = np.abs(residual) <= 3.5 * mad
    Xk, yk = X[keep], y[keep]
    coef, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
    residual = yk - Xk @ coef
    dof = max(1, len(yk) - X.shape[1])
    sigma2 = float(residual @ residual) / dof
    try:
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xk.T @ Xk)))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.inf)
    total = float(((yk - yk.mean()) ** 2).sum())
    r2 = 1.0 - float(residual @ residual) / total if total > 0 else 0.0
    return coef, se, r2, int(keep.sum()), int((~keep).sum())


def _pairs(rows):
    """(row, dt_hours, next_row) for consecutive readings with a usable interval."""
    rows = sorted(rows, key=lambda r: r['time'])
    for a, b in zip(rows, rows[1:]):
        minutes = (b['time'] - a['time']).total_seconds() / 60
        if MIN_STEP_MIN <= minutes <= MAX_STEP_MIN:
            yield a, minutes / 60, b


def _accept(name, coef, se, spread, min_spread, low, high):
    if spread < min_spread:
        return False, f'{name}: the driving difference varies too little (spread {spread:.2f} < {min_spread})'
    if not np.isfinite(se) or coef <= 0 or coef < 2 * se:
        return False, f'{name}: estimate {coef:.4f} ± {se:.4f} is not significantly positive'
    if not low <= coef <= high:
        return False, f'{name}: estimate {coef:.4f} is outside the plausible range {low}–{high}'
    return True, None


def calibrate(rows):
    """``rows``: dicts with time (aware datetime), indoor and outdoor values; returns a JSON-safe report."""
    pairs = list(_pairs(rows))
    report = {'readings': len(rows), 'pairs': len(pairs), 'min_pairs': MIN_PAIRS, 'fits': {}, 'warnings': []}
    params = {'heat_loss_per_h': None, 'co2_per_person_ppm_h': 20.0, 'co2_infiltration_per_h': 0.0,
              'humidity_exchange_per_h': 0.05}
    if len(pairs) < MIN_PAIRS:
        report['warnings'].append(f'Need at least {MIN_PAIRS} reading pairs 2–30 min apart; got {len(pairs)}. '
                                  'Defaults kept.')
        report['params'] = asdict(PhysicsParams(source='default (insufficient data)'))
        return report

    # Temperature
    x = np.array([a['outdoor_temperature_c'] - a['temperature_c'] for a, _, _ in pairs])
    y = np.array([(b['temperature_c'] - a['temperature_c']) / h for a, h, b in pairs])
    coef, se, r2, n, dropped = _robust_fit(np.column_stack([x, np.ones_like(x)]), y)
    ok, why = _accept('heat_loss_per_h', coef[0], se[0], float(np.std(x)), 2.0, 0.005, 1.0)
    report['fits']['temperature'] = {'slope': coef[0], 'slope_se': se[0], 'intercept_c_per_h': coef[1], 'r2': r2,
                                     'used': n, 'outliers': dropped, 'accepted': ok,
                                     'simulator_default': DEFAULT_HEAT_LOSS}
    if ok:
        params['heat_loss_per_h'] = float(coef[0])
    else:
        report['warnings'].append(why)

    # CO2
    excess = np.array([a['co2_ppm'] - 400.0 for a, _, _ in pairs])
    # Occupancy during the interval: the mean of both ends (people stay for a while in real homes).
    occupants = np.array([(a['total_occupants'] + b['total_occupants']) / 2 for a, _, b in pairs], float)
    y = np.array([(b['co2_ppm'] - a['co2_ppm']) / h for a, h, b in pairs])
    with_people = occupants.std() > 0.3
    X = np.column_stack([-excess, occupants, np.ones_like(excess)] if with_people else [-excess, np.ones_like(excess)])
    coef, se, r2, n, dropped = _robust_fit(X, y)
    ok, why = _accept('co2_infiltration_per_h', coef[0], se[0], float(np.std(excess)), 100.0, 0.01, 5.0)
    fit = {'infiltration': coef[0], 'infiltration_se': se[0], 'r2': r2, 'used': n, 'outliers': dropped,
           'accepted': ok, 'simulator_default_infiltration': 0.0}
    if ok:
        params['co2_infiltration_per_h'] = float(coef[0])
    else:
        report['warnings'].append(why)
    if with_people:
        fit.update({'per_person_ppm_h': coef[1], 'per_person_se': se[1], 'simulator_default_per_person': 20.0})
        person_ok, why = _accept('co2_per_person_ppm_h', coef[1], se[1], float(occupants.std()), 0.3, 1.0, 3000.0)
        fit['per_person_accepted'] = person_ok
        if person_ok:
            params['co2_per_person_ppm_h'] = float(coef[1])
        else:
            report['warnings'].append(why)
    else:
        fit['generation_ppm_h'] = coef[1]
        report['warnings'].append('co2_per_person_ppm_h: total_occupants was not recorded or constant; '
                                  'only total CO₂ generation is reported, the per-person default is kept.')
    report['fits']['co2'] = fit

    # Humidity
    x = np.array([a['outdoor_humidity_pct'] - a['humidity_pct'] for a, _, _ in pairs])
    y = np.array([(b['humidity_pct'] - a['humidity_pct']) / h for a, h, b in pairs])
    coef, se, r2, n, dropped = _robust_fit(np.column_stack([x, np.ones_like(x)]), y)
    ok, why = _accept('humidity_exchange_per_h', coef[0], se[0], float(np.std(x)), 3.0, 0.005, 3.0)
    report['fits']['humidity'] = {'slope': coef[0], 'slope_se': se[0], 'intercept_pct_per_h': coef[1], 'r2': r2,
                                  'used': n, 'outliers': dropped, 'accepted': ok, 'simulator_default': 0.05}
    if ok:
        params['humidity_exchange_per_h'] = float(coef[0])
    else:
        report['warnings'].append(why)

    report['fits'] = json.loads(json.dumps(report['fits'], default=float))
    report['params'] = asdict(PhysicsParams(**params, source='calibrated').validate())
    return report


def rows_from_records(records, weather):
    """Readings (live records or parsed IndoorStates) plus outdoor weather into calibration rows."""
    rows, stamps = [], []
    for record in records:
        indoor = record['indoor'] if isinstance(record, dict) else record.__dict__
        stamp = indoor['timestamp']
        stamp = dt.datetime.fromisoformat(stamp) if isinstance(stamp, str) else stamp
        stamp = stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)
        stamps.append(stamp)
        rows.append({'time': stamp, **{k: float(indoor[k]) for k in ('temperature_c', 'humidity_pct', 'co2_ppm')},
                     'total_occupants': float(indoor.get('total_occupants', 0))})
    if weather is not None:
        weather.prefetch(stamps)
    sources = []
    for row in rows:
        if weather is not None:
            snapshot, source = weather.get(row['time'])
        else:
            from smart_house.environment.weather_archive import fixed_weather
            snapshot, source = fixed_weather(row['time']), 'fixed'
        row['outdoor_temperature_c'], row['outdoor_humidity_pct'] = snapshot.temperature_c, snapshot.humidity_pct
        sources.append(source)
    return rows, {'open_meteo_readings': sources.count('open-meteo'), 'fixed_readings': sources.count('fixed')}


def load_params(path):
    """PhysicsParams from a calibration report (or a bare params object) written by ``calibrate``."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    data = data.get('params', data)
    allowed = {'heat_loss_per_h', 'co2_per_person_ppm_h', 'co2_infiltration_per_h', 'humidity_exchange_per_h', 'source'}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError('Unknown physics parameters: ' + ', '.join(sorted(unknown)))
    return PhysicsParams(**data).validate()
