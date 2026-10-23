import json
import tempfile
import unittest
from pathlib import Path
from smart_house.environment.readings import load_readings


class ReadingsTests(unittest.TestCase):
    def parse(self, content, suffix='.json'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ('readings' + suffix)
            path.write_text(content, encoding='utf-8')
            return load_readings(path)

    def test_formats_match(self):
        root = Path(__file__).resolve().parents[1] / 'examples'
        self.assertEqual(load_readings(root / 'readings.json'),
                         load_readings(root / 'readings.txt'))

    def test_invalid_measurements(self):
        base = dict(timestamp='2026-10-23T08:00:00', temperature_c=22, humidity_pct=50)
        for change in [dict(humidity_pct=101), dict(co2_ppm=-1),
                       dict(temperature_c=float('nan')), dict(total_occupants=1.5),
                       dict(temperature_c=True), dict(humidty_pct=50),
                       dict(timestamp='yesterday'), dict(co2_ppm=None)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.parse(json.dumps(base | change))

    def test_empty_missing_and_malformed(self):
        for content in ['[]', '{}', 'null', '[1]', '{']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.parse(content)

    def test_bad_csv(self):
        for text in ['timestamp,temperature_c,humidity_pct\nnow,22,50,extra',
                     'timestamp,temperature_c,temperature_c\na,1,2']:
            with self.assertRaises(ValueError):
                self.parse(text, '.txt')

    def test_defaults_and_presence(self):
        state = self.parse('{"timestamp":"2026-10-23T08:00:00","temperature_c":19,"humidity_pct":50,"total_occupants":2}')[0]
        self.assertTrue(all(state.occupancy.values()))
        self.assertTrue(all(t == 19 for t in state.room_temps.values()))
        self.assertEqual(state.co2_ppm, 450)


if __name__ == '__main__':
    unittest.main()
