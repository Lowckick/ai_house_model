"""Small local HTTP application; no third-party web dependency required."""
import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from smart_house.analytics.history import History, encode
from smart_house.analytics.ingestion import ImportService, ValidationError
from smart_house.analytics.lab import ModelLab, QueueFull
from smart_house.analytics.live import LiveService
from smart_house.environment.readings import MAX_FILE_BYTES, inspect_readings

LOGGER = logging.getLogger(__name__)
MAX_BODY = MAX_FILE_BYTES * 2  # JSON string escaping can enlarge uploaded UTF-8 text


def make_server(host, port, model, db, weather=None, live_token=None, notifier=None):
    """``live_token``: when set, sensors must send ``Authorization: Bearer <token>`` to POST /api/live/<id>."""
    service = ImportService(model, History(db), weather)
    lab = ModelLab(model, service.history)
    live = LiveService(service, notifier)
    page = Path(__file__).with_name('index.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        """HTTP routes of the panel, history, lab and live sensor API."""
        def log_message(self, format, *args):
            """Send access logs to the logging module instead of stderr."""
            LOGGER.info(format, *args)

        def respond(self, status, payload, content_type='application/json; charset=utf-8'):
            """Send JSON or HTML with security headers."""
            data = payload if isinstance(payload, bytes) else encode(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            """Page, history, lab and live sensor reads."""
            parts = urlsplit(self.path)
            route, query = parts.path, parse_qs(parts.query)
            try:
                if route == '/api/live':
                    return self.respond(200, {'sensors': live.sensors()})
                if route.startswith('/api/live/'):
                    sensor, _, tail = route[len('/api/live/'):].partition('/')
                    hours = int(query.get('hours', ['24'])[0])
                    if tail == 'report':
                        return self.respond(200, live.report(sensor, hours))
                    if tail == '':
                        import datetime as dt
                        since = live.clock() - dt.timedelta(hours=hours)
                        return self.respond(200, {'sensor_id': sensor, 'readings': live.readings(
                            sensor, since, int(query.get('limit', ['500'])[0]))})
                if route == '/':
                    return self.respond(200, page, 'text/html; charset=utf-8')
                if route == '/api/lab/models':
                    return self.respond(200, {'models': lab.catalog()})
                if route == '/api/lab/stats':
                    return self.respond(200, lab.stats())
                if route == '/api/lab/jobs':
                    return self.respond(200, {'jobs': lab.jobs()})
                if route.startswith('/api/lab/jobs/'):
                    return self.respond(200, lab.job(route.rsplit('/', 1)[-1]))
                if route == '/api/history':
                    return self.respond(200, {'imports': service.history.list_imports(),
                                              'evaluations': service.history.list_evaluations()})
                for prefix, read in [('/api/imports/', service.history.get_import),
                                     ('/api/evaluations/', service.history.get_evaluation)]:
                    if route.startswith(prefix) and route[len(prefix):].isdigit():
                        return self.respond(200, read(int(route[len(prefix):])))
                self.respond(404, {'error': 'Not found'})
            except KeyError:
                self.respond(404, {'error': 'Not found'})
            except ValueError as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                LOGGER.exception('History request failed')
                self.respond(500, {'error': 'Не вдалося прочитати історію. Перевірте журнал сервера.'})

        def do_POST(self):
            # Browser requests must originate from this application; JSON prevents form POSTs.
            """Validation, imports, evaluations, lab requests and live readings (JSON only, same origin)."""
            if (self.headers.get('Origin') not in (None, 'http://' + self.headers.get('Host', ''))
                    or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
                return self.respond(403, {'error': 'Cross-origin request rejected'})
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                return self.respond(415, {'error': 'Content-Type must be application/json'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BODY:
                    return self.respond(413, {'error': 'Request exceeds size limit or is empty'})
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object')
                route = urlsplit(self.path).path
                if route.startswith('/api/live/'):
                    if live_token and not hmac.compare_digest(self.headers.get('Authorization', ''),
                                                              'Bearer ' + live_token):
                        return self.respond(401, {'error': 'Missing or invalid sensor token'})
                    return self.respond(201, live.ingest(route[len('/api/live/'):], body))
                if route == '/api/lab/predict':
                    return self.respond(200, lab.predict(body))
                if route == '/api/lab/jobs':
                    return self.respond(202, lab.submit(body))
                if route == '/api/lab/rating':
                    return self.respond(200, lab.rate(body.get('run_id'), body.get('rating')))
                if route in ('/api/validate', '/api/import'):
                    filename, text = body.get('filename'), body.get('text')
                    if not isinstance(filename, str) or not isinstance(text, str):
                        raise ValueError('filename and text must be strings')
                    if route == '/api/validate':
                        _, report = inspect_readings(text, Path(filename).suffix)
                        return self.respond(200, report)
                    return self.respond(201, service.import_text(filename, text))
                if route == '/api/evaluate':
                    values = [body.get('episodes', 5), body.get('steps', 144), body.get('seed', 42)]
                    if any(type(v) is not int for v in values):
                        raise ValueError('episodes, steps and seed must be integers')
                    return self.respond(201, service.evaluate(*values))
                self.respond(404, {'error': 'Not found'})
            except ValidationError as exc:
                self.respond(422, {'error': str(exc), 'validation': exc.report})
            except QueueFull as exc:
                self.respond(429, {'error': str(exc)})
            except KeyError:
                self.respond(404, {'error': 'Not found'})
            except (ValueError, UnicodeError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                LOGGER.exception('Import/evaluation request failed')
                self.respond(500, {'error': 'Не вдалося обробити запит. Перевірте модель і журнал сервера.'})

    class LabServer(ThreadingHTTPServer):
        """Threaded server that also stops the lab worker when closed."""
        def server_close(self):
            """Close the socket, then wait for and stop the lab worker."""
            super().server_close()
            lab.close()

    try:
        server = LabServer((host, port), Handler)
    except Exception:
        lab.close()
        raise
    server.daemon_threads = True
    return server


def notifier_from_env():
    """Telegram alerts when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set; otherwise none."""
    import os

    from smart_house.analytics.live import telegram_notifier
    token, chat = os.environ.get('TELEGRAM_BOT_TOKEN'), os.environ.get('TELEGRAM_CHAT_ID')
    return telegram_notifier(token, chat) if token and chat else None


def serve(host, port, model, db, weather=None):
    """Run the web panel until interrupted; a token is required when listening beyond localhost."""
    import ipaddress
    import os

    import torch
    torch.set_num_threads(2)
    token = os.environ.get('SMART_HOUSE_LIVE_TOKEN') or None
    try:
        loopback = host == 'localhost' or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not token:
        raise ValueError('Listening beyond this computer lets sensors on the network write readings: '
                         'set SMART_HOUSE_LIVE_TOKEN first (sensors send "Authorization: Bearer <token>")')
    server = make_server(host, port, model, db, weather, token, notifier_from_env())
    print(f'Smart House web listening on {host}:{server.server_port}. Stop with Ctrl-C.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
