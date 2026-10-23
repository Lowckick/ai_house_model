"""Small local HTTP application; no third-party web dependency required."""
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from smart_house.analytics.history import History, encode
from smart_house.analytics.ingestion import ImportService, ValidationError
from smart_house.environment.readings import MAX_FILE_BYTES, inspect_readings

LOGGER = logging.getLogger(__name__)
MAX_BODY = MAX_FILE_BYTES * 2  # JSON string escaping can enlarge uploaded UTF-8 text


def make_server(host, port, model, db):
    service = ImportService(model, History(db))
    page = Path(__file__).with_name('index.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            LOGGER.info(format, *args)

        def respond(self, status, payload, content_type='application/json; charset=utf-8'):
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
            route = urlsplit(self.path).path
            try:
                if route == '/':
                    return self.respond(200, page, 'text/html; charset=utf-8')
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
            except Exception:
                LOGGER.exception('History request failed')
                self.respond(500, {'error': 'Не вдалося прочитати історію. Перевірте журнал сервера.'})

        def do_POST(self):
            # Browser requests must originate from this application; JSON prevents form POSTs.
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
            except (ValueError, UnicodeError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                LOGGER.exception('Import/evaluation request failed')
                self.respond(500, {'error': 'Не вдалося обробити запит. Перевірте модель і журнал сервера.'})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(host, port, model, db):
    import torch
    torch.set_num_threads(2)
    server = make_server(host, port, model, db)
    print(f'Smart House web listening on {host}:{server.server_port}. Stop with Ctrl-C.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
