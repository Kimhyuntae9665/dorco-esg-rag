import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit


class LocalHandler(BaseHTTPRequestHandler):
    max_body = 16384

    def log_message(self, *_):
        pass  # Do not persist document context or user questions in access logs.

    def allowed(self):
        port = self.server.server_address[1]
        if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
            self.send_json({'error': 'Invalid Host'}, 403)
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
            self.send_json({'error': 'Cross-origin request refused'}, 403)
            return False
        return True

    def body(self):
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            raise ValueError('Expected application/json')
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= self.max_body:
            raise ValueError('Request length must be 1..16384 bytes')
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError('Expected JSON object')
        return value

    def send_bytes(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8', status)


def confined_file(root, request_path):
    relative = urlsplit(request_path).path.lstrip('/') or 'index.html'
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None
    return path
