#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
import threading

BACKEND_URL = "http://localhost:8500"
PORT = 8080

class ProxyHandler(BaseHTTPRequestHandler):
    def _proxy_request(self):
        """Versuch, Backend zu erreichen, sonst 503 zurückgeben"""
        try:
            # Backend anfragen
            backend_resp = requests.get(f"{BACKEND_URL}{self.path}", timeout=2)
            
            # Statuscode weitergeben
            self.send_response(backend_resp.status_code)

            # Header weitergeben, außer Transfer-Encoding (Chunked)
            for k, v in backend_resp.headers.items():
                if k.lower() != "transfer-encoding":
                    self.send_header(k, v)
            self.end_headers()

            # Content zurückgeben
            self.wfile.write(backend_resp.content)

        except requests.exceptions.RequestException:
            # Backend offline → 503-Seite
            self.send_response(503)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

            html = """
<html>
<head><title>503 Service Temporarily Unavailable</title></head>
<body>
<h1>503 Service Temporarily Unavailable</h1>
<p>Das HPC-Backend ist momentan offline. Bitte versuchen Sie es später erneut.</p>
<p>Python sagt hallo!</p>
</body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))

    def do_GET(self):
        self._proxy_request()

    def do_POST(self):
        # Optional: POST-Requests auch an Backend weiterleiten
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else None

            backend_resp = requests.post(f"{BACKEND_URL}{self.path}", data=post_data, headers=self.headers, timeout=5)

            self.send_response(backend_resp.status_code)
            for k, v in backend_resp.headers.items():
                if k.lower() != "transfer-encoding":
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(backend_resp.content)

        except requests.exceptions.RequestException:
            self.send_response(503)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

            html = """
<html>
<head><title>503 Service Temporarily Unavailable</title></head>
<body>
<h1>503 Service Temporarily Unavailable</h1>
<p>Das HPC-Backend ist momentan offline. Bitte versuchen Sie es später erneut.</p>
<p>Python sagt hallo!</p>
</body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))

def run_server():
    server = HTTPServer(('127.0.0.1', PORT), ProxyHandler)
    print(f"Smart proxy running on http://127.0.0.1:{PORT}")
    server.serve_forever()

if __name__ == "__main__":
    # Server in eigenem Thread starten, falls später weitere Tasks nötig sind
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    server_thread.join()
