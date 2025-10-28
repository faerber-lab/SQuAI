#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests

BACKEND_URL = "http://localhost:8500"
PORT = 8080

class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Versuch, das Backend zu erreichen
            r = requests.get(f"{BACKEND_URL}{self.path}", timeout=2)
            self.send_response(r.status_code)
            for k, v in r.headers.items():
                if k.lower() != "transfer-encoding":
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(r.content)
        except requests.exceptions.RequestException:
            # Backend offline → 503 Seite
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
            self.wfile.write(html.encode("utf-8"))  # <--- WICHTIG: hier wird der String zu Bytes

if __name__ == "__main__":
    server = HTTPServer(('127.0.0.1', PORT), ProxyHandler)
    print(f"Smart proxy running on http://127.0.0.1:{PORT}")
    server.serve_forever()
