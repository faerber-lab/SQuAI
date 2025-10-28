#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
import threading
import subprocess
import configparser
import html

BACKEND_URL = "http://localhost:8500"
PORT = 8080
CONFIG_FILE = "defaults.ini"

def read_hpc_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    hpc_config = {}
    if "hpc" in config:
        hpc_config["username"] = config["hpc"].get("username", "")
        hpc_config["partition"] = config["hpc"].get("partition", "")
    return hpc_config

def ssh_command(username, cmd):
    """Führe SSH-Befehl aus, gib stdout und Fehler zurück"""
    full_cmd = ["ssh", f"{username}@login1.capella.hpc.tu-dresden.de", cmd]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None, result.stderr.strip()
        return result.stdout.strip(), None
    except Exception as e:
        return None, str(e)

def generate_hpc_status_html(username, partition):
    """Erzeuge HTML mit HPC-Status"""
    stdout, err = ssh_command(username, "squeue --me -o '%i %T %j %u %P %M %D %R'")
    html_parts = []

    html_parts.append("""
<html>
<head>
<title>HPC Status</title>
<style>
body { background-color: #1e1e1e; color: #c0c0c0; font-family: monospace; padding: 20px; }
h1, h2 { color: #ffffff; }
pre { background-color: #2e2e2e; padding: 10px; border-radius: 6px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #444; padding: 6px; text-align: left; }
th { background-color: #333; }
tr:nth-child(even) { background-color: #2a2a2a; }
</style>
</head>
<body>
<h1>HPC Job Status</h1>
""")

    if err:
        html_parts.append(f"<p style='color:red;'>SSH connection didn't work: {html.escape(err)}</p>")
        html_parts.append("</body></html>")
        return "".join(html_parts)

    if not stdout:
        html_parts.append("<p>No jobs found for user.</p>")
        html_parts.append("</body></html>")
        return "".join(html_parts)

    jobs = []
    for line in stdout.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        job = {
            "JOBID": parts[0],
            "STATE": parts[1],
            "NAME": parts[2],
            "USER": parts[3],
            "PARTITION": parts[4],
            "TIME": parts[5],
            "NODES": parts[6],
            "NODELIST(REASON)": parts[7],
        }
        jobs.append(job)

    # Prüfe, ob ein Job running ist
    any_running = any(j["STATE"] == "R" for j in jobs)

    # Tabelle aller Jobs
    html_parts.append("<table><tr><th>JobID</th><th>State</th><th>Name</th><th>Partition</th><th>Time</th><th>Nodes</th><th>NodeList/Reason</th></tr>")
    for j in jobs:
        html_parts.append(f"<tr><td>{html.escape(j['JOBID'])}</td><td>{html.escape(j['STATE'])}</td><td>{html.escape(j['NAME'])}</td><td>{html.escape(j['PARTITION'])}</td><td>{html.escape(j['TIME'])}</td><td>{html.escape(j['NODES'])}</td><td>{html.escape(j['NODELIST(REASON)'])}</td></tr>")
    html_parts.append("</table>")

    if any_running:
        html_parts.append("<p style='color:yellow;'>At least one job is currently running but not reachable.</p>")
    else:
        # Für alle pending jobs whypending ausführen
        for j in jobs:
            if j["STATE"] == "PD":
                stdout, err = ssh_command(username, f"whypending {j['JOBID']}")
                html_parts.append(f"<h2>Job {j['JOBID']} Pending Info:</h2>")
                if err:
                    html_parts.append(f"<p style='color:red;'>Error executing whypending: {html.escape(err)}</p>")
                else:
                    html_parts.append(f"<pre>{html.escape(stdout)}</pre>")

    html_parts.append("</body></html>")
    return "".join(html_parts)


class ProxyHandler(BaseHTTPRequestHandler):
    def _proxy_request(self):
        """Versuch, Backend zu erreichen, sonst HPC Status anzeigen"""
        try:
            backend_resp = requests.get(f"{BACKEND_URL}{self.path}", timeout=2)
            self.send_response(backend_resp.status_code)
            for k, v in backend_resp.headers.items():
                if k.lower() != "transfer-encoding":
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(backend_resp.content)
        except requests.exceptions.RequestException:
            # Backend offline → HPC Status
            hpc_config = read_hpc_config()
            html_content = generate_hpc_status_html(hpc_config.get("username", ""), hpc_config.get("partition", ""))
            self.send_response(503)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

    def do_GET(self):
        self._proxy_request()

    def do_POST(self):
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
            hpc_config = read_hpc_config()
            html_content = generate_hpc_status_html(hpc_config.get("username", ""), hpc_config.get("partition", ""))
            self.send_response(503)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))


def run_server():
    server = HTTPServer(('127.0.0.1', PORT), ProxyHandler)
    print(f"Smart proxy running on http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    server_thread.join()
