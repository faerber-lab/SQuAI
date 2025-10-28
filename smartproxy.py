#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
import threading
import subprocess
import configparser
import html
import re

BACKEND_URL = "http://localhost:8500"
PORT = 8080
CONFIG_FILE = "defaults.ini"

def ansi_to_html(text):
    """
    Wandelt ANSI-Farb-Codes in HTML <span> mit Farben um.
    Unterstützt Standardfarben und fett (bold).
    """
    # ANSI color map
    ansi_colors = {
        '30': 'black',
        '31': 'red',
        '32': 'green',
        '33': 'yellow',
        '34': 'blue',
        '35': 'magenta',
        '36': 'cyan',
        '37': 'white',
        '90': 'grey',  # hellschwarz
        '91': 'red',
        '92': 'green',
        '93': 'yellow',
        '94': 'blue',
        '95': 'magenta',
        '96': 'cyan',
        '97': 'white',
    }

    # ANSI regex: \x1b[<codes>m
    ansi_regex = re.compile(r'\x1b\[([0-9;]+)m')

    html_parts = []
    open_tags = []

    last_end = 0
    for match in ansi_regex.finditer(text):
        start, end = match.span()
        codes = match.group(1).split(';')

        # Text zwischen ANSI-Codes übernehmen
        html_parts.append(text[last_end:start])
        last_end = end

        # Reset (0)
        if '0' in codes:
            while open_tags:
                html_parts.append('</span>')
                open_tags.pop()
            codes.remove('0')

        # bold (1)
        style = ''
        if '1' in codes:
            style += 'font-weight:bold;'
            codes.remove('1')

        # Farben
        for code in codes:
            color = ansi_colors.get(code)
            if color:
                style += f'color:{color};'

        if style:
            html_parts.append(f'<span style="{style}">')
            open_tags.append('</span>')

    # Restlichen Text
    html_parts.append(text[last_end:])
    while open_tags:
        html_parts.append(open_tags.pop())

    return ''.join(html_parts)

def read_hpc_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    hpc_config = {}
    if "hpc" in config:
        hpc_config["username"] = config["hpc"].get("username", "").strip()
        hpc_config["partition"] = config["hpc"].get("partition", "").strip()
    return hpc_config

def ssh_command(username, cmd, timeout=10):
    """
    Run an SSH command and return (stdout, error). Error is None on success.
    """
    full_cmd = ["ssh", f"{username}@login1.capella.hpc.tu-dresden.de", cmd]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            # prefer stderr, fallback to a generic message
            return None, result.stderr.strip() or f"Command returned code {result.returncode}"
        return result.stdout.strip(), None
    except subprocess.TimeoutExpired as e:
        return None, f"SSH command timed out after {timeout}s"
    except Exception as e:
        return None, str(e)

def parse_whypending(stdout):
    """
    Extract a few key pieces from whypending output:
      - Reason paragraph (first paragraph that starts with 'Reason' or first non-empty paragraph)
      - Position in queue
      - Estimated start time
    Returns a dict with keys: reason, position, estimated_start, full_text
    """
    result = {"reason": None, "position": None, "estimated_start": None, "full_text": stdout or ""}
    if not stdout:
        return result

    lines = stdout.splitlines()
    # Join into paragraphs (separated by blank lines)
    paragraphs = []
    cur = []
    for ln in lines:
        if ln.strip() == "":
            if cur:
                paragraphs.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(ln.rstrip())
    if cur:
        paragraphs.append("\n".join(cur).strip())

    # Find paragraph starting with Reason (case-insensitive) first
    reason_par = None
    for p in paragraphs:
        if p.lower().startswith("reason"):
            reason_par = p
            break
    if not reason_par and paragraphs:
        # fallback to first paragraph
        reason_par = paragraphs[0]

    if reason_par:
        result["reason"] = reason_par

    # Position in queue
    m = re.search(r"Position in queue:\s*(\d+)", stdout, flags=re.IGNORECASE)
    if m:
        result["position"] = m.group(1)

    # Estimated start time
    m2 = re.search(r"Estimated start time:\s*(.*)", stdout, flags=re.IGNORECASE)
    if m2:
        est = m2.group(1).strip()
        result["estimated_start"] = est if est else None

    return result

def generate_hpc_status_html(username, partition):
    """
    Generate a dark-mode HTML page with clear, English messages:
      - SSH connection errors
      - No jobs
      - For each job: clear status (running / pending / other)
      - For pending jobs: run whypending and show parsed info + full output
    The internal squeue table is NOT shown to the user.
    """
    html_parts = []
    html_parts.append("""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HPC Status</title>
<style>
body { background-color: #0f1113; color: #d7d7d7; font-family: Inter, Roboto, "DejaVu Sans", monospace; padding: 24px; }
.container { max-width: 980px; margin: 0 auto; }
h1 { color: #ffffff; margin-bottom: 8px; }
.card { background: #111316; border: 1px solid #222427; border-radius: 8px; padding: 14px; margin-bottom: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.5); }
.header-line { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.status-error { color: #ff6b6b; font-weight:700; }
.status-warn { color: #ffcc66; font-weight:700; }
.status-ok { color: #6be26b; font-weight:700; }
.job-id { font-weight:700; color:#fff; }
.small { font-size:0.92rem; color:#9aa0a6; }
pre { background-color: #0b0d0e; color: #d6d6d6; padding: 12px; border-radius: 6px; overflow-x:auto; border:1px solid #1f2526; }
.kv { margin:6px 0; }
.k { color:#9fb3ff; font-weight:700; margin-right:8px; }
.v { color:#d7d7d7; }
.note { color:#aab2b6; font-size:0.92rem; margin-top:8px; }
</style>
</head>
<body>
<div class="container">
<h1>HPC Job Status</h1>
""")

    # Get jobs via squeue (only JOBID and STATE), no header (-h)
    stdout, err = ssh_command(username, "squeue --me -h -o '%i %T'")
    if err:
        html_parts.append('<div class="card"><div class="header-line"><div class="status-error">SSH connection didn\'t work</div></div>')
        html_parts.append(f'<div class="note">Detailed error: {html.escape(err)}</div>')
        html_parts.append("</div></div></body></html>")
        return "".join(html_parts)

    if not stdout.strip():
        html_parts.append('<div class="card"><div class="header-line"><div class="status-warn">No jobs found for user</div></div>')
        html_parts.append('<div class="note">You currently have no jobs visible to <code>squeue --me</code>.</div>')
        html_parts.append("</div></div></body></html>")
        return "".join(html_parts)

    # Parse jobs list: each line -> JOBID STATE...
    jobs = []
    for line in stdout.splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split(None, 1)
        if len(parts) == 0:
            continue
        jobid = parts[0]
        state_raw = parts[1] if len(parts) > 1 else ""
        # Normalize: take first token of state (in case slurm adds flags), uppercase
        state_token = state_raw.split()[0].upper() if state_raw else ""
        jobs.append({"JOBID": jobid, "STATE_RAW": state_raw, "STATE": state_token})

    # If any job is running (STATE startswith 'R'), show running message and stop (as requested)
    any_running = any(j["STATE"].startswith("R") for j in jobs)
    if any_running:
        # Find first running job(s) and list them
        running_jobs = [j for j in jobs if j["STATE"].startswith("R")]
        html_parts.append('<div class="card">')
        html_parts.append('<div class="header-line"><div class="status-warn">At least one job is currently running but not reachable</div></div>')
        html_parts.append('<div class="note">The following running job(s) were detected. The proxy will not attempt <code>whypending</code> when jobs are running.</div>')
        for r in running_jobs:
            html_parts.append(f'<div class="kv"><span class="k">Job</span><span class="job-id">{html.escape(r["JOBID"])}</span> <span class="small">state={html.escape(r["STATE_RAW"])}</span></div>')
        html_parts.append("</div></div></body></html>")
        return "".join(html_parts)

    # No running jobs -> handle pending and other states
    pending_jobs = [j for j in jobs if j["STATE"].startswith("PENDING")]
    other_jobs = [j for j in jobs if not (j["STATE"].startswith("PENDING") or j["STATE"].startswith("R"))]

    if pending_jobs:
        # For each pending job, run whypending and present parsed info + full output
        for j in pending_jobs:
            jid = j["JOBID"]
            html_parts.append('<div class="card">')
            html_parts.append(f'<div class="header-line"><div class="status-warn">Job {html.escape(jid)} is pending</div></div>')
            # Run whypending with a bit more timeout
            wp_stdout, wp_err = ssh_command(username, f"whypending {jid}", timeout=20)
            if wp_err:
                html_parts.append(f'<div class="note status-error">Error executing <code>whypending {html.escape(jid)}</code>: {html.escape(wp_err)}</div>')
                html_parts.append("</div>")
                continue
            parsed = parse_whypending(wp_stdout)
            # Show extracted fields if available
            if parsed.get("position"):
                html_parts.append(f'<div class="kv"><span class="k">Position in queue:</span><span class="v">{ansi_to_html(html.escape(parsed["position"]))}</span></div>')
            if parsed.get("estimated_start"):
                html_parts.append(f'<div class="kv"><span class="k">Estimated start time:</span><span class="v">{ansi_to_html(html.escape(parsed["estimated_start"]))}</span></div>')
            if parsed.get("reason"):
                html_parts.append(f'<div class="kv"><span class="k">Reason:</span><span class="v">{ansi_to_html(html.escape(parsed["reason"].splitlines()[0]))}</span></div>')
                # Show reason paragraph in a small pre for detail
                html_parts.append(f'<div class="note">Reason details below.</div>')
            # Full whypending output for completeness
            html_parts.append(f'<pre>{ansi_to_html(html.escape(wp_stdout))}</pre>')
            html_parts.append("</div>")
    else:
        # No pending jobs found
        html_parts.append('<div class="card"><div class="header-line"><div class="status-ok">No pending jobs</div></div>')
        html_parts.append("</div>")

    # Include other jobs (if any) as informational
    if other_jobs:
        html_parts.append('<div class="card">')
        html_parts.append('<div class="header-line"><div class="small">Other job states detected:</div></div>')
        for o in other_jobs:
            html_parts.append(f'<div class="kv"><span class="k">Job</span><span class="job-id">{html.escape(o["JOBID"])}</span> <span class="small">state={html.escape(o["STATE_RAW"])}</span></div>')
        html_parts.append("</div>")

    html_parts.append("</div></body></html>")
    return "".join(html_parts)

class ProxyHandler(BaseHTTPRequestHandler):
    def _proxy_request(self):
        try:
            backend_resp = requests.get(f"{BACKEND_URL}{self.path}", timeout=2)
            self.send_response(backend_resp.status_code)
            # copy headers except transfer-encoding
            for k, v in backend_resp.headers.items():
                if k.lower() != "transfer-encoding":
                    self.send_header(k, v)
            self.end_headers()
            # write body
            if backend_resp.content:
                try:
                    self.wfile.write(backend_resp.content)
                except BrokenPipeError:
                    # client disconnected
                    pass
        except requests.exceptions.RequestException:
            hpc_config = read_hpc_config()
            username = hpc_config.get("username", "")
            partition = hpc_config.get("partition", "")
            html_content = generate_hpc_status_html(username, partition)
            self.send_response(503)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content.encode("utf-8"))))
            self.end_headers()
            try:
                self.wfile.write(html_content.encode("utf-8"))
            except BrokenPipeError:
                pass

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
            if backend_resp.content:
                try:
                    self.wfile.write(backend_resp.content)
                except BrokenPipeError:
                    pass
        except requests.exceptions.RequestException:
            hpc_config = read_hpc_config()
            username = hpc_config.get("username", "")
            partition = hpc_config.get("partition", "")
            html_content = generate_hpc_status_html(username, partition)
            self.send_response(503)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content.encode("utf-8"))))
            self.end_headers()
            try:
                self.wfile.write(html_content.encode("utf-8"))
            except BrokenPipeError:
                pass

def run_server():
    server = HTTPServer(('127.0.0.1', PORT), ProxyHandler)
    print(f"Smart proxy running on http://127.0.0.1:{PORT}")
    server.serve_forever()

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    server_thread.join()
