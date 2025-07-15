#!/usr/bin/env python3
"""
Very small HTTP server that shows the machine’s hostname, greets the user,
and shows GPU info if available via nvidia-smi.

Start the server with:
    python3 hpc.py
Then visit the printed URL or check ~/hpc_server_host_and_file.
"""

import socket
import os
import subprocess
from flask import Flask, request
from werkzeug.serving import make_server

app = Flask(__name__)
HOSTNAME = socket.gethostname()


def get_nvidia_smi_output():
    """Return the output of `nvidia-smi` if available, otherwise None."""
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return completed.stdout
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as e:
        return f"nvidia-smi failed:\n{e.stderr}"


@app.route("/", methods=["GET", "POST"])
def index():
    """Root page — shows hostname, optional greeting, and GPU info."""
    name = request.form.get("name")
    if name:
        greeting = f"Hi {name}. This script runs on {HOSTNAME}."
    else:
        greeting = f"This script runs on {HOSTNAME}."

    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        greeting += f" It is running as SLURM job ID {slurm_job_id}."

    nvidia_output = get_nvidia_smi_output()
    if nvidia_output is not None:
        gpu_info = f"<h2>GPU Info (from nvidia-smi)</h2><pre>{nvidia_output}</pre>"
    else:
        gpu_info = "<h2>No GPU found or nvidia-smi not available.</h2>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>HPC Server</title></head>
<body style="font-family:sans-serif">
  <h1>{greeting}</h1>
  <form method="post">
    <label for="name">Your name:</label>
    <input id="name" name="name" type="text" placeholder="Enter your name" required>
    <button type="submit">Send</button>
  </form>
  {gpu_info}
</body>
</html>"""


def find_free_port():
    """Find a free port by binding to port 0 and letting the OS choose."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_sock:
        temp_sock.bind(('', 0))
        return temp_sock.getsockname()[1]


def write_host_and_port_file(host, port):
    """Write hostname:port to ~/hpc_server_host_and_file."""
    filepath = os.path.expanduser("~/hpc_server_host_and_file")
    try:
        with open(filepath, "w") as f:
            f.write(f"{host}:{port}\n")
    except OSError as e:
        print(f"Failed to write {filepath}: {e}")


if __name__ == "__main__":
    port = find_free_port()
    write_host_and_port_file(HOSTNAME, port)
    print(f"Server running at http://{HOSTNAME}:{port}/")
    try:
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Server failed to start: {e}")
