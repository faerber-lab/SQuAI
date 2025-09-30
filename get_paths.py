import os
import sys
import subprocess
import re
from datetime import datetime, timedelta

def parse_remaining_time(remaining_str):
    """
    Wandelt Strings wie '99 days 22 hours' in timedelta um.
    """
    days_match = re.search(r"(\d+)\s+days?", remaining_str)
    hours_match = re.search(r"(\d+)\s+hours?", remaining_str)

    days = int(days_match.group(1)) if days_match else 0
    hours = int(hours_match.group(1)) if hours_match else 0

    return timedelta(days=days, hours=hours)

def get_ws_list_paths(min_days=8):
    """
    Ruft ws_list auf und gibt den Pfad des Workspaces mit der höchsten Nummer zurück,
    dessen Restlaufzeit mehr als min_days beträgt.
    """
    try:
        result = subprocess.run(
            ["ws_list"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
    except Exception as e:
        sys.stderr.write(f"Fehler beim Ausführen von ws_list: {e}\n")
        return None

    ws_entries = re.findall(
        r"^id:\s*(faiss(?:_\d+)?)\s*[\s\S]*?workspace directory\s*:\s*(\S+)\s*[\s\S]*?remaining time\s*:\s*(.*?)\n",
        output,
        re.MULTILINE
    )

    valid_workspaces = []

    for ws_name, ws_path, remaining_str in ws_entries:
        remaining = parse_remaining_time(remaining_str)
        if remaining > timedelta(days=min_days):
            # Extrahiere Zahl am Ende des Namens oder 0, wenn faiss
            number_match = re.search(r"faiss(?:_(\d+))?", ws_name)
            number = int(number_match.group(1)) if number_match and number_match.group(1) else 0
            valid_workspaces.append((number, ws_path))

    if not valid_workspaces:
        return None

    # Höchste Nummer auswählen
    valid_workspaces.sort(reverse=True, key=lambda x: x[0])
    return valid_workspaces[0][1]

def get_main_data_dir():
    # Erst ws_list versuchen
    ws_path = get_ws_list_paths(min_days=8)
    if ws_path and os.path.isdir(ws_path):
        return ws_path

    # Fallback wie bisher
    fallback_paths = [
        "/projects/p_scads_finetune/squai_faiss",
        "/data/horse/ws/inbe405h-unarxive",
    ]

    resolved_path = None

    for path in fallback_paths:
        # Prüfen, ob es eine "data_dir"-Datei ist
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    content = f.read().strip()
                if content and os.path.isdir(content):
                    resolved_path = content
                    break
            except (OSError, IOError) as e:
                sys.stderr.write(f"Fehler beim Lesen von {path}: {e}\n")
                continue

        # Direktes Verzeichnis prüfen
        if os.path.isdir(path):
            resolved_path = path
            break

    if resolved_path is None:
        sys.stderr.write(
            "Kein gültiges Datenverzeichnis gefunden. "
            f"Versuchte Pfade: {', '.join(fallback_paths)}\n"
        )
        sys.exit(1)

    return resolved_path

def get_bm25_python_path():
    # Definierter Pfad
    predefined_path = "/home/inbe405h/bm25_env/bin/python"
    if os.path.isfile(predefined_path):
        return predefined_path

    # Pfad im Home-Verzeichnis
    home_dir = os.path.expanduser("~")
    home_venv_path = os.path.join(home_dir, "bm25_env")
    python_path = os.path.join(home_venv_path, "bin", "python")

    marker_file = os.path.join(home_venv_path, ".installed")

    if os.path.isfile(python_path) and os.path.isfile(marker_file):
        # Venv existiert bereits und wurde installiert
        return str(python_path)

    # Falls venv noch nicht existiert, erstellen
    if not os.path.isdir(home_venv_path):
        os.makedirs(home_venv_path, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", home_venv_path], check=True)

    # Überprüfen ob requirements.txt existiert
    requirements_path = os.path.join("Retrieval_BM25", "requirements.txt")
    if not os.path.isfile(requirements_path):
        raise FileNotFoundError(f"Requirements-Datei nicht gefunden: {requirements_path}")

    # Pip innerhalb der venv upgraden und dependencies installieren, falls noch nicht installiert
    if not os.path.isfile(marker_file):
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([python_path, "-m", "pip", "install", "-r", requirements_path], check=True)
        # Marker-Datei erstellen
        with open(marker_file, "w") as f:
            f.write("installed\n")

    return str(python_path)
