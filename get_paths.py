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

def is_readable_directory(path):
    """
    Check if the given path is a readable directory.

    Args:
        path (str): Directory path to check.

    Returns:
        bool: True if the path is a readable directory, False otherwise.
    """
    try:
        if not os.path.exists(path):
            print(f"Error: Path does not exist -> {path}", file=sys.stderr)
            return False
        if not os.path.isdir(path):
            print(f"Error: Path is not a directory -> {path}", file=sys.stderr)
            return False
        if not os.access(path, os.R_OK):
            print(f"Error: Directory is not readable -> {path}", file=sys.stderr)
            return False
        try:
            # Attempt to list the directory to confirm readability
            _ = os.listdir(path)
        except Exception as e:
            print(f"Error: Could not list directory contents -> {path}\nReason: {e}", file=sys.stderr)
            return False

        return True

    except Exception as e:
        print(f"Unexpected error while checking directory -> {path}\nReason: {e}", file=sys.stderr)
        return False

def get_ws_list_paths(min_days=8):
    """
    Ruft ws_list auf und gibt den Pfad des Workspaces mit der höchsten Nummer zurück,
    dessen Restlaufzeit mehr als min_days beträgt.
    """

    directory_path = "/data/horse/ws/s3811141-faiss/inbe405h-unarxive"
    if is_readable_directory(directory_path):
        return directory_path

    print("Trying to look for workspace...")
    try:
        result = subprocess.run(
            ["ws_list"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout

        print(f"ws_list result: {result}")
    except Exception as e:
        sys.stderr.write(f"Fehler beim Ausführen von ws_list: {e}\n")
        return None

    ws_entries = re.findall(
        r"^id:\s*(faiss(?:_\d+)?)\s*[\s\S]*?workspace directory\s*:\s*(\S+)\s*[\s\S]*?remaining time\s*:\s*(.*?)\n",
        output,
        re.MULTILINE
    )

    print(f"ws_entries: {ws_entries}")

    valid_workspaces = []

    for ws_name, ws_path, remaining_str in ws_entries:
        remaining = parse_remaining_time(remaining_str)
        print(f"Workspace: {ws_name} ({ws_path}, remaining: {remaining})")
        if remaining > timedelta(days=min_days):
            # Extrahiere Zahl am Ende des Namens oder 0, wenn faiss
            number_match = re.search(r"faiss(?:_(\d+))?", ws_name)
            number = int(number_match.group(1)) if number_match and number_match.group(1) else 0
            valid_workspaces.append((number, ws_path))

    if not valid_workspaces:
        print("No valid workspace found")
        return None

    # Höchste Nummer auswählen
    valid_workspaces.sort(reverse=True, key=lambda x: x[0])
    print(f"Using found valid workspace: {valid_workspaces[0][1]}")
    return valid_workspaces[0][1]

def get_main_data_dir():
    return "/squai/s3811141-faiss/inbe405h-unarxive"

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
