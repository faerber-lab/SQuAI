import os
import sys
import subprocess

def get_main_data_dir():
    fallback_paths = [
        "/projects/p_scads_finetune/squai_faiss",
        f"{os.getenv('HOME')}/data_dir",
        "/data/horse/ws/inbe405h-unarxive",
        "/data/horse/ws/s3811141-faiss/inbe405h-unarxive"
    ]

    resolved_path = None

    for path in fallback_paths:
        # Erst prüfen, ob es eine "data_dir"-Datei ist
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

        # Wenn es direkt ein Verzeichnis ist
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
