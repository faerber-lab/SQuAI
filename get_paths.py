import os
import sys

def get_main_data_dir():
    fallback_paths = [
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
    return "/home/inbe405h/bm25_env/bin/python"
