import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
APP_FILE = PROJECT_DIR / "app.py"


def venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main():
    print("\n=== ATHMOSFERA | INICIANDO APP ===\n")

    python_exec = venv_python()

    if not VENV_DIR.exists() or not python_exec.exists():
        print("No existe el entorno virtual .venv.")
        print("Primero ejecuta:")
        print("python crear_venv.py")
        sys.exit(1)

    if not APP_FILE.exists():
        print("ERROR: No se encontró app.py")
        sys.exit(1)

    print("Usando Python del entorno virtual:")
    print(python_exec)
    print("\nAbre en tu navegador:")
    print("http://127.0.0.1:5000")
    print("\nPara detener la app, presiona CTRL + C.\n")

    subprocess.call([str(python_exec), str(APP_FILE)])


if __name__ == "__main__":
    main()
