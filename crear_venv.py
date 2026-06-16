import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"


def run(command):
    print("Ejecutando:", " ".join(str(x) for x in command))
    subprocess.check_call(command)


def venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main():
    print("\n=== ATHMOSFERA | CREAR ENTORNO VIRTUAL ===\n")

    if not REQUIREMENTS.exists():
        print("ERROR: No se encontró requirements.txt")
        sys.exit(1)

    if not VENV_DIR.exists():
        print("Creando entorno virtual en .venv ...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("El entorno virtual .venv ya existe.")

    python_exec = venv_python()

    if not python_exec.exists():
        print("ERROR: No se encontró el Python del entorno virtual.")
        sys.exit(1)

    print("\nActualizando pip...")
    run([str(python_exec), "-m", "pip", "install", "--upgrade", "pip"])

    print("\nInstalando dependencias...")
    run([str(python_exec), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    print("\nListo. Ahora ejecuta:")
    print("python ejecutar_app.py")
    print("\n")


if __name__ == "__main__":
    main()
