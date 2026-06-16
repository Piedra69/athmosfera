import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"


def main():
    print("\n=== ATHMOSFERA | VERIFICAR VENV ===\n")

    print("Python actual:")
    print(sys.executable)

    print("\nCarpeta del proyecto:")
    print(PROJECT_DIR)

    print("\nExiste .venv:")
    print(VENV_DIR.exists())

    if os.name == "nt":
        python_venv = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_venv = VENV_DIR / "bin" / "python"

    print("\nPython del venv:")
    print(python_venv)

    print("\nExiste Python del venv:")
    print(python_venv.exists())

    if python_venv.exists():
        print("\nTodo bien. Puedes ejecutar:")
        print("python ejecutar_app.py")
    else:
        print("\nFalta crear el venv. Ejecuta:")
        print("python crear_venv.py")


if __name__ == "__main__":
    main()
