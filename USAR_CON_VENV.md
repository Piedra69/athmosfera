# Athmosfera con entorno virtual

Esta versión está preparada para trabajar con un entorno virtual `.venv`.

## Paso 1: Crear el entorno virtual

Ejecuta:

```bash
python crear_venv.py
```

Este script hace automáticamente:

1. Crea la carpeta `.venv`.
2. Actualiza pip.
3. Instala todas las dependencias de `requirements.txt`.

## Paso 2: Lanzar la aplicación

Ejecuta:

```bash
python ejecutar_app.py
```

Después abre:

```text
http://127.0.0.1:5000
```

## Paso 3: Usar la app

La página tiene dos opciones:

1. Subir imagen.
2. Cámara en vivo.

## Verificar si el venv está bien

Puedes ejecutar:

```bash
python verificar_venv.py
```

## Importante

No necesitas activar manualmente el venv con:

```bash
.venv\Scripts\activate
```

o

```bash
source .venv/bin/activate
```

El archivo `ejecutar_app.py` usa directamente el Python que está dentro de `.venv`.

## Si algo falla

Borra la carpeta `.venv` y vuelve a ejecutar:

```bash
python crear_venv.py
```

## Para entrenar el modelo real usando el venv

Primero crea el venv:

```bash
python crear_venv.py
```

Luego ejecuta el entrenamiento usando el Python del venv.

En Windows:

```bash
.venv\Scripts\python.exe train_real_model.py --data dataset --epochs 15
```

En macOS/Linux:

```bash
.venv/bin/python train_real_model.py --data dataset --epochs 15
```
