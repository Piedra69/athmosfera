# Athmosfera con venv

Esta versión trabaja con entorno virtual `.venv`.

## Forma más fácil de ejecutar

Primero crea el entorno virtual:

```bash
python crear_venv.py
```

Luego inicia la aplicación:

```bash
python ejecutar_app.py
```

Abre:

```text
http://127.0.0.1:5000
```

En Windows también puedes usar doble clic en:

```text
crear_venv_windows.bat
ejecutar_windows.bat
```

---

# Athmosfera

Aplicación web en Python con Flask y TensorFlow para reconocer el clima visible en una imagen.

Incluye dos modos:

1. Subir imagen.
2. Cámara en vivo con análisis automático.

## Importante

El backend es Python con Flask y TensorFlow.

Para que la cámara funcione en navegador se necesita JavaScript, porque Python no puede abrir directamente la cámara del navegador. JavaScript solo captura la imagen y la envía al backend Python. La predicción sigue siendo hecha por TensorFlow en Python.

## Qué reconoce

- soleado
- lluvioso
- nublado
- nevado
- niebla
- tormenta

## Ejecutar localmente

Crear entorno virtual:

Windows:

python -m venv venv
venv\Scripts\activate

macOS/Linux:

python3 -m venv venv
source venv/bin/activate

Instalar dependencias:

pip install -r requirements.txt

Ejecutar:

python app.py

Abrir:

http://127.0.0.1:5000

## Cámara en celular

Para cámara en tiempo real desde celular, normalmente necesitas HTTPS.

Esto significa:

- Localhost en laptop suele funcionar: http://127.0.0.1:5000
- En celular usando IP local como http://192.168.1.10:5000 puede bloquear la cámara
- En Render/Railway/PythonAnywhere con HTTPS sí debería pedir permisos de cámara correctamente

## Entrenar con imágenes reales

Coloca imágenes en:

dataset/
├── soleado/
├── lluvioso/
├── nublado/
├── nevado/
├── niebla/
└── tormenta/

Ejemplo:

dataset/soleado/sol1.jpg
dataset/lluvioso/lluvia1.jpg
dataset/nublado/nube1.jpg

Recomendación:

- Mínimo: 50 imágenes por clase.
- Bueno: 100 a 200 imágenes por clase.
- Mejor: 300 o más imágenes por clase.

Entrenar:

python train_real_model.py --data dataset --epochs 15

Luego iniciar la app:

python app.py

## GitHub Pages

GitHub Pages no ejecuta Python, Flask ni TensorFlow.

Puedes subir el repositorio a GitHub, pero para que funcione la predicción debes desplegarlo en:

- Render
- Railway
- PythonAnywhere
- Replit
- VPS propio

El proyecto ya incluye:

- Procfile
- render.yaml

## Despliegue recomendado en Render

1. Crea un repositorio en GitHub.
2. Sube este proyecto.
3. En Render crea un Web Service.
4. Conecta tu repositorio.
5. Usa:
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn app:app

## Nota académica

El sistema no predice el clima futuro como meteorología profesional.
Clasifica el clima visible en una imagen o en la cámara en vivo.
