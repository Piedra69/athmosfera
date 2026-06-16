import base64
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw, ImageFilter
from tensorflow.keras import layers
from werkzeug.utils import secure_filename


APP_NAME = "Athmosfera"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = MODEL_DIR / "athmosfera_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

IMG_SIZE = 128
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
WEATHER_CLASSES = ["soleado", "lluvioso", "nublado", "nevado", "niebla", "tormenta"]

MODEL_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


HTML_PAGE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Athmosfera | Cámara en tiempo real</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, Helvetica, sans-serif;
            color: white;
            background:
                radial-gradient(circle at top left, rgba(91, 171, 255, .85), transparent 30%),
                radial-gradient(circle at bottom right, rgba(166, 95, 255, .75), transparent 32%),
                linear-gradient(135deg, #061527, #10182f 45%, #22143e);
        }

        .page {
            width: min(1120px, 94%);
            margin: auto;
            padding: 28px 0 46px;
        }

        .hero, .panel, .camera-box, .result-card, .info-box {
            border: 1px solid rgba(255, 255, 255, .18);
            background: rgba(255, 255, 255, .12);
            box-shadow: 0 24px 80px rgba(0, 0, 0, .28);
            backdrop-filter: blur(18px);
            border-radius: 32px;
        }

        .hero {
            padding: 34px;
            text-align: center;
            margin-bottom: 24px;
        }

        .badge {
            display: inline-block;
            padding: 9px 16px;
            border-radius: 999px;
            background: rgba(255, 255, 255, .15);
            color: #d8edff;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: .12em;
            font-weight: 800;
        }

        h1 {
            margin: 18px 0 12px;
            font-size: clamp(3.1rem, 10vw, 7rem);
            letter-spacing: -.08em;
            line-height: .9;
        }

        .subtitle {
            max-width: 780px;
            margin: 0 auto;
            color: rgba(255, 255, 255, .82);
            font-size: 17px;
            line-height: 1.65;
        }

        .grid {
            display: grid;
            grid-template-columns: 1.1fr .9fr;
            gap: 22px;
        }

        .panel {
            padding: 22px;
        }

        .tabs {
            display: flex;
            gap: 12px;
            margin-bottom: 18px;
            flex-wrap: wrap;
        }

        .tab-button, button {
            border: 0;
            cursor: pointer;
            border-radius: 16px;
            padding: 13px 18px;
            font-weight: 900;
            color: #061527;
            background: linear-gradient(135deg, #e5f8ff, #bfcaff);
            transition: .22s;
        }

        .tab-button.secondary, button.secondary {
            color: white;
            background: rgba(255, 255, 255, .13);
            border: 1px solid rgba(255, 255, 255, .18);
        }

        button.danger {
            color: white;
            background: rgba(255, 80, 80, .25);
            border: 1px solid rgba(255, 130, 130, .30);
        }

        .tab-button:hover, button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 40px rgba(0, 0, 0, .20);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .upload-label {
            cursor: pointer;
            min-height: 180px;
            display: grid;
            place-items: center;
            gap: 8px;
            padding: 22px;
            border: 2px dashed rgba(255, 255, 255, .38);
            border-radius: 26px;
            background: rgba(255, 255, 255, .08);
            text-align: center;
            transition: .25s;
        }

        .upload-label:hover {
            transform: translateY(-2px);
            background: rgba(255, 255, 255, .15);
            border-color: rgba(255, 255, 255, .72);
        }

        .upload-label span { font-size: 3.2rem; }

        .upload-label small {
            color: rgba(255, 255, 255, .72);
            line-height: 1.5;
        }

        input[type=file] { display: none; }

        .actions {
            display: flex;
            gap: 12px;
            margin-top: 16px;
            flex-wrap: wrap;
        }

        .camera-box { padding: 16px; }

        video, canvas, #previewImage {
            width: 100%;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, .20);
            background: rgba(0, 0, 0, .30);
        }

        video {
            aspect-ratio: 4 / 3;
            object-fit: cover;
        }

        canvas { display: none; }

        #previewImage {
            display: none;
            max-height: 430px;
            object-fit: cover;
        }

        .status {
            margin-top: 14px;
            color: rgba(255, 255, 255, .78);
            line-height: 1.5;
        }

        .result-card { padding: 24px; }

        .small-title {
            color: rgba(255, 255, 255, .68);
            font-weight: 900;
            letter-spacing: .1em;
            text-transform: uppercase;
            font-size: 13px;
            margin: 0;
        }

        .main-result {
            font-size: clamp(2.3rem, 8vw, 4.3rem);
            margin: 8px 0 6px;
        }

        .confidence {
            color: #d8edff;
            font-size: 18px;
            margin-bottom: 22px;
        }

        .bar-row { margin-bottom: 16px; }

        .bar-info {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 7px;
            font-weight: 800;
            color: rgba(255, 255, 255, .88);
        }

        .bar-bg {
            height: 12px;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(255, 255, 255, .16);
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #e8f8ff, #b8c6ff);
        }

        .error {
            margin-top: 14px;
            padding: 13px 16px;
            border-radius: 16px;
            color: #ffe6e6;
            background: rgba(255, 80, 80, .18);
            border: 1px solid rgba(255, 130, 130, .35);
            display: none;
        }

        .info-box {
            margin-top: 22px;
            padding: 20px;
            color: rgba(255, 255, 255, .82);
            line-height: 1.65;
        }

        .info-box strong { color: white; }

        @media (max-width: 820px) {
            .page { padding-top: 18px; }
            .hero { padding: 26px 20px; border-radius: 26px; }
            .grid { grid-template-columns: 1fr; }
            .panel, .result-card, .info-box { padding: 18px; border-radius: 24px; }
            .tab-button, button { width: 100%; }
            .actions { display: grid; grid-template-columns: 1fr; }
            .subtitle { font-size: 15.5px; }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <p class="badge">TensorFlow + Flask + cámara</p>
            <h1>Athmosfera</h1>
            <p class="subtitle">
                Reconocimiento de clima visible por imagen o cámara en tiempo real.
                Puedes subir una foto o activar la cámara para que el sistema analice cuadros automáticamente.
            </p>
        </section>

        <section class="grid">
            <div class="panel">
                <div class="tabs">
                    <button class="tab-button" id="tabUpload">Subir imagen</button>
                    <button class="tab-button secondary" id="tabCamera">Cámara en vivo</button>
                </div>

                <div id="uploadMode" class="tab-content active">
                    <form id="uploadForm">
                        <label class="upload-label" for="imageInput">
                            <span>☁️</span>
                            <strong id="fileLabel">Selecciona una imagen</strong>
                            <small>También sirve desde celular. JPG, PNG, WEBP o BMP.</small>
                        </label>
                        <input type="file" name="image" id="imageInput" accept="image/*" capture="environment" required>
                        <div class="actions">
                            <button type="submit">Analizar imagen</button>
                        </div>
                    </form>

                    <div class="camera-box" style="margin-top: 18px;">
                        <img id="previewImage" alt="Imagen subida">
                    </div>
                </div>

                <div id="cameraMode" class="tab-content">
                    <div class="camera-box">
                        <video id="video" autoplay playsinline muted></video>
                        <canvas id="canvas" width="128" height="128"></canvas>
                    </div>

                    <div class="actions">
                        <button id="startCamera" type="button">Iniciar cámara</button>
                        <button id="switchCamera" type="button" class="secondary">Cambiar cámara</button>
                        <button id="stopCamera" type="button" class="danger">Detener</button>
                    </div>

                    <p class="status" id="cameraStatus">
                        La cámara analiza una captura aproximadamente cada segundo.
                    </p>
                </div>

                <div class="error" id="errorBox"></div>
            </div>

            <div class="result-card">
                <p class="small-title">Resultado principal</p>
                <h2 class="main-result" id="mainResult">Sin análisis</h2>
                <p class="confidence" id="confidence">Sube una imagen o activa la cámara.</p>
                <div id="bars"></div>
            </div>
        </section>

        <section class="info-box">
            <strong>Importante:</strong>
            la cámara en tiempo real funciona en navegador usando permisos de cámara.
            En celular normalmente necesita HTTPS. Localmente funciona mejor en laptop con
            <strong>http://127.0.0.1:5000</strong>. Para celular real, despliega la app en Render
            u otro hosting con HTTPS.
        </section>
    </main>

    <script>
        const tabUpload = document.getElementById("tabUpload");
        const tabCamera = document.getElementById("tabCamera");
        const uploadMode = document.getElementById("uploadMode");
        const cameraMode = document.getElementById("cameraMode");

        const uploadForm = document.getElementById("uploadForm");
        const imageInput = document.getElementById("imageInput");
        const fileLabel = document.getElementById("fileLabel");
        const previewImage = document.getElementById("previewImage");

        const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        const startCameraButton = document.getElementById("startCamera");
        const switchCameraButton = document.getElementById("switchCamera");
        const stopCameraButton = document.getElementById("stopCamera");
        const cameraStatus = document.getElementById("cameraStatus");

        const mainResult = document.getElementById("mainResult");
        const confidence = document.getElementById("confidence");
        const bars = document.getElementById("bars");
        const errorBox = document.getElementById("errorBox");

        let stream = null;
        let cameraLoop = null;
        let isPredicting = false;
        let facingMode = "environment";

        function showError(message) {
            errorBox.style.display = "block";
            errorBox.textContent = message;
        }

        function clearError() {
            errorBox.style.display = "none";
            errorBox.textContent = "";
        }

        function activateTab(mode) {
            clearError();

            if (mode === "upload") {
                uploadMode.classList.add("active");
                cameraMode.classList.remove("active");
                tabUpload.classList.remove("secondary");
                tabCamera.classList.add("secondary");
            } else {
                cameraMode.classList.add("active");
                uploadMode.classList.remove("active");
                tabCamera.classList.remove("secondary");
                tabUpload.classList.add("secondary");
            }
        }

        tabUpload.addEventListener("click", () => activateTab("upload"));
        tabCamera.addEventListener("click", () => activateTab("camera"));

        imageInput.addEventListener("change", () => {
            if (imageInput.files.length > 0) {
                fileLabel.textContent = imageInput.files[0].name;
                previewImage.src = URL.createObjectURL(imageInput.files[0]);
                previewImage.style.display = "block";
            }
        });

        function renderResult(data) {
            if (data.error) {
                showError(data.error);
                return;
            }

            clearError();
            mainResult.textContent = data.label;
            confidence.textContent = `${data.confidence}% de confianza`;

            bars.innerHTML = "";
            data.items.forEach(item => {
                const row = document.createElement("div");
                row.className = "bar-row";
                row.innerHTML = `
                    <div class="bar-info">
                        <span>${item.label}</span>
                        <span>${item.probability}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width:${item.probability}%;"></div>
                    </div>
                `;
                bars.appendChild(row);
            });
        }

        uploadForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            clearError();

            const formData = new FormData(uploadForm);

            try {
                const response = await fetch("/predict_upload", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    showError(data.error || "No se pudo analizar la imagen.");
                    return;
                }

                if (data.image) {
                    previewImage.src = data.image;
                    previewImage.style.display = "block";
                }

                renderResult(data);
            } catch (error) {
                showError("Error de conexión con Flask. Revisa si app.py está ejecutándose.");
            }
        });

        async function startCamera() {
            clearError();

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                showError("Tu navegador no permite abrir la cámara desde esta página.");
                return;
            }

            stopCamera();

            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: facingMode,
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    },
                    audio: false
                });

                video.srcObject = stream;
                cameraStatus.textContent = "Cámara activa. Analizando en tiempo real...";
                cameraLoop = setInterval(captureAndPredict, 1100);
            } catch (error) {
                showError("No se pudo abrir la cámara. En celular normalmente necesitas HTTPS y aceptar permisos.");
            }
        }

        function stopCamera() {
            if (cameraLoop) {
                clearInterval(cameraLoop);
                cameraLoop = null;
            }

            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }

            video.srcObject = null;
            cameraStatus.textContent = "Cámara detenida.";
        }

        async function captureAndPredict() {
            if (!stream || isPredicting || video.readyState < 2) return;

            isPredicting = true;

            try {
                const context = canvas.getContext("2d");
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL("image/jpeg", 0.75);

                const response = await fetch("/predict_frame", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ image: imageData })
                });

                const data = await response.json();

                if (!response.ok) {
                    showError(data.error || "No se pudo analizar la cámara.");
                } else {
                    renderResult(data);
                }
            } catch (error) {
                showError("Error analizando la cámara.");
            } finally {
                isPredicting = false;
            }
        }

        startCameraButton.addEventListener("click", startCamera);

        switchCameraButton.addEventListener("click", () => {
            facingMode = facingMode === "environment" ? "user" : "environment";
            startCamera();
        });

        stopCameraButton.addEventListener("click", stopCamera);
    </script>
</body>
</html>
'''


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def random_weather_image(label, size=IMG_SIZE):
    img = Image.new("RGB", (size, size), (120, 160, 200))
    draw = ImageDraw.Draw(img, "RGBA")

    if label == "soleado":
        bg1 = (80, 165, 255)
        bg2 = (170, 220, 255)
        for y in range(size):
            r = int(bg1[0] + (bg2[0] - bg1[0]) * y / size)
            g = int(bg1[1] + (bg2[1] - bg1[1]) * y / size)
            b = int(bg1[2] + (bg2[2] - bg1[2]) * y / size)
            draw.line((0, y, size, y), fill=(r, g, b))
        cx, cy = random.randint(26, 96), random.randint(20, 55)
        radius = random.randint(13, 22)
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(255, 218, 70, 255))
        for _ in range(8):
            angle = random.random() * 6.28
            x1 = cx + int(np.cos(angle) * (radius + 8))
            y1 = cy + int(np.sin(angle) * (radius + 8))
            x2 = cx + int(np.cos(angle) * (radius + 25))
            y2 = cy + int(np.sin(angle) * (radius + 25))
            draw.line((x1, y1, x2, y2), fill=(255, 236, 120, 180), width=2)

    elif label == "nublado":
        img = Image.new("RGB", (size, size), (135, 150, 165))
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(8):
            x = random.randint(-15, size - 20)
            y = random.randint(10, size - 30)
            w = random.randint(45, 85)
            h = random.randint(22, 42)
            color = random.choice([(210, 215, 220, 210), (180, 185, 195, 220), (155, 165, 175, 220)])
            draw.ellipse((x, y, x+w, y+h), fill=color)
            draw.ellipse((x+20, y-10, x+w+10, y+h+8), fill=color)

    elif label == "lluvioso":
        img = Image.new("RGB", (size, size), (70, 85, 110))
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(7):
            x = random.randint(-10, size - 25)
            y = random.randint(5, 60)
            w = random.randint(55, 95)
            h = random.randint(24, 44)
            draw.ellipse((x, y, x+w, y+h), fill=(95, 105, 125, 230))
        for _ in range(42):
            x = random.randint(0, size)
            y = random.randint(40, size)
            length = random.randint(10, 22)
            draw.line((x, y, x-4, y+length), fill=(120, 185, 255, 190), width=2)

    elif label == "nevado":
        img = Image.new("RGB", (size, size), (185, 205, 225))
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle((0, int(size * 0.65), size, size), fill=(245, 248, 255, 255))
        for _ in range(70):
            x = random.randint(0, size)
            y = random.randint(0, size)
            r = random.choice([1, 1, 2, 2, 3])
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 255, 255, 230))

    elif label == "niebla":
        img = Image.new("RGB", (size, size), (170, 180, 185))
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(10):
            y = random.randint(10, size - 20)
            alpha = random.randint(80, 145)
            draw.rounded_rectangle((-20, y, size + 20, y + random.randint(10, 20)), radius=10, fill=(230, 235, 238, alpha))
        img = img.filter(ImageFilter.GaussianBlur(radius=2.5))

    elif label == "tormenta":
        img = Image.new("RGB", (size, size), (25, 32, 50))
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(7):
            x = random.randint(-15, size - 25)
            y = random.randint(8, 70)
            w = random.randint(60, 100)
            h = random.randint(28, 48)
            draw.ellipse((x, y, x+w, y+h), fill=(45, 50, 68, 240))
        x = random.randint(38, 88)
        y = random.randint(42, 70)
        bolt = [
            (x, y),
            (x - 12, y + 30),
            (x + 3, y + 30),
            (x - 10, y + 70),
            (x + 22, y + 24),
            (x + 5, y + 24),
        ]
        draw.polygon(bolt, fill=(255, 232, 80, 245))

    return img


def generate_demo_dataset(samples_per_class=60):
    xs = []
    ys = []

    for index, label in enumerate(WEATHER_CLASSES):
        for _ in range(samples_per_class):
            img = random_weather_image(label)
            arr = np.array(img).astype("float32")
            arr += np.random.normal(0, 6.5, arr.shape).astype("float32")
            arr = np.clip(arr, 0, 255)
            xs.append(arr)
            ys.append(index)

    xs = np.array(xs, dtype="float32")
    ys = np.array(ys, dtype="int32")

    order = np.arange(len(xs))
    np.random.shuffle(order)

    return xs[order], ys[order]


def build_model(num_classes):
    model = tf.keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255.0),
        layers.Conv2D(24, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(48, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(96, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(96, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def ensure_demo_model():
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        return

    print("No se encontró modelo. Creando modelo demo de Athmosfera...")
    x, y = generate_demo_dataset(samples_per_class=60)
    model = build_model(len(WEATHER_CLASSES))

    model.fit(
        x,
        y,
        epochs=5,
        batch_size=24,
        validation_split=0.15,
        verbose=1
    )

    model.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(WEATHER_CLASSES, file, ensure_ascii=False, indent=4)

    print("Modelo demo creado correctamente.")


def load_model_and_labels():
    ensure_demo_model()

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    return model, labels


MODEL, LABELS = load_model_and_labels()


def prepare_pil_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32")
    return np.expand_dims(arr, axis=0)


def prepare_image_from_path(path):
    image = Image.open(path)
    return prepare_pil_image(image)


def predict_array(arr):
    predictions = MODEL.predict(arr, verbose=0)[0]
    best_index = int(np.argmax(predictions))

    items = []
    for label, probability in zip(LABELS, predictions):
        items.append({
            "label": label.capitalize(),
            "probability": round(float(probability) * 100, 2)
        })

    items.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "label": LABELS[best_index].capitalize(),
        "confidence": round(float(predictions[best_index]) * 100, 2),
        "items": items
    }


def predict_path(path):
    arr = prepare_image_from_path(path)
    return predict_array(arr)


def image_to_data_url(path):
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    ext = Path(path).suffix.replace(".", "").lower() or "png"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{encoded}"


def data_url_to_pil(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.route("/", methods=["GET"])
def index():
    return HTML_PAGE


@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Primero selecciona una imagen."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Formato no permitido. Usa JPG, JPEG, PNG, WEBP o BMP."}), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    result = predict_path(save_path)
    result["image"] = image_to_data_url(save_path)

    return jsonify(result)


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "No se recibió imagen desde la cámara."}), 400

    try:
        image = data_url_to_pil(image_data)
        arr = prepare_pil_image(image)
        result = predict_array(arr)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"No se pudo analizar el cuadro de cámara: {exc}"}), 400


@app.route("/health")
def health():
    return jsonify({
        "app": APP_NAME,
        "status": "ok",
        "model_loaded": MODEL is not None,
        "labels": LABELS
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
