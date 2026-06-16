# Dataset de Athmosfera

Coloca aquí tus imágenes reales organizadas por clase.

Estructura:

dataset/
├── soleado/
├── lluvioso/
├── nublado/
├── nevado/
├── niebla/
└── tormenta/

Cada carpeta representa una clase.

Después ejecuta:

python train_real_model.py --data dataset --epochs 15
