import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers


IMG_SIZE = 128
SEED = 123


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ])

    model = tf.keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
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


def main():
    parser = argparse.ArgumentParser(description="Entrena Athmosfera con imágenes reales.")
    parser.add_argument("--data", default="dataset", help="Carpeta del dataset.")
    parser.add_argument("--epochs", type=int, default=15, help="Cantidad de épocas.")
    parser.add_argument("--batch", type=int, default=24, help="Tamaño de lote.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError("No existe la carpeta dataset.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch,
        label_mode="int"
    )

    class_names = train_ds.class_names

    if len(class_names) < 2:
        raise ValueError("Necesitas al menos dos carpetas de clases con imágenes.")

    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    model = build_model(len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "athmosfera_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    model.save(model_dir / "athmosfera_model.keras")

    with open(model_dir / "labels.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=4)

    print("Modelo real entrenado y guardado correctamente.")
    print("Archivo:", model_dir / "athmosfera_model.keras")
    print("Clases:", class_names)
    print("Última precisión de validación:", history.history.get("val_accuracy", ["N/A"])[-1])


if __name__ == "__main__":
    main()
