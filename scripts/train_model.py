"""Train and evaluate a digit classifier from class-folder image data.

Expected layout:
data/digits/BSL/
  train/
    0/ image1.jpg ...
    1/ image2.jpg ...
  test/
    0/ image3.jpg ...
    1/ image4.jpg ...

Or use a single class-folder directory:
data/digits/
  0/ image1.jpg ...
  1/ image2.jpg ...
  ...
Run: python scripts/train_model.py --data data/digits/BSL --output models/digits.keras --labels models/digits_labels.json
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (64, 64)


def prepare_dataset(dataset, training=False):
    dataset = dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        dataset = dataset.shuffle(1024, seed=123, reshuffle_each_iteration=True)
    return dataset.prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/digits/BSL")
    parser.add_argument("--output", default="models/digits.keras")
    parser.add_argument("--labels", default="models/digits_labels.json")
    parser.add_argument("--metrics", default="models/digits_metrics.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset folder not found: {data_dir}")
    train_dir = data_dir / "train" if (data_dir / "train").is_dir() else data_dir
    test_dir = data_dir / "test" if (data_dir / "test").is_dir() else None
    class_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if len(class_names) < 2:
        raise SystemExit(f"Create at least two class subfolders inside {train_dir}.")
    if not 0 < args.validation_split < 1:
        raise SystemExit("--validation-split must be greater than 0 and less than 1.")
    if test_dir is not None:
        test_classes = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
        if test_classes != class_names:
            raise SystemExit(
                f"Test labels {test_classes} do not match training labels {class_names}."
            )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=args.validation_split,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
        class_names=class_names,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=args.validation_split,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
        class_names=class_names,
        shuffle=False,
    )
    train_ds = prepare_dataset(train_ds, training=True)
    val_ds = prepare_dataset(val_ds)

    model_layers = [
        tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomTranslation(0.06, 0.06),
        tf.keras.layers.RandomZoom(0.1),
    ]
    for filters in (32, 64, 128):
        model_layers.extend(
            [
                tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(),
            ]
        )
    model_layers.extend(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(len(class_names), activation="softmax"),
        ]
    )
    model = tf.keras.Sequential(model_layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    output = Path(args.output)
    labels_path = Path(args.labels)
    metrics_path = Path(args.metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output, monitor="val_accuracy", mode="max", save_best_only=True
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if test_dir is not None:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=IMG_SIZE,
            batch_size=args.batch_size,
            label_mode="int",
            class_names=class_names,
            shuffle=False,
        )
        test_ds = prepare_dataset(test_ds)
        test_loss, test_accuracy = model.evaluate(test_ds, return_dict=False)
        print(f"Final test loss: {test_loss:.4f}")
        print(f"Final test accuracy: {test_accuracy:.4%}")
        predictions = model.predict(test_ds, verbose=0)
        true_labels = tf.concat([labels for _, labels in test_ds], axis=0)
        predicted_labels = tf.argmax(predictions, axis=1, output_type=tf.int32)
        confusion = tf.math.confusion_matrix(
            true_labels, predicted_labels, num_classes=len(class_names)
        )
        class_accuracies = tf.math.divide_no_nan(
            tf.linalg.diag_part(confusion),
            tf.reduce_sum(confusion, axis=1),
        )
        per_digit_accuracy = {
            class_name: float(class_accuracy)
            for class_name, class_accuracy in zip(class_names, class_accuracies)
        }
        for class_name, class_accuracy in per_digit_accuracy.items():
            print(f"Test accuracy for {class_name}: {class_accuracy:.2%}")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "test_samples": int(tf.reduce_sum(confusion)),
                    "test_loss": float(test_loss),
                    "test_accuracy": float(test_accuracy),
                    "per_digit_accuracy": per_digit_accuracy,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved test metrics: {metrics_path}")
    else:
        print("No test/ folder found; final held-out test evaluation was skipped.")

    model.save(output)
    labels_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model: {output}\nSaved labels: {labels_path}")


if __name__ == "__main__":
    main()
