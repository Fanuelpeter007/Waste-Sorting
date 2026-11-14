"""Train a TrashNet classifier with transfer learning and export it to TensorFlow Lite.

The exported model matches what src/sorter.py expects:
  - input: 1 x 256 x 256 x 3, float32, RGB scaled to [0, 1]
  - output: 1 x 6 class scores in the order of CLASS_LIST

Dataset layout (one folder per class):
    DATA_DIR/train/cardboard/*.jpg ... DATA_DIR/train/trash/*.jpg
    DATA_DIR/test/cardboard/*.jpg  ... DATA_DIR/test/trash/*.jpg

Usage:
    python src/train_classifier.py --data data/trashnet --backbone efficientnet_b3
    python src/train_classifier.py --data data/trashnet --backbone mobilenet_v2 --epochs 30 --fine-tune 20
"""

import argparse
import os

import tensorflow as tf

CLASS_LIST = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
INPUT_SIZE = (256, 256)

# Each entry: Keras constructor and the matching preprocessing function.
# The model receives pixels in [0, 1]; they are rescaled to [0, 255] and then
# passed through the backbone's own preprocessing inside the model.
BACKBONES = {
    'efficientnet_b0': (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    'efficientnet_b3': (tf.keras.applications.EfficientNetB3, tf.keras.applications.efficientnet.preprocess_input),
    'mobilenet_v2': (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    'resnet50': (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
    'densenet121': (tf.keras.applications.DenseNet121, tf.keras.applications.densenet.preprocess_input),
}


def parse_args():
    """Read command line options."""
    p = argparse.ArgumentParser(description="Train a TrashNet classifier and export it to TFLite.")
    p.add_argument("--data", required=True, help="Folder with train/ and test/ subfolders")
    p.add_argument("--backbone", default="efficientnet_b3", choices=sorted(BACKBONES))
    p.add_argument("--epochs", type=int, default=20, help="Epochs with the backbone frozen")
    p.add_argument("--fine-tune", type=int, default=0,
                   help="Extra epochs with the top layers of the backbone unfrozen")
    p.add_argument("--unfreeze", type=int, default=30,
                   help="Number of backbone layers to unfreeze for fine-tuning")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--weights", default="imagenet",
                   help="Backbone weights: 'imagenet' or 'none' (default: imagenet)")
    p.add_argument("--out", default="models", help="Output folder")
    return p.parse_args()


def load_split(folder, batch, shuffle, subset=None):
    """Load images as float32 tensors in [0, 1] with integer labels in CLASS_LIST order."""
    ds = tf.keras.utils.image_dataset_from_directory(
        folder, labels="inferred", class_names=CLASS_LIST, image_size=INPUT_SIZE,
        batch_size=batch, shuffle=shuffle, seed=42,
        validation_split=0.1 if subset else None, subset=subset)
    return ds.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)


def build_model(backbone_name, weights):
    """Frozen pretrained backbone with a small classification head."""
    constructor, preprocess = BACKBONES[backbone_name]
    base = constructor(include_top=False, weights=weights, input_shape=INPUT_SIZE + (3,))
    base.trainable = False

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augment")

    inputs = tf.keras.Input(shape=INPUT_SIZE + (3,), name="image")
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(255.0)(x)
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_LIST), activation="softmax", name="scores")(x)
    return tf.keras.Model(inputs, outputs), base


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_ds = load_split(os.path.join(args.data, "train"), args.batch, True, "training")
    val_ds = load_split(os.path.join(args.data, "train"), args.batch, False, "validation")
    test_ds = load_split(os.path.join(args.data, "test"), args.batch, False)

    weights = None if args.weights == "none" else args.weights
    model, base = build_model(args.backbone, weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    if args.fine_tune > 0:
        base.trainable = True
        for layer in base.layers[:-args.unfreeze]:
            layer.trainable = False
        # Keep batch norm statistics fixed while fine-tuning.
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune)

    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy: {acc:.4f}  (loss {loss:.4f})")

    keras_path = os.path.join(args.out, f"{args.backbone}.keras")
    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_path = os.path.join(args.out, f"{args.backbone}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(converter.convert())
    print(f"Saved {keras_path} and {tflite_path}")


if __name__ == "__main__":
    main()
