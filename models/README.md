# Models

Trained models are not stored in git. Put them here.

| File | Made by | Used by |
|---|---|---|
| `model.tflite` | `src/train_classifier.py` (rename the output, for example `efficientnet_b3.tflite`) | `src/sorter.py` (default path) |
| `<backbone>.keras` | `src/train_classifier.py` | Further training or conversion |
| `custom_model_lite/detect.tflite` | `notebooks/tflite_ssd_detection_training.ipynb` | TFLite object detection scripts (not `src/sorter.py`) |

`src/sorter.py` expects a classifier with a 1 x 256 x 256 x 3 float32 input in
[0, 1] and six outputs in this order: cardboard, glass, metal, paper, plastic,
trash.
