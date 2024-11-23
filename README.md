# Real-Time Trash Classification on Embedded Devices

## Introduction
This project focuses on developing a real-time trash classification system leveraging deep learning models. The initiative addresses a critical challenge in waste management by automating the segregation of recyclable and non-recyclable waste categories, improving accuracy and efficiency over manual methods.

While the paper associated with this project evaluates the performance of various deep learning architectures, this repository extends the scope to include deployment of the best-performing model on a Raspberry Pi 5. The result is a practical, embedded system capable of classifying waste in real-time using a Raspberry Pi-compatible HQ camera.

---

## Methodology

### Dataset
The **TrashNet** dataset was selected for its comprehensive and publicly available collection of approximately 400–500 images per class. It categorizes waste into six classes: cardboard, glass, metal, paper, plastic, and general trash.

### Model Selection
Several convolutional neural network (CNN) architectures were explored, including:
- **EfficientNet** (B0 to B7 variants)
- **MobileNetV2**
- **ResNet50**
- **DenseNet121**

#### Key Evaluation Metrics
- **Accuracy:** Measured on training, validation, and test datasets.
- **Inference Time:** Time taken for the model to classify an object.
- **Frames Per Second (FPS):** The rate at which the system processes frames in real-time.
- **Model Size:** Memory requirements for deployment.

### Deployment
Models were trained using **TensorFlow** and **Keras**, and optimized for embedded deployment by converting them to TensorFlow Lite format. The Raspberry Pi 5 served as the deployment platform, using its computing capabilities and GPIO for integration with external sensors.

The Raspberry Pi 5 was paired with a Sony IMX477R HQ camera module to capture waste images for classification. The inference pipeline processes live camera feed to perform real-time predictions.

---

## Results

### Pre-Deployment
- **EfficientNet B3** achieved the best balance of accuracy (85.27%) and performance metrics among tested models.
- Lightweight models like **MobileNetV2** offered faster inference times but were less stable in predictions.

### Post-Deployment
After converting to TensorFlow Lite and running on Raspberry Pi 5:
- **MobileNetV2** exhibited the fastest performance (15 FPS, 32 ms inference time) but suffered from unstable predictions.
- **EfficientNet B3** was identified as the most suitable model for real-time deployment, balancing accuracy, inference time (188 ms), and FPS (4.8).

---

## Hardware Requirements
1. **Raspberry Pi 5:**
   - Quad-core 64-bit Arm Cortex-A76 CPU
   - 4GB or 8GB SDRAM options
2. **Raspberry Pi HQ Camera:**
   - Sony IMX477R sensor
   - 12.3 MP resolution
3. **Power Supply:** Compatible with Raspberry Pi 5
4. **SD Card:** For storing the trained model and necessary scripts

---

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Fanuelpeter007/Waste-Sorting
2. Install dependencies:
   ```bash
   pip install tensorflow tensorflow-lite opencv-python
3. Transfer the trained TensorFlow Lite model to the Raspberry Pi.
4. Connect the HQ camera module to the Raspberry Pi and ensure it is configured.
5. Run the real-time classification script:
   ```bash
   python classify.py

## Future Work
Enhance model accuracy by fine-tuning additional layers.
Increase training epochs for deeper learning.
Experiment with advanced augmentation techniques for dataset enrichment.
Explore deployment on alternative embedded platforms, such as NVIDIA Jetson.


# Check out the project here:
https://www.dropbox.com/scl/fi/gn2253p7blcot3ejch9rx/Trash-classifier-video.mp4?rlkey=lbdat4oshmrfep06salcrdqlg&e=1&st=50te1myi&dl=0
