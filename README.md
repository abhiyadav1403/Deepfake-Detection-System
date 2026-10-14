# Deepfake Detection System

An image and video analysis project that uses **EfficientNet-B0**,
PyTorch, and Gradio to classify images as **REAL** or **FAKE**. For
videos, the application samples frames and averages the model's
predictions.

> **Important:** This is a machine-learning research project. A
> prediction or confidence score is not definitive proof that media is
> authentic or manipulated.

## Project Overview

The current model was trained using Dataset 1 and Dataset 2, kept in
separate folders and combined logically during training with PyTorch
`ConcatDataset`.

-   **Class 0:** FAKE
-   **Class 1:** REAL
-   **Backbone:** EfficientNet-B0
-   **Classifier:** Dropout (`p=0.2`) followed by a linear layer
    (`1280 → 2`)

The model can misclassify AI-generated images from generators or image
types not sufficiently represented in the training data. The current
results should not be interpreted as proof of reliable detection for
every AI image.

## Features

-   Upload an image and predict REAL or FAKE.
-   Display FAKE and REAL scores as percentages.
-   Upload a video and analyze sampled frames.
-   Sample approximately one frame per second from videos.
-   Average frame-level probabilities for a video-level prediction.
-   Evaluate with accuracy, precision, recall, F1-score, confusion
    matrix, and ROC-AUC.
-   Run on CPU; the app uses CUDA when available.

## Technology Stack

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  PyTorch                             Model training and inference

  Torchvision                         EfficientNet-B0 and image
                                      transforms

  NumPy                               Numerical calculations and
                                      probability averaging

  OpenCV                              Reading and sampling video frames

  Pillow                              Loading and converting images

  Gradio                              Web interface

  Matplotlib                          Evaluation charts

  Jupyter Notebook                    Dataset analysis, preprocessing,
                                      training, evaluation
  -----------------------------------------------------------------------

`scikit-learn` was intended for some evaluation utilities, but the
current local installation encountered a Windows DLL/Application Control
error. The reported evaluation metrics were calculated without relying
on it.

## Model Workflow

### Image

1.  Load the uploaded image and convert it to RGB.
2.  Resize it to `224 × 224`.
3.  Convert it to a tensor and normalize using ImageNet mean and
    standard deviation.
4.  Run EfficientNet-B0.
5.  Apply softmax to obtain FAKE and REAL scores.
6.  Display the predicted class and scores.

### Video

1.  Open the video with OpenCV.
2.  Sample approximately one frame per second.
3.  Convert each frame from BGR to RGB.
4.  Run each sampled frame through the image classifier.
5.  Average FAKE and REAL scores to produce the video result.

**Note:** Video detection currently uses frame sampling and probability
averaging. It does not use an LSTM or directly learn temporal motion
patterns.

## Datasets

Dataset folders remain separate for traceability. They are combined
logically during training; images are not physically merged.

``` text
dataset/
├── dataset_1/
│   ├── train/
│   │   ├── fake/
│   │   └── real/
│   ├── validation/
│   │   ├── fake/
│   │   └── real/
│   └── test/
│       ├── fake/
│       └── real/
└── dataset_2/
    ├── train/
    │   ├── fake/
    │   └── real/
    ├── validation/
    │   ├── fake/
    │   └── real/
    └── test/
        ├── fake/
        └── real/
```

### Current image counts

  Split          Dataset 1   Dataset 2    Combined
  ------------ ----------- ----------- -----------
  Train              1,428       3,884       5,312
  Validation           306         547         853
  Test                 307       1,126       1,433
  **Total**      **2,041**   **5,557**   **7,598**

These are the recorded local counts. Dataset 2 has 5,557 validated
images available locally. When adding more datasets, record their
sources and keep test data separate from training.

## Evaluation Results

Recorded evaluation on the combined 1,433-image test set:

  Metric                 Result
  ---------------- ------------
  Test accuracy      **89.88%**
  ROC-AUC            **96.98%**
  FAKE precision         93.58%
  FAKE recall            84.74%
  FAKE F1-score          88.94%
  REAL precision         87.04%
  REAL recall            94.63%
  REAL F1-score          90.68%

Recorded confusion matrix counts:

  Actual / Predicted     Predicted FAKE   Predicted REAL
  -------------------- ---------------- ----------------
  Actual FAKE                       583              105
  Actual REAL                        40              705

These counts correspond to 583 correctly predicted fake images, 705
correctly predicted real images, 105 fake images predicted as real, and
40 real images predicted as fake.

**Interpretation:** these metrics describe the held-out test set only.
They do not guarantee equivalent performance on new AI generators, live
camera photos, or all image categories. At least one newly generated AI
image was classified as REAL during manual testing, so external
evaluation is needed before claiming general AI-image detection.

## Project Structure

``` text
deepfake/
├── dataset/
│   ├── dataset_1/
│   └── dataset_2/
├── face_dataset/                 # Face-focused dataset, if used
├── models/
│   └── deepfake_efficientnet_b0_final.pth
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_face_training.ipynb
├── results/                      # Charts and evaluation outputs
├── test_images/
├── test_videos/
├── app.py                        # Gradio application
├── requirements.txt
├── split_dataset.py
├── FINAL_DATASET.csv             # Dataset 2 metadata, if retained
└── README.md
```

Update this example tree if your actual local files differ.

## Installation

Run commands from the project root, where `app.py` is located.

### 1. Activate the virtual environment (Windows PowerShell)

``` powershell
.\.venv\Scripts\Activate.ps1
```

If the environment is already selected in VS Code, you may not need to
activate it again.

### 2. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

Minimum application dependencies:

``` bash
python -m pip install torch torchvision gradio opencv-python numpy Pillow
```

For notebooks and plots:

``` bash
python -m pip install matplotlib jupyter pandas
```

Use the official PyTorch installation selector if you need a specific
CPU/CUDA build.

## Run the Application

From the project root:

``` bash
python app.py
```

The terminal should report that the model loaded successfully and
display a local Gradio URL, commonly `http://127.0.0.1:7860`.

The app expects the checkpoint at:

``` text
models/deepfake_efficientnet_b0_final.pth
```

Ensure the checkpoint is compatible with the EfficientNet-B0
architecture in `app.py`.

## How to Use

### Image Detection

1.  Open the **Image Detection** tab.
2.  Upload a JPG, JPEG, or PNG image.
3.  Click **Analyze Image**.
4.  Review the predicted class and FAKE/REAL scores.

### Video Detection

1.  Open the **Video Detection** tab.
2.  Upload a supported video.
3.  Click **Analyze Video**.
4.  Review the prediction, average scores, and number of sampled frames.

The displayed scores are softmax outputs and may not be calibrated
probabilities of authenticity.

## Limitations

-   The classifier learns visual patterns from its training examples; it
    does not inherently know whether an image was AI-generated.
-   Unseen generators and image categories may be misclassified.
-   Camera photos may differ from training data due to lighting, blur,
    resizing, compression, and camera processing.
-   The reported test metrics are not a benchmark for all AI-generated
    images.
-   Video analysis averages predictions across sampled frames and does
    not model temporal motion.
-   A confidence score is not proof of authenticity.
-   Treat the system as a research prototype, not as a forensic
    guarantee.

## Future Improvements

-   Build an external test set with unseen AI-generated images and
    genuine camera photographs.
-   Include multiple generators and categories such as people, animals,
    objects, buildings, nature, products, and artwork.
-   Track source and generator metadata and avoid source leakage across
    train/test splits.
-   Report results separately by dataset, generator, and image category.
-   Improve fake-image recall while monitoring real-image false
    positives.
-   Compare architectures and validate changes on a separate test set.
-   Add explainability and uncertainty messaging.
-   If temporal video detection is required, investigate a sequence
    model rather than describing frame averaging as temporal learning.

## Responsible Use

Use model predictions as one signal for further review, not as the sole
basis for declaring media authentic or accusing someone of manipulation.
Preserve source information and obtain consent where appropriate when
testing personal images or videos.
