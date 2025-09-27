# EfficientNetV2-Based Insect Species Classifier with Streamlit UI

This repository provides an end-to-end deep learning solution for insect image classification using PyTorch and EfficientNetV2. It includes scripts for training and evaluating the model, as well as a Streamlit web app for interactive testing and prediction.

## Features

- **EfficientNetV2 Model:** State-of-the-art architecture for image classification.
- **PyTorch Training Pipeline:** Easily train and evaluate on your own insect dataset.
- **Streamlit Web App:** Upload images and get predictions with confidence scores in your browser.
- **Supports 5 Insect Classes:** Easily customizable for more classes.
- **GPU Support:** Automatically detects and utilizes available GPU.

## Project Structure

```
CNN/
├── app.py                      # Streamlit web app for predictions
├── dataset/
│   ├── classes.txt             # List of class labels and names
│   ├── train.txt               # Training data file
│   ├── val.txt                 # Validation data file
│   ├── test.txt                # Test data file
│   ├── train/                  # Training images (organized by class)
│   ├── val/                    # Validation images (organized by class)
│   ├── test/                   # Test images (organized by class)
│   └── train_efficientnetv2.py # Model training script
├── efficientnetv2s_best.pth    # Saved model weights (after training)
├── venv/                       # Python virtual environment
└── requirements.txt            # Python dependencies (optional)
```

## Setup Instructions

1. **Clone the repository:**

   ```powershell
   git clone https://github.com/kartikahlawat/Deep-Learning-Insect-Classifier-EfficientNetV2-Streamlit.git
   cd Deep-Learning-Insect-Classifier-EfficientNetV2-Streamlit
   ```

2. **Create and activate a virtual environment:**

   ```powershell
   python -m venv venv
   & .\venv\Scripts\Activate
   ```

3. **Install dependencies:**

   ```powershell
   pip install torch torchvision timm albumentations opencv-python pandas tqdm streamlit
   ```

4. **Prepare your dataset:**
   - Place your images in the `dataset/train`, `dataset/val`, and `dataset/test` folders, organized by class.
   - Update `train.txt`, `val.txt`, and `test.txt` with image paths and labels.
   - Edit `classes.txt` to list your class labels and names.

## Training the Model

Run the training script to train EfficientNetV2 on your dataset:

```powershell
python dataset/train_efficientnetv2.py
```

Model weights will be saved as `efficientnetv2s_best.pth` after each improvement in validation loss.

## Running the Streamlit App

Launch the web app to test images and view predictions:

```powershell
streamlit run app.py
```

- Upload an image to see the predicted class, class number, and confidence scores for all classes.

## Customization

- **Number of Classes:** Edit `classes.txt` and update the training script as needed.
- **Model Architecture:** You can change the model in `train_efficientnetv2.py` to use other EfficientNetV2 variants or architectures supported by `timm`.
- **Augmentations:** Modify the `train_transform` and `val_transform` in the training script for different image preprocessing.

## Troubleshooting

- **GPU Not Detected:** Ensure you have CUDA-compatible hardware and the correct PyTorch version.
- **Albumentations Warning:** The code disables version checks to avoid warnings when offline.
- **Git Push Issues:** If you see push errors, run `git pull origin main --rebase` and resolve any conflicts before pushing again.

## License

This project is released under the MIT License.
EfficientNetV2-Based Insect Species Classifier with Streamlit UI
This repository provides an end-to-end deep learning solution for insect image classification using PyTorch and EfficientNetV2. It includes scripts for training and evaluating the model, as well as a Streamlit web app for interactive testing and prediction.

Features
EfficientNetV2 Model: State-of-the-art architecture for image classification.
PyTorch Training Pipeline: Easily train and evaluate on your own insect dataset.
Streamlit Web App: Upload images and get predictions with confidence scores in your browser.
Supports 5 Insect Classes: Easily customizable for more classes.
GPU Support: Automatically detects and utilizes available GPU.
Project Structure
Setup Instructions
Clone the repository:

Create and activate a virtual environment:

Install dependencies:

Prepare your dataset:

Place your images in the train, val, and test folders, organized by class.
Update train.txt, val.txt, and test.txt with image paths and labels.
Edit classes.txt to list your class labels and names.
Training the Model
Run the training script to train EfficientNetV2 on your dataset:

Model weights will be saved as efficientnetv2s_best.pth after each improvement in validation loss.

Running the Streamlit App
Launch the web app to test images and view predictions:

Upload an image to see the predicted class, class number, and confidence scores for all classes.
Customization
Number of Classes: Edit classes.txt and update the training script as needed.
Model Architecture: You can change the model in train_efficientnetv2.py to use other EfficientNetV2 variants or architectures supported by timm.
Augmentations: Modify the train_transform and val_transform in the training script for different image preprocessing.
Troubleshooting
GPU Not Detected: Ensure you have CUDA-compatible hardware and the correct PyTorch version.
Albumentations Warning: The code disables version checks to avoid warnings when offline.
Git Push Issues: If you see push errors, run git pull origin main --rebase and resolve any conflicts before pushing again.
License
This project is released under the MIT License.

Feel free to modify this README to better fit your dataset, workflow, or audience!
