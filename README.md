# MNIST Deep Learning Digit Recognizer

This repository contains a deep learning project for recognizing handwritten digits using the MNIST dataset. It includes a trained convolutional neural network (CNN) model, a Tkinter-based GUI application for testing the model, and Jupyter notebooks for training and testing the model.

## Repository Structure

```
MNIST_deep.py              # Script for training the CNN model
mnist_model.h5             # Pre-trained MNIST model file
model_testing.py           # Tkinter-based GUI application for digit recognition
model_training.ipynb       # Jupyter notebook for training the CNN model
```

## Features

- **Model Training**: Train a CNN on the MNIST dataset using `model_training.ipynb` or `MNIST_deep.py`.
- **Model Testing**: Test the trained model using:
  - A Tkinter-based GUI application (`model_testing.py`).
- **Pre-trained Model**: Use the pre-trained model (`mnist_model.h5`) for predictions.
- **Auto-Erase Functionality**: The GUI includes a checkbox to toggle the auto-erase feature, which clears the canvas automatically after a prediction.

## Requirements

To run the code in this repository, you need the following dependencies:

- Python 3.7+
- TensorFlow
- NumPy
- scikit-image
- Pillow
- Matplotlib
- Tkinter (comes pre-installed with Python)

Install the required packages using pip:

```bash
pip install tensorflow numpy scikit-image pillow matplotlib ipycanvas
```

## Usage

### 1. Train the Model (Optional)
To train the model, use the `model_training.ipynb` notebook or the `MNIST_deep.py` script. The trained model will be saved as `mnist_model.h5`.

```bash
python MNIST_deep.py
```

### 2. Test the Model with Tkinter GUI
Run the `model_testing.py` script to launch a GUI application where you can draw digits and get predictions.

```bash
python model_testing.py
```


## How It Works

1. **Model Architecture**: The CNN model consists of convolutional layers, max-pooling layers, a dense layer, and a dropout layer for regularization.
2. **Input Preprocessing**: The input image is resized to 28x28 pixels, normalized to [0, 1], and reshaped to match the model's input shape.
3. **Prediction**: The model predicts the digit by outputting probabilities for each class (0-9), and the class with the highest probability is selected.

## Tkinter GUI Features

The `model_testing.py` script provides a graphical user interface (GUI) for testing the model. The features include:

- **Canvas**: A 280x280 pixel canvas where users can draw digits.
- **Clear Button**: Clears the canvas for a new drawing.
- **Predict Button**: Predicts the digit drawn on the canvas.
- **Auto-Erase Checkbox**: Toggles the functionality to automatically clear the canvas after a prediction.

### Example Workflow

1. Draw a digit on the canvas.
2. Click the "Predict" button to see the predicted digit.
3. If "Auto Erase" is enabled, the canvas will clear automatically after the prediction.

## Screenshots

### Tkinter GUI

![image](https://github.com/user-attachments/assets/e5162277-3b60-4b48-94f4-3dceda845afe)


![image](https://github.com/user-attachments/assets/1c2a5a70-4eea-4da7-b42a-34ce54ec294a)


![image](https://github.com/user-attachments/assets/0d3014f4-01fa-4732-a9f5-93fc9b7dda68)

