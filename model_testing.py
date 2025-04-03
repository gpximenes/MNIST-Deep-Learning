import tkinter as tk
from tkinter import *
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageOps

# Load the pre-trained MNIST model
model = load_model("mnist_model.h5")

class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Digit Recognition")
        self.resizable(False, False)

        self.canvas_width = 280
        self.canvas_height = 280

        # Create a Tkinter canvas widget with white background
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Create a PIL image to store the drawing (white background)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events to the canvas for drawing
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.last_x, self.last_y = None, None

        # Create Clear and Predict buttons
        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.clear_button.grid(row=1, column=0, pady=10)
        self.predict_button = tk.Button(self, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=1, pady=10)

        # Label to show the prediction result
        self.result_label = tk.Label(self, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

        # Add a checkbox to toggle auto-erase functionality
        self.auto_erase_var = tk.BooleanVar(value=True)
        self.auto_erase_checkbox = tk.Checkbutton(self, text="Auto Erase", variable=self.auto_erase_var)
        self.auto_erase_checkbox.grid(row=3, column=0, columnspan=2, pady=5)

        self.PREDICITON_FINISHED = False

    def start_draw(self, event):
        if self.PREDICITON_FINISHED and self.auto_erase_var.get():
            self.clear()
            self.PREDICITON_FINISHED = False

        self.last_x, self.last_y = event.x, event.y

    def draw_lines(self, event):
        # Draw line on the Tkinter canvas widget
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=15, fill="black", capstyle=ROUND, smooth=True)
        # Draw line on the PIL image (used for processing)
        self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=15)
        self.last_x, self.last_y = event.x, event.y

    def clear(self):
        # Clear the canvas widget and the PIL image
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and click Predict")

    def predict(self):
        # Invert the image (converting from white background and black drawing to MNIST style)
        inverted_image = ImageOps.invert(self.image)
        # Resize to 28x28 as expected by the model
        resized_image = inverted_image.resize((28, 28), Image.ANTIALIAS)
        # Convert to numpy array
        image_array = np.array(resized_image)
        # Normalize the image
        image_array = image_array / 255.0
        # Reshape to match the model's input shape (1, 28, 28, 1)
        input_img = image_array.reshape(1, 28, 28, 1)
        
        # Get the prediction from the model
        prediction = model.predict(input_img)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        self.result_label.config(text=f"Predicted digit: {predicted_digit}")

        self.PREDICITON_FINISHED = True


if __name__ == "__main__":
    app = DigitRecognizer()
    app.mainloop()
