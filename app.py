import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the trained SRCNN model
model = tf.keras.models.load_model("srcnn_model.h5")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Apply the SRCNN model to the uploaded low-resolution image."""
    # Load image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image for consistency
    target_size = (256, 256)
    img = cv2.resize(img, target_size)
    
    # Simulate low resolution by downscaling and upscaling
    scale = 2
    lr = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
    lr = cv2.resize(lr, target_size)
    
    # Normalize image
    lr_norm = lr / 255.0

    # Predict using SRCNN model
    sr = model.predict(np.expand_dims(lr_norm, axis=0))[0]

    # Convert back to 8-bit format for saving
    sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)

    # Save output image
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    Image.fromarray(sr).save(result_path)

    return result_path

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        
        file = request.files["file"]
        
        if file.filename == "":
            return "No selected file"
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Process image using SRCNN
            result_path = process_image(filepath)
            
            return render_template("result.html", 
                                   original=filepath, 
                                   result=result_path)
    
    return render_template("index.html")

@app.route("/static/<path:filename>")
def send_file(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)
