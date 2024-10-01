import os
from flask import Flask, request, render_template, jsonify
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load YOLOv5 model for image detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Route to render HTML
@app.route('/')
def index():
    return render_template('index.html')

# Function to detect the product using YOLOv5
def detect_product(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    detected_items = results.pandas().xyxy[0]
    detected_product = detected_items['name'][0] if not detected_items.empty else 'Unknown'
    return detected_product

# Function to generate a prompt
def generate_prompt(detected_product, event, promo_details):
    prompt = f" Get {promo_details} off on {detected_product}!"
    return prompt

# Function to generate a banner by overlaying text on the uploaded image
def generate_banner_on_image(image_path, prompt, output_path="static/generated_banner.png"):
    # Open the uploaded image
    image = Image.open(image_path)
    
    # Initialize drawing on the image
    draw = ImageDraw.Draw(image)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except IOError:
        font = ImageFont.load_default()

    # Define text position and color
    text_position = text_position = (50, 50)   # Adjust this to position the text at the bottom
    text_color = (255, 255, 255)  # White text (change if needed)

    # Add a shadow (optional)
    shadow_offset = (3, 3)
    draw.text((text_position[0] + shadow_offset[0], text_position[1] + shadow_offset[1]), prompt, font=font, fill=(0, 0, 0))  # Black shadow

    # Add text on the image
    draw.text(text_position, prompt, font=font, fill=text_color)

    # Save the resulting image with the text overlay
    image.save(output_path)
    
    return output_path

# API route for generating banner
@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    event = request.form['event']
    promo = request.form['promo']

    # Save uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Detect product in the image
    detected_product = detect_product(image_path)

    # Generate prompt based on detection and inputs
    prompt = generate_prompt(detected_product, event, promo)

    # Generate banner on the provided image
    banner_path = generate_banner_on_image(image_path, prompt)

    return jsonify({"bannerUrl": "/" + banner_path})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
