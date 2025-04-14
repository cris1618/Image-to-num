import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2
import numpy as np
import csv
from itertools import islice

def calculate_digit_level_accuracy(labels, predictions, nrows=99):
    predictions_dict = {}
    with open(predictions, newline="") as pred_file:
        reader = csv.DictReader(pred_file)
        for row in reader:
            predictions_dict[row["image"]] = row["recognized_text"]
    
    total_digits = 0
    correct_digits = 0

    with open(labels, newline="") as labels_file:
        reader = csv.DictReader(labels_file)
        for row in islice(reader, nrows):
            image_name = row["filename"]
            true_value = row["label"]
            predicted_value = predictions_dict.get(image_name, "")

            for label, pred in zip(true_value, predicted_value):
                total_digits += 1
                if label == pred:
                    correct_digits += 1
            
            total_digits += abs(len(true_value) - len(predicted_value))
    
    digit_accuracy = correct_digits / total_digits if total_digits > 0 else 0
    print(f"Digit-Level Accuracy: {digit_accuracy:.2%}")
    return digit_accuracy

def auto_crop_image(img_path, output_path=None):
    # Load the image in Grayscale
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize image for speed
    base_width = 1500 # 1500 optimal
    height, width = image.shape
    w_percent = (base_width / float(width))
    h_size = int((float(height)* float(w_percent)))
    image = cv2.resize(image, (base_width, h_size))

    # Apply Gaussian blur, reduce noise and smoothen image
    blurred = cv2.GaussianBlur(image, (5,5), 0)

    # Use adaptive thresholding to handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred,
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11,  # Block size; must be odd
        2    # Constant subtracted from mean
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return Image.fromarray(image)
    
    # Filter the contours, remove very small regions
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    if not valid_contours:
        return Image.fromarray(image)
    
    # Get bounding boxes around all valid contours
    x_min = np.inf
    y_min = np.inf
    x_max = -np.inf
    y_max = -np.inf
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]

    if output_path:
        cv2.imwrite(output_path, cropped)
    
    return Image.fromarray(cropped)

def preprocess_image(input_path, output_path=None):
    # First open the image
    img = Image.open(input_path).convert("RGB")

    # Smaller resolution (This is to run it on my CPU faster, but on a GPU this part of the code 
    # could be removed and the application should work better with a better image resolution)
    base_width = 800 # 800 optimal
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1])* float(w_percent)))
    img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
    
    # Crop it
    width, height = img.size
    left = int(0.033 * width)
    top = int(0.4 * height)
    right = int(0.75 * width)
    bottom = height
    cropped = img.crop((left, top, right, bottom))

    # Convert grayscale
    gray = cropped.convert("L")
    #gray = img.convert("L")

    # Invert the colors
    inverted = ImageOps.invert(gray)

    # aplly gaussian
    blurred = inverted.filter(ImageFilter.GaussianBlur(3.5)) # 3.5 optimal

    # Increase contrast
    enhancer = ImageEnhance.Contrast(blurred)
    processed = enhancer.enhance(10) # 10 optimal

    if output_path:
        processed.save(output_path)
    
    return processed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Not enough arguments")
        sys.exit(1)
    
    input_img = sys.argv[1]
    output_img = None
    if len(sys.argv) > 2:
        output_img = sys.argv[2]
    
    processed_img = preprocess_image(input_img, output_img)
    print("Processing Complete.")
