import argparse
import os
import gc
import numpy as np
import pytesseract
from utils import preprocess_image

def extract_digits(pil_image, conf_thresh=0.0):
    # Configure to only detect digits
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    #text = pytesseract.image_to_string(pil_image, config=config) 
    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
    #digits = "".join(ch for ch in text if ch.isdigit())

    detections = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            try:
                conf = float(data["conf"][i])
            except ValueError:
                conf = 0.0
            # Filter to keep only digits.
            filtered_text = "".join(ch for ch in text if ch.isdigit())
            if filtered_text:
                left_coord = data["left"][i]
                detections.append({"left": left_coord, "text": filtered_text, "conf": conf})
    
    # Sort detections by the left coordinate.
    detections = sorted(detections, key=lambda x: x["left"])
    
    final_digits = []
    for d in detections:
        # If confidence is low, replace each digit in this detection with 'X'.
        if d["conf"] < conf_thresh:
            final_digits.extend(["X"] * len(d["text"]))
        else:
            final_digits.extend(list(d["text"]))
    
    # Ensure exactly 6 characters
    if len(final_digits) < 6:
        final_digits.extend(["X"] * (6 - len(final_digits)))
    elif len(final_digits) > 6:
        final_digits = final_digits[:6]
    
    return "".join(final_digits)


    

def main():
    parser = argparse.ArgumentParser(description="Digit Recognition CLI using Tesseract OCR")
    parser.add_argument("images", nargs="+", help="Paths to the input images")
    args = parser.parse_args()
    

    for img_path in args.images:
        # Preprocess the image
        pil_image = preprocess_image(img_path)

        # Extracts the digits
        recognized_text = extract_digits(pil_image)

        # If not recognized put an X
        #if not recognized_text:
            #recognized_text = "X"

        print(f"{os.path.basename(img_path)} {recognized_text}")

        # Collect garbage
        gc.collect()

if __name__ == "__main__":
    main()


