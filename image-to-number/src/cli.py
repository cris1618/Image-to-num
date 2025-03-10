import argparse
from utils import preprocess_image
#from model import get_model
import torch
from torchvision import transforms
import os
import easyocr
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Digit Recognition CLI")
    parser.add_argument("images", nargs="+", help="Paths to the input images")
    args = parser.parse_args()

    reader = easyocr.Reader(["en"], gpu=False)

    for img_path in args.images:
        # Preprocess the image
        pil_image = preprocess_image(img_path)
        image_np = np.array(pil_image)

        # run easyocr
        results = reader.readtext(image_np, detail=1, paragraph=False)

        recognized_text = ""

        for (bbox, text, conf) in results:
            if conf > 0.5:
                filtered_text = "".join(ch for ch in text if ch.isdigit())
                recognized_text += filtered_text
                print(f"Number detected: {filtered_text}, Confidence: {conf:.3f}")
            else:
                recognized_text += "X"
            
        if not recognized_text:
            recognized_text = "X"
    
        
        print(f"{os.path.basename(img_path)} {recognized_text}")


if __name__ == "__main__":
    main()
