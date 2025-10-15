import argparse
from utils import preprocess_image
import pandas as pd
#from model import get_model
import torch
from torchvision import transforms
import os
import easyocr
import numpy as np
import gc

def main():
    parser = argparse.ArgumentParser(description="Digit Recognition CLI")
    parser.add_argument("images", nargs="+", help="Paths to the input images")
    #parser.add_argument("--eval_csv", type=str, default=None, help="Path to CSV with ground-truth labels")
    args = parser.parse_args()
    
    reader = easyocr.Reader(["en"], gpu=False)

    #results_dict = {}

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
        
        #filename = os.path.basename(img_path)
        #results_dict[filename] = recognized_text
    
        print(f"{os.path.basename(img_path)} {recognized_text}")

        # Collect garbage manually
        gc.collect()
    
  


if __name__ == "__main__":
    main()
    """ if args.eval_csv:
        df_truth = pd.read_csv(args.eval_csv)
        df_results = pd.DataFrame(list(results_dict.items()), columns=["filename", "predicted"])
        # Merge on filename
        df_eval = pd.merge(df_truth, df_results, on="filename", how="inner")
        
        # Calculate accuracy (exact match for the 6-digit string)
        df_eval["correct"] = df_eval.apply(lambda row: row["predicted"] == row["label"], axis=1)
        accuracy = df_eval["correct"].mean() * 100
        print(f"\nEvaluation Accuracy: {accuracy:.2f}%")
        
        # Print mismatches for further inspection
        mismatches = df_eval[df_eval["correct"] == False]
        if not mismatches.empty:
            print("\nMismatches:")
            print(mismatches)"""  
