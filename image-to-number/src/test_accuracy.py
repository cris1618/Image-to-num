from utils import calculate_digit_level_accuracy
import pandas as pd

# Import labels and predictions
file_path_labels = "/home/uycdcdycdgycdydc/Image-to-num/image-to-number/data/training/labels.csv"
file_path_pred = "/home/uycdcdycdgycdydc/Image-to-num/image-to-number/predictions.csv"

# Test accuracy
digits_accuracy = calculate_digit_level_accuracy(file_path_labels, file_path_pred, nrows=499)
