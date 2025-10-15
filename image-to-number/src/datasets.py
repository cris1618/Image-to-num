import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import preprocess_image

# Create the class to import the data in a suitable format
class DigitDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform 
        self.image_files = []
        self.labels = {}

        # Read the csv
        with open(label_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                label = row["label"].strip()

                if label:
                    self.image_files.append(filename)
                    self.labels[filename] = label
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.root_dir, filename)
        # preprocess the image
        pil_image = preprocess_image(img_path)
        if self.transform:
            pil_image = self.transform(pil_image)
        # Extracting labels
        label_str = self.labels[filename]
        label = [int(digit) for digit in label_str]

        return pil_image, label