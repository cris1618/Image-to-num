import torch 
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from datasets import DigitDataset
from model import get_model

def train_crnn(batch_size = 32, lr = 0.001, num_epochs = 100):
    # transformation
    transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

    # directories
    root_dir = "image-to-number/data/training/tdset/tdset"
    label_file = "image-to-number/data/training/labels.csv"

    # Import the data
    full_dataset = DigitDataset(root_dir=root_dir, label_file=label_file, transform=transform)

    # Split them
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True)

    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True)

    # Import the model (Pretrained for now)
    model = get_model()
    model.train()

    criterion = torch.nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop 
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            output = output.permute(1, 0, 2)

            batch_size = output.size(1)
            # Flatten the labels into a 1D tensor
            target_sequnces = torch.tensor([digit for label in labels for digit in label], dtype=torch.long)

            # Target length of 6, as the number of ch in each image
            target_lengths = torch.full((batch_size,), 6, dtype=torch.long)
            input_lengths = torch.full((batch_size,), output.size(0), dtype=torch.long)

            loss = criterion(output, target_sequnces, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "experiments/checkpoints/crnn_finetuned.pth")

if __name__ == "__main__":
    train_crnn()