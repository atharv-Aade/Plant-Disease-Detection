import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from torchvision.transforms import ToTensor

# Define the classes
CLASSES = ['healthy','infected']
NUM_CLASSES = len(CLASSES)

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Get data from local directory
data_dir = Path("data")
train_dataset = datasets.ImageFolder(root=data_dir / "train", transform=transform)
dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: 3 channels (RGB), 224x224
            nn.Conv2d(3, 32, (3,3)), # Output: 32x222x222
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 32x111x111
            nn.Conv2d(32, 64, (3,3)), # Output: 64x109x109
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 64x54x54
            nn.Conv2d(64, 128, (3,3)), # Output: 128x52x52
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 128x26x26
            nn.Flatten(), 
            nn.Dropout(0.5),
            nn.Linear(128 * 26 * 26, 512),  # Correct input size based on previous layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x): 
        return self.model(x)

# Create models directory if it doesn't exist
Path("models").mkdir(exist_ok=True)

# Instance of the neural network, loss, optimizer 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__":
    # For inference
    try:
        with open("models/PlantDiseaseAlgo.pt", "rb") as f:
            clf.load_state_dict(load(f))
        
        # Make sure to use the same transform as training
        def predict_image(image_path):
            clf.eval()  # Set to evaluation mode
            img = Image.open(image_path)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = clf(img_tensor)
                prediction = torch.argmax(output).item()
                
            return CLASSES[prediction]

        # Example usage:
        prediction = predict_image('WIN_20241222_08_23_55_Pro.jpg')
        print(f"Predicted class: {prediction}")

    except FileNotFoundError:
        print("No saved model found. Starting training...")
        # Training code
        for epoch in range(10):
            running_loss = 0.0
            for batch_idx, (X, y) in enumerate(dataset): 
                X, y = X.to(device), y.to(device) 
                
                # Forward pass
                yhat = clf(X) 
                loss = loss_fn(yhat, y) 
                
                # Backward pass
                opt.zero_grad()
                loss.backward() 
                opt.step() 
                
                running_loss += loss.item()
                if batch_idx % 10 == 9:
                    print(f'Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {running_loss/10:.3f}')
                    running_loss = 0.0

            print(f"Epoch: {epoch} completed")
        
        # Save the model
        model_path = Path("models") / "PlantDiseaseAlgo.pt"
        torch.save(clf.state_dict(), model_path)
        print(f"Model saved to {model_path}")