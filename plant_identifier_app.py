import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

# from plant_identifier import PlantDataset, test_folder, SimplePlantClassifier


# from .plant_identifier import SimplePlantClassifier

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


class SimplePlantClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(SimplePlantClassifier, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

class PlantDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

# Visualization
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


# Create an instance of the model
plant_model = SimplePlantClassifier(num_classes=3)

# Load the saved state dictionary into the model
plant_model.load_state_dict(torch.load('model.pth'))

# Move the model to the desired device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plant_model.to(device)
# The model is now ready to be used for inference
plant_model.eval()  # Set the model to evaluation mode
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
test_folder = './plants_dataset/test'
test_dataset = PlantDataset(test_folder, transform=image_transform)
with torch.no_grad():
    # Example usage
    test_images = glob('./plants_dataset/test/*/*')
    test_examples = np.random.choice(test_images, 10)
    for example in test_examples:
        original_image, image_tensor = preprocess_image(example, image_transform)
        probabilities = predict(plant_model, image_tensor, device)

        # Assuming dataset.classes gives the class names
        class_names = test_dataset.classes
        visualize_predictions(original_image, probabilities, class_names)
