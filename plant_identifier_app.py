import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import torch
import torch.nn as nn

from plant_identifier import PlantDataset, test_folder, SimplePlantClassifier


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
plant_model = SimplePlantClassifier()

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
