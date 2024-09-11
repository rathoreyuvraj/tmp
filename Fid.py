import torch
import pyiqa
from PIL import Image
from torchvision import transforms
import os

# Helper function to load and transform images into tensors
def load_images_from_folder(folder, transform):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = Image.open(img_path).convert('RGB')  # Load image and convert to RGB
        image_tensor = transform(image)
        images.append(image_tensor)
    return torch.stack(images)  # Stack all images into a single tensor

# Define transform to convert images to tensor
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to match Inception model input size
    transforms.ToTensor(),
])

# Load real and generated images
real_images_folder = 'path_to_real_images'
generated_images_folder = 'path_to_generated_images'

real_images = load_images_from_folder(real_images_folder, transform)
generated_images = load_images_from_folder(generated_images_folder, transform)

# Add batch dimension and scale image values to [0, 1]
real_images = real_images.unsqueeze(0).float()
generated_images = generated_images.unsqueeze(0).float()

# Initialize FID metric from pyiqa
fid_model = pyiqa.create_metric('fid')

# Compute FID score between real and generated images
fid_score = fid_model(generated_images, real_images)
print(f"FID Score: {fid_score.item()}")
