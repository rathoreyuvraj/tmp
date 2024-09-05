import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import lpips

# Function to calculate PSNR for color images
def calculate_psnr(gt_image, colorized_image):
    mse = np.mean((gt_image - colorized_image) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# Function to calculate SSIM for color images
def calculate_ssim(gt_image, colorized_image):
    # Split into RGB channels
    ssim_r = ssim(gt_image[:, :, 2], colorized_image[:, :, 2], data_range=gt_image.max() - gt_image.min())
    ssim_g = ssim(gt_image[:, :, 1], colorized_image[:, :, 1], data_range=gt_image.max() - gt_image.min())
    ssim_b = ssim(gt_image[:, :, 0], colorized_image[:, :, 0], data_range=gt_image.max() - gt_image.min())
    
    # Average SSIM across channels
    ssim_value = (ssim_r + ssim_g + ssim_b) / 3
    return ssim_value

# Function to convert images to tensors for LPIPS
def convert_to_tensor(image):
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 * 2 - 1  # Normalize to [-1, 1]
    return image_tensor.unsqueeze(0)  # Add batch dimension

# Load the ground truth and colorized image
gt_image = cv2.imread('ground_truth.png')  # Ground truth image (color)
colorized_image = cv2.imread('colorized_image.png')  # Colorized image

# Ensure images have the same size
assert gt_image.shape == colorized_image.shape, "Images must have the same dimensions."

# Calculate PSNR
psnr_value = calculate_psnr(gt_image, colorized_image)
print(f'PSNR for the colorized image: {psnr_value} dB')

# Calculate SSIM
ssim_value = calculate_ssim(gt_image, colorized_image)
print(f'SSIM for the colorized image: {ssim_value}')

# Load LPIPS model (you can use 'alex', 'vgg', or 'squeeze')
lpips_model = lpips.LPIPS(net='alex')

# Convert the images to tensors
gt_tensor = convert_to_tensor(gt_image)
colorized_tensor = convert_to_tensor(colorized_image)

# Calculate LPIPS
lpips_value = lpips_model(gt_tensor, colorized_tensor)
print(f'LPIPS for the colorized image: {lpips_value.item()}')
