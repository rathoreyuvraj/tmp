import cv2
import numpy as np

# Function to calculate SSIM for RGB images
def ssim_color(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM for each color channel separately
    ssim_r = ssim_single_channel(img1[:, :, 2], img2[:, :, 2])
    ssim_g = ssim_single_channel(img1[:, :, 1], img2[:, :, 1])
    ssim_b = ssim_single_channel(img1[:, :, 0], img2[:, :, 0])
    
    # Return the mean SSIM across the channels
    return (ssim_r + ssim_g + ssim_b) / 3

# Helper function to compute SSIM for a single channel
def ssim_single_channel(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Mean of images
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    # Variance and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# Load your RGB images
gt_image = cv2.imread('ground_truth.png')
inpainted_image = cv2.imread('inpainted.png')

# Calculate SSIM for color images
ssim_value = ssim_color(gt_image, inpainted_image)
print(f'SSIM for RGB images: {ssim_value}')
