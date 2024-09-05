import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load images (assuming grayscale for simplicity)
gt_image = cv2.imread('ground_truth.png', cv2.IMREAD_GRAYSCALE)
inpainted_image = cv2.imread('inpainted.png', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)  # Mask should be binary (0 and 1)

# Apply the mask to focus only on the inpainted regions
gt_region = gt_image[mask == 1]
inpainted_region = inpainted_image[mask == 1]

# Calculate SSIM for the inpainted region
ssim_value = ssim(gt_region, inpainted_region, data_range=gt_image.max() - gt_image.min())

print(f'SSIM for the inpainted region: {ssim_value}')


# Load color images
gt_image_color = cv2.imread('ground_truth.png')
inpainted_image_color = cv2.imread('inpainted.png')
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

# Split the channels
gt_r, gt_g, gt_b = cv2.split(gt_image_color)
inpainted_r, inpainted_g, inpainted_b = cv2.split(inpainted_image_color)

# Calculate SSIM for each channel in the masked region
ssim_r = ssim(gt_r[mask == 1], inpainted_r[mask == 1], data_range=255)
ssim_g = ssim(gt_g[mask == 1], inpainted_g[mask == 1], data_range=255)
ssim_b = ssim(gt_b[mask == 1], inpainted_b[mask == 1], data_range=255)

# Average SSIM across channels
ssim_value = (ssim_r + ssim_g + ssim_b) / 3
print(f'SSIM for the inpainted region (color image): {ssim_value}')
