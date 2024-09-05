import numpy as np
import cv2

# Function to calculate PSNR
def calculate_psnr(gt_image, inpainted_image, mask):
    # Extract the inpainted region using the mask
    gt_region = gt_image[mask == 1]
    inpainted_region = inpainted_image[mask == 1]
    
    # Calculate MSE
    mse = np.mean((gt_region - inpainted_region) ** 2)
    if mse == 0:
        return float('inf')  # No difference, perfect reconstruction
    
    # Calculate PSNR
    max_pixel_value = 255.0  # Assuming 8-bit image
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr

# Example usage:
# Load images (assuming grayscale for simplicity, you can adapt for RGB)
gt_image = cv2.imread('ground_truth.png', cv2.IMREAD_GRAYSCALE)
inpainted_image = cv2.imread('inpainted.png', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)  # Mask should be binary (0 and 1)

# Calculate PSNR for inpainted regions
psnr_value = calculate_psnr(gt_image, inpainted_image, mask)
print(f'PSNR for the inpainted region: {psnr_value} dB')
