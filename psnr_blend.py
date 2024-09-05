import numpy as np
import cv2

# Function to calculate PSNR
def calculate_psnr(blended_gt, inpainted_image, mask, alpha=0.5):
    # Recover the ground truth portion from the blended image using alpha
    gt_region = (blended_gt - (1 - alpha) * inpainted_image) / alpha
    
    # Ensure the mask is binary (1 for inpainted region, 0 for others)
    gt_inpainted_region = gt_region[mask == 1]
    inpainted_region = inpainted_image[mask == 1]
    
    # Calculate MSE
    mse = np.mean((gt_inpainted_region - inpainted_region) ** 2)
    if mse == 0:
        return float('inf')  # No difference, perfect reconstruction
    
    # Calculate PSNR
    max_pixel_value = 255.0  # Assuming 8-bit image
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr

# Example usage:
# Load images (assuming grayscale for simplicity, you can adapt for RGB)
blended_gt = cv2.imread('blended_gt.png', cv2.IMREAD_GRAYSCALE)
inpainted_image = cv2.imread('inpainted.png', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)  # Mask should be binary (0 and 1)

# Calculate PSNR for inpainted regions using blended GT
psnr_value = calculate_psnr(blended_gt, inpainted_image, mask, alpha=0.7)  # Assuming alpha=0.7
print(f'PSNR for the inpainted region with blended GT: {psnr_value} dB')
