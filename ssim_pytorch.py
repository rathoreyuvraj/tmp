import torch
import piq
import cv2

# Load images
gt_image = cv2.imread('ground_truth.png')
inpainted_image = cv2.imread('inpainted.png')

# Convert to PyTorch tensors (normalized to [0, 1] and shape [C, H, W])
gt_tensor = torch.from_numpy(gt_image).permute(2, 0, 1).float() / 255.0
inpainted_tensor = torch.from_numpy(inpainted_image).permute(2, 0, 1).float() / 255.0

# Calculate SSIM
ssim_value = piq.ssim(gt_tensor.unsqueeze(0), inpainted_tensor.unsqueeze(0), data_range=1.0)
print(f'SSIM for RGB images: {ssim_value.item()}')
