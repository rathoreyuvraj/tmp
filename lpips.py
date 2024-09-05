import torch
import lpips
import cv2
import numpy as np

# Load LPIPS model (pretrained with AlexNet, VGG, or SqueezeNet)
lpips_model = lpips.LPIPS(net='alex')  # You can use 'alex', 'vgg', or 'squeeze'

# Load images
gt_image = cv2.imread('ground_truth.png')  # Ground truth image
inpainted_image = cv2.imread('inpainted.png')  # Inpainted image
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)  # Binary mask for inpainted region (1 for inpainted region, 0 otherwise)

# Convert images to tensors (and normalize them to [-1, 1] as required by LPIPS)
gt_tensor = torch.from_numpy(gt_image).permute(2, 0, 1).float() / 255.0 * 2 - 1  # [C, H, W]
inpainted_tensor = torch.from_numpy(inpainted_image).permute(2, 0, 1).float() / 255.0 * 2 - 1  # [C, H, W]

# Expand dimensions to [1, C, H, W] (batch size of 1)
gt_tensor = gt_tensor.unsqueeze(0)
inpainted_tensor = inpainted_tensor.unsqueeze(0)

# Apply mask to isolate the inpainted region
# Set non-inpainted regions to zero
gt_tensor_masked = gt_tensor.clone()
inpainted_tensor_masked = inpainted_tensor.clone()

# Convert mask to a 3-channel tensor and apply it
mask_tensor = torch.from_numpy(mask).float() / 255.0  # Mask is [H, W]
mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, H, W]
mask_tensor = torch.cat([mask_tensor] * 3, dim=1)  # Repeat mask across channels to get [1, 3, H, W]

# Apply the mask to ground truth and inpainted images
gt_tensor_masked = gt_tensor_masked * mask_tensor
inpainted_tensor_masked = inpainted_tensor_masked * mask_tensor

# Calculate LPIPS for the inpainted regions
lpips_value = lpips_model(gt_tensor_masked, inpainted_tensor_masked)
print(f'LPIPS value for the inpainted region: {lpips_value.item()}')
