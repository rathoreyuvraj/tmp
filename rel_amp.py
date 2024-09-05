import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Function to compute relative log amplitude
def relative_log_amplitude(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT to the image
    fft_image = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft_image)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fft_shift)
    
    # Convert to log amplitude
    log_amplitude = np.log(magnitude_spectrum + 1e-8)  # Adding a small value to avoid log(0)
    
    # Normalize by subtracting the max value to ensure negative relative log amplitudes
    relative_log_amp = log_amplitude - np.max(log_amplitude)
    
    return relative_log_amp

# Function to plot the relative log amplitudes of two images
def plot_relative_log_amplitudes(image1, image2):
    # Compute relative log amplitude for both images
    log_amp1 = relative_log_amplitude(image1)
    log_amp2 = relative_log_amplitude(image2)
    
    # Take the frequency spectrum of the center row
    freq_row1 = log_amp1[log_amp1.shape[0] // 2, :]
    freq_row2 = log_amp2[log_amp2.shape[0] // 2, :]
    
    # Frequencies corresponding to the Fourier transform
    frequencies = np.linspace(0, np.pi, len(freq_row1))
    
    # Plotting
    plt.figure(figsize=(10, 8))  # Elongate the y-axis
    plt.plot(frequencies, freq_row1, label='Image 1', color='blue', linewidth=2)
    plt.plot(frequencies, freq_row2, label='Image 2', color='red', linewidth=2)
    
    # Elongate Y-axis for better visibility of differences
    plt.ylim([-10, 0])
    
    # Labels and legend
    plt.xlabel('Frequency (radians)')
    plt.ylabel('Relative Log Amplitude')
    plt.title('Relative Log Amplitude for Two Images')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# Load the two images (replace 'image1_path' and 'image2_path' with actual paths)
image1_path = '/mnt/data/image1.png'  # Placeholder path
image2_path = '/mnt/data/image2.png'  # Placeholder path

# Open images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Plot the relative log amplitudes for the two images
plot_relative_log_amplitudes(image1, image2)
