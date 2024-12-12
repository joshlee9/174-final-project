import cv2
import numpy as np

def apply_non_local_means_denoising(image, h=10, template_window_size=7, search_window_size=21):
    """
    Apply Non-Local Means Denoising to reduce noise while preserving details.

    Parameters:
    - image: Input image (grayscale or color).
    - h: Parameter regulating filter strength. Higher h removes more noise but may remove details.
    - template_window_size: Size of the template patch used to compute weights.
    - search_window_size: Size of the window used to compute weighted average.

    Returns:
    - denoised: Denoised image.
    """
    if image is None:
        raise ValueError("Input image is None")

    # If color image, convert to RGB because OpenCV's denoising expects it
    if len(image.shape) == 3 and image.shape[2] == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)

    return denoised

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load a noisy example image
    img = cv2.imread('datasets/yaseen_panorama/noisy_sample.jpg')

    # Apply Non-Local Means Denoising
    denoised = apply_non_local_means_denoising(img, h=12, template_window_size=7, search_window_size=21)

    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Noisy Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Denoised Image')
    plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
