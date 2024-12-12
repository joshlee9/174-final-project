import cv2
import numpy as np

def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian Blur to smooth the image.

    Parameters:
    - image: Input image (grayscale or color).
    - kernel_size: Size of the Gaussian kernel (width, height). Must be positive and odd.
    - sigma: Standard deviation in X and Y directions.

    Returns:
    - blurred: Blurred image.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Validate kernel size
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        raise ValueError("Kernel size must be odd numbers")

    blurred = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)

    return blurred

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    img = cv2.imread('datasets/yaseen_panorama/sample.jpg')

    # Apply Gaussian Blur
    blurred = apply_gaussian_filter(img, kernel_size=(7, 7), sigma=2.0)

    # Display the results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gaussian Blurred')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
