import cv2
import numpy as np

def apply_bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply Bilateral Filter to smooth the image while preserving edges.

    Parameters:
    - image: Input image (grayscale or color).
    - diameter: Diameter of each pixel neighborhood.
    - sigma_color: Filter sigma in the color space.
    - sigma_space: Filter sigma in the coordinate space.

    Returns:
    - filtered: Bilateral filtered image.
    """
    if image is None:
        raise ValueError("Input image is None")

    filtered = cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    return filtered

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    img = cv2.imread('datasets/yaseen_panorama/sample.jpg')

    # Apply Bilateral Filter
    bilateral = apply_bilateral_filter(img, diameter=15, sigma_color=100, sigma_space=100)

    # Display the results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Bilateral Filtered')
    plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
