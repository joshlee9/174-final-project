import cv2
import numpy as np

def apply_laplacian_filter(image, ksize=3, scale=1, delta=0):
    """
    Apply Laplacian filter to detect edges and fine details.

    Parameters:
    - image: Input image (grayscale or color).
    - ksize: Aperture size used to compute the second-derivative filters. Must be positive and odd.
    - scale: Optional scale factor for the computed Laplacian values.
    - delta: Optional delta value added to the results.

    Returns:
    - laplacian: Image after applying the Laplacian filter.
    """
    if image is None:
        raise ValueError("Input image is None")

    # If color image, convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
    laplacian = cv2.convertScaleAbs(laplacian)

    return laplacian

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    img = cv2.imread('datasets/yaseen_panorama/sample.jpg')

    # Apply Laplacian Filter
    laplacian = apply_laplacian_filter(img, ksize=3, scale=1, delta=0)

    # Display the results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Laplacian Filtered')
    plt.imshow(laplacian, cmap='gray')
    plt.axis('off')

    plt.show()
