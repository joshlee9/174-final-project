import cv2
import numpy as np

def apply_median_filter(image, kernel_size=5):
    """
    Apply Median Filter to reduce noise while preserving edges.

    Parameters:
    - image: Input image (grayscale or color).
    - kernel_size: Size of the median filter kernel. Must be odd and greater than 1.

    Returns:
    - median_filtered: Median filtered image.
    """
    if image is None:
        raise ValueError("Input image is None")

    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd number greater than or equal to 3")

    median_filtered = cv2.medianBlur(image, ksize=kernel_size)

    return median_filtered

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    img = cv2.imread('datasets/yaseen_panorama/sample.jpg')

    # Apply Median Filter
    median = apply_median_filter(img, kernel_size=7)

    # Display the results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Median Filtered')
    plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
