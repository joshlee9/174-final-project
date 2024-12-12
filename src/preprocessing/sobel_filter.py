import cv2
import numpy as np

def apply_sobel(image, ksize=3):
    """
    Apply Sobel filter to detect edges in both horizontal and vertical directions.

    Parameters:
    - image: Input image (grayscale or color).
    - ksize: Size of the extended Sobel kernel. Must be 1, 3, 5, or 7.

    Returns:
    - sobel_combined: Image with combined Sobel edges.
    - sobel_x: Horizontal edges.
    - sobel_y: Vertical edges.
    """
    if image is None:
        raise ValueError("Input image is None")

    # If color image, convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Sobel filter in x direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_x = cv2.convertScaleAbs(sobel_x)

    # Apply Sobel filter in y direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Combine the two Sobel images
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    return sobel_combined, sobel_x, sobel_y

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    img = cv2.imread('datasets/yaseen_panorama/sample.jpg')

    # Apply Sobel filter
    sobel_combined, sobel_x, sobel_y = apply_sobel(img, ksize=3)

    # Display the results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Sobel X')
    plt.imshow(sobel_x, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Sobel Y')
    plt.imshow(sobel_y, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Sobel Combined')
    plt.imshow(sobel_combined, cmap='gray')
    plt.axis('off')

    plt.show()
