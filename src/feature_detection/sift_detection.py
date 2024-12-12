import cv2
import numpy as np

def detect_and_compute_sift(image, n_features=0):
    """
    Detect keypoints and compute SIFT descriptors for an image.

    Parameters:
    - image: Input image (grayscale or color).
    - n_features: The number of best features to retain. If 0, all are retained.

    Returns:
    - keypoints: Detected keypoints.
    - descriptors: SIFT descriptors.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=n_features)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors

def draw_keypoints(image, keypoints):
    """
    Draw keypoints on the image.

    Parameters:
    - image: Input image (grayscale or color).
    - keypoints: Detected keypoints.

    Returns:
    - image_with_keypoints: Image with keypoints drawn.
    """
    image_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return image_with_keypoints

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    img = cv2.imread('datasets/yaseen_panorama/sample.jpg')

    # Detect SIFT features
    keypoints, descriptors = detect_and_compute_sift(img, n_features=500)

    print(f"Number of SIFT keypoints detected: {len(keypoints)}")

    # Draw keypoints on the image
    img_with_keypoints = draw_keypoints(img, keypoints)

    # Display the results
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Keypoints')
    plt.axis('off')
    plt.show()
