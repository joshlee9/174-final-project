import cv2
import numpy as np

def detect_and_compute_orb(image, n_features=500, scale_factor=1.2, n_levels=8):
    """
    Detect keypoints and compute ORB descriptors for an image.

    Parameters:
    - image: Input image (grayscale or color).
    - n_features: Maximum number of features to retain.
    - scale_factor: Pyramid decimation ratio.
    - n_levels: Number of pyramid levels.

    Returns:
    - keypoints: Detected keypoints.
    - descriptors: ORB descriptors.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

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

    # Detect ORB features
    keypoints, descriptors = detect_and_compute_orb(img, n_features=1000)

    print(f"Number of ORB keypoints detected: {len(keypoints)}")

    # Draw keypoints on the image
    img_with_keypoints = draw_keypoints(img, keypoints)

    # Display the results
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('ORB Keypoints')
    plt.axis('off')
    plt.show()
